# === Standard Libraries ===
import os
import glob
from datetime import datetime

# === Data Handling ===
import pandas as pd
import numpy as np

# === ML Tools ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# === Utilities ===
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Alpaca & Databento ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from databento import DBNStore


# === Configuration ===
API_KEY = ""
SECRET = ""


# === Data Extraction Functions ===

def dbn_to_df(file_path):
    """Convert a Databento DBN file to a DataFrame."""
    store = DBNStore.from_file(file_path)
    return store.to_df()


def extract_data(folder_path):
    os.makedirs("raw_csvs", exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".dbn"):
            print(f"Processing {filename}...")
            df = dbn_to_df(os.path.join(folder_path, filename))
            output = os.path.join("raw_csvs", f"{os.path.splitext(filename)[0]}.csv")
            df.to_csv(output, index=False)


def extract_symbol_from_csvs(input_folder, output_folder, symbol, chunksize=100_000):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{symbol}.csv")
    header_written = False

    with open(output_path, 'w', encoding='utf-8') as out_file:
        for filename in os.listdir(input_folder):
            if not filename.endswith(".csv"):
                continue

            try:
                for chunk in pd.read_csv(os.path.join(input_folder, filename), chunksize=chunksize):
                    if 'symbol' not in chunk.columns:
                        continue
                    symbol_chunk = chunk[chunk['symbol'] == symbol]
                    if not symbol_chunk.empty:
                        symbol_chunk.to_csv(out_file, header=not header_written, index=False, mode='a')
                        header_written = True
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"Filtered rows saved to: {output_path}")


def get_historical_data(api_key, secret_key, symbol, startdate, endDate, daily=False):
    client = StockHistoricalDataClient(api_key, secret_key)
    timeframe = TimeFrame.Day if daily else TimeFrame(5, TimeFrameUnit.Minute)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=startdate,
        end=endDate
    )
    bars = client.get_stock_bars(request)
    df = pd.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    return df


# === Feature Engineering ===

def build_footprint_with_engineered_features(df, N=50, output_dir=None):
    df = df.drop(columns=[
        'rtype', 'publisher_id', 'instrument_id', 'channel_id',
        'order_id', 'flags', 'ts_in_delta', 'sequence'
    ], errors='ignore')

    df = df[(df['action'] == 'T') & (df['side'].isin(['A', 'B']))].copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df['time_bin'] = df['ts_event'].dt.floor('5T')

    features_list = []

    for symbol, sym_df in df.groupby('symbol'):
        sym_df['date'] = sym_df['ts_event'].dt.date
        for date, day_df in sym_df.groupby('date'):
            min_price, max_price = day_df['price'].min(), day_df['price'].max()
            bins = np.linspace(min_price, max_price, N + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            for time_bin, group in day_df.groupby('time_bin'):
                ask = group[group['side'] == 'A'][['price', 'size']]
                bid = group[group['side'] == 'B'][['price', 'size']]

                if ask.empty and bid.empty:
                    continue

                ask_hist, _ = np.histogram(ask['price'], bins=bins, weights=ask['size'])
                bid_hist, _ = np.histogram(bid['price'], bins=bins, weights=bid['size'])

                row = {
                    'symbol': symbol,
                    'time_bin': time_bin,
                    'total_ask_volume': ask_hist.sum(),
                    'total_bid_volume': bid_hist.sum(),
                    'volume_imbalance': ask_hist.sum() - bid_hist.sum(),
                    'max_ask_bin': ask_hist.max(),
                    'max_bid_bin': bid_hist.max(),
                }

                for i in range(N):
                    row.update({
                        f'ask_bin_{i}': ask_hist[i],
                        f'bid_bin_{i}': bid_hist[i],
                        f'ask_price_{i}': bin_centers[i],
                        f'bid_price_{i}': bin_centers[i],
                        f'ask_dollar_vol_{i}': ask_hist[i] * bin_centers[i],
                        f'bid_dollar_vol_{i}': bid_hist[i] * bin_centers[i],
                    })

                features_list.append(row)

    footprint_df = pd.DataFrame(features_list)
    if footprint_df.empty:
        return pd.DataFrame()

    # Normalize
    cols = ['total_ask_volume', 'total_bid_volume', 'volume_imbalance', 'max_ask_bin', 'max_bid_bin']
    cols += [c for c in footprint_df.columns if 'dollar_vol' in c]
    scaler = MinMaxScaler()
    footprint_df[cols] = scaler.fit_transform(footprint_df[cols])
    footprint_df.dropna(inplace=True)

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for symbol in footprint_df['symbol'].unique():
            out_path = os.path.join(output_dir, f"{symbol}_footprint.csv")
            footprint_df[footprint_df['symbol'] == symbol].to_csv(out_path, index=False)
            print(f"Saved features for {symbol} to {out_path}")

    return footprint_df


# === Model Training ===

def train(df, features):
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    joblib.dump(model, "random_forest_model.pkl")
    print("Model saved as random_forest_model.pkl")


# === Pipeline to Create Labeled Dataset ===

def create_footprint(output_dir="merged_csvs", X=8, symbols=None):
    print("Generating footprint features...")
    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        input_path = os.path.join("extracted_symbols", f"{symbol}.csv")
        if not os.path.exists(input_path):
            print(f"Missing input for symbol: {symbol}")
            continue

        df = pd.read_csv(input_path)
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        df['date'] = df['ts_event'].dt.date

        chunks = []
        for _, day_df in tqdm(df.groupby('date')):
            try:
                chunk = build_footprint_with_engineered_features(day_df, N=50)
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error on {day_df['date'].iloc[0]}: {e}")

        features_df = pd.concat(chunks, ignore_index=True)
        print(f"Collected features: {features_df.shape}")

        # Labeling
        print(f"Getting historic candles for {symbol}...")
        candles = get_historical_data(API_KEY, SECRET, symbol,
                                      startdate=datetime(2025, 5, 12, 7, 55, 0),
                                      endDate=datetime(2025, 7, 9, 23, 55, 0))

        candles['timestamp'] = pd.to_datetime(candles['timestamp'])
        candles['close_X'] = candles['close'].shift(-X)
        candles['pct_change_to_close_X'] = (candles['close_X'] - candles['close']) / candles['close'] * 100
        candles['target'] = (candles['pct_change_to_close_X'] > 0.5).astype(int)

        label_df = candles[['timestamp', 'low', 'high', 'pct_change_to_close_X', 'target']]
        combined = pd.merge(features_df, label_df, left_on='time_bin', right_on='timestamp', how='inner')

        combined.dropna(inplace=True)
        output_file = os.path.join(output_dir, f"{symbol}_footprint_with_targets.csv")
        combined.to_csv(output_file, index=False)
        print(f"Saved labeled dataset: {output_file}")

    print("Finished processing all symbols.")


# === Entry Point ===

if __name__ == "__main__":
    features = [
        'total_ask_volume', 'total_bid_volume', 'volume_imbalance',
        'max_ask_bin', 'max_bid_bin',
    ] + [f'ask_dollar_vol_{i}' for i in range(50)] + [f'bid_dollar_vol_{i}' for i in range(50)]

    # Example usage
    # extract_symbol_from_csvs('raw_csvs', 'extracted_symbols', 'SOXL')
    # create_footprint(output_dir="merged_csvs", X=8, symbols=['SOXL'])

    combined_df = pd.read_csv(os.path.join("processed", "all_footprints.csv"))
    train(combined_df, features=features)
