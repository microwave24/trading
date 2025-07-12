import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import joblib
from databento import DBNStore
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm  # optional progress bar

API_KEY = ""
SECRET = ""


def dbn_to_df(file_path):
    """
    Convert a DBN file to a pandas DataFrame.
    """
    store = DBNStore.from_file(file_path)
    df = store.to_df()
    return df

def get_historical_data(api_key, secret_key, symbol, startdate=None, endDate=None, daily=False):
    # Create the client
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = None
    # Create the request
    if daily: # If daily data is requested, we get daily bars
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,  # Daily bars
            start=startdate,
            end=endDate
        )
    else:
        request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),  # variable minute bars
        start=startdate,
        end= endDate 
    )
    

    # Fetch bars
    bars = client.get_stock_bars(request_params)
    # Convert the list of bars to a DataFrame
    data = pd.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    return data

def build_footprint_with_engineered_features(df, N=50, output_dir=None):
    """
    Build 5-minute order flow footprint histograms with engineered features.
    Returns a DataFrame with:
    - ask/bid volume bins
    - bin center prices
    - dollar volume (volume × price) per bin
    - aggregate features (imbalance, total vol, etc.)
    """
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    import os

    # Drop unused columns
    df = df.drop(columns=[
        'rtype', 'publisher_id', 'instrument_id', 'channel_id', 
        'order_id', 'flags', 'ts_in_delta', 'sequence'
    ], errors='ignore')

    # Filter for trades only and valid sides
    df_filtered = df[(df['action'] == 'T') & (df['side'].isin(['A', 'B']))].copy()

    # Parse timestamps and time bins
    df_filtered['ts_event'] = pd.to_datetime(df_filtered['ts_event'])
    df_filtered['time_bin'] = df_filtered['ts_event'].dt.floor('5T')

    features_list = []

    for symbol, sym_df in df_filtered.groupby('symbol'):
        sym_df['date'] = sym_df['ts_event'].dt.date  # Break by day to reduce memory

        for date, date_df in sym_df.groupby('date'):
            min_price = date_df['price'].min()
            max_price = date_df['price'].max()
            bins = np.linspace(min_price, max_price, N + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            for time_bin, group in date_df.groupby('time_bin'):
                ask_volumes = group.loc[group['side'] == 'A', ['price', 'size']]
                bid_volumes = group.loc[group['side'] == 'B', ['price', 'size']]

                # Skip empty candle
                if ask_volumes.empty and bid_volumes.empty:
                    continue

                ask_hist, _ = np.histogram(ask_volumes['price'], bins=bins, weights=ask_volumes['size'])
                bid_hist, _ = np.histogram(bid_volumes['price'], bins=bins, weights=bid_volumes['size'])

                feature_row = {
                    'symbol': symbol,
                    'time_bin': time_bin,
                    'total_ask_volume': ask_hist.sum(),
                    'total_bid_volume': bid_hist.sum(),
                    'volume_imbalance': ask_hist.sum() - bid_hist.sum(),
                    'max_ask_bin': ask_hist.max(),
                    'max_bid_bin': bid_hist.max(),
                }

                for i in range(N):
                    feature_row[f'ask_bin_{i}'] = ask_hist[i]
                    feature_row[f'bid_bin_{i}'] = bid_hist[i]
                    feature_row[f'ask_price_{i}'] = bin_centers[i]
                    feature_row[f'bid_price_{i}'] = bin_centers[i]
                    feature_row[f'ask_dollar_vol_{i}'] = ask_hist[i] * bin_centers[i]
                    feature_row[f'bid_dollar_vol_{i}'] = bid_hist[i] * bin_centers[i]

                features_list.append(feature_row)

    # Create DataFrame
    footprint_df = pd.DataFrame(features_list)

    if footprint_df.empty:
        return pd.DataFrame()  # Nothing to normalize or save

    # Normalize selected features
    cols_to_normalize = ['total_ask_volume', 'total_bid_volume', 'volume_imbalance', 'max_ask_bin', 'max_bid_bin']
    cols_to_normalize += [col for col in footprint_df.columns if col.startswith('ask_dollar_vol_') or col.startswith('bid_dollar_vol_')]

    scaler = MinMaxScaler()
    footprint_df[cols_to_normalize] = scaler.fit_transform(footprint_df[cols_to_normalize])

    # Save per-symbol CSVs
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for symbol in footprint_df['symbol'].unique():
            symbol_df = footprint_df[footprint_df['symbol'] == symbol]
            filepath = os.path.join(output_dir, f"{symbol}_footprint.csv")
            symbol_df.to_csv(filepath, index=False)
            print(f"Saved {len(symbol_df)} rows for '{symbol}' to {filepath}")

    footprint_df.dropna(inplace=True)
    return footprint_df
    
def train(df, features):
    """
    Train a Random Forest model on the provided training DataFrame.
    """
    # Define features and target
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,  # number of trees
        max_depth=None,    # grow until all leaves are pure
        random_state=42,
        n_jobs=-1, # use all available cores
        class_weight='balanced'           
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # Save the model
    joblib.dump(rf, "random_forest_model.pkl")
    print("Model trained and saved as random_forest_model.pkl")
    

def extract_data(filepath):
    for filename in os.listdir(folder_path):
        if filename.endswith(".dbn"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")

            df = dbn_to_df(file_path)
            df.to_csv(os.path.join("raw_csvs", f"{os.path.splitext(filename)[0]}.csv"), index=False)
            #build_footprint_with_engineered_features(df)

def extract_symbol_from_csvs(input_folder, output_folder, symbol, chunksize=100_000):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{symbol}.csv")

    with open(output_path, 'w', encoding='utf-8') as out_file:
        header_written = False

        for filename in os.listdir(input_folder):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(input_folder, filename)
            print(f"Processing {file_path}...")

            try:
                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    if 'symbol' not in chunk.columns:
                        continue

                    symbol_chunk = chunk[chunk['symbol'] == symbol]

                    if not symbol_chunk.empty:
                        symbol_chunk.to_csv(
                            out_file,
                            header=not header_written,
                            index=False,
                            mode='a'
                        )
                        header_written = True

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Filtered rows saved to: {output_path}")

#extract_data(folder_path)

def create_footprint(output_dir="merged_csvs", X=8, symbols=None):
    print("Creating footprint with engineered features...")
    
    # for every symbol retreive the historical data
    for symbol in symbols:
        symbol_csv_path = os.path.join("extracted_symbols", f"{symbol}.csv")
        if not os.path.exists(symbol_csv_path):
            print(f"CSV for symbol {symbol} not found at {symbol_csv_path}")
            continue

        raw_df = pd.read_csv(symbol_csv_path)

        raw_df['ts_event'] = pd.to_datetime(raw_df['ts_event'])
        raw_df['date'] = raw_df['ts_event'].dt.date  # split by day

        chunks = []
        for day, day_df in tqdm(raw_df.groupby('date')):
            try:
                chunk = build_footprint_with_engineered_features(day_df, N=50, output_dir=None)
                chunks.append(chunk)
            except Exception as e:
                print(f"Skipping {day} due to error: {e}")
                continue

        df = pd.concat(chunks, ignore_index=True)

        print(f"Getting historic data for symbol: {symbol}")
        historic_data = get_historical_data(API_KEY, SECRET, symbol, startdate=datetime(2025, 5, 12, 7, 55, 0), endDate=datetime(2025, 7, 9, 23, 55, 0), daily=False)
        historic_data['timestamp'] = pd.to_datetime(historic_data['timestamp'])
        historic_data = historic_data.sort_values(by='timestamp').reset_index(drop=True)

        # Calculate future % change
        historic_data['close_X'] = historic_data['close'].shift(-X)
        historic_data['pct_change_to_close_X'] = (
            (historic_data['close_X'] - historic_data['close']) / historic_data['close']
        ) * 100

        # create target column based on future % change
        candlestick_df = historic_data[['timestamp', 'low', 'high', 'pct_change_to_close_X']].copy()
        candlestick_df['target'] = (candlestick_df['pct_change_to_close_X'] > 0.5).astype(int)

        # Merge on timestamp: match df_tqqq['time_bin'] with candlestick_df['timestamp']
        combined_df = pd.merge(
            df,
            candlestick_df,
            left_on='time_bin',
            right_on='timestamp',
            how='inner'  # only keep matching timestamps
        )

        # Optional: drop NaNs
        combined_df.dropna(inplace=True)
        # Export processed combined data to CSV
        output_path = os.path.join(output_dir, f"{symbol}_footprint_with_targets.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Exported processed data to {output_path}")
    print("processed!")

    
if __name__ == "__main__":
    folder_path = r"D:\Data\Extracted"

    #extract_symbol_from_csvs('raw_csvs', 'extracted_symbols', 'SOXL')
    symbols = ['SOXL']
    create_footprint(output_dir="merged_csvs", X=8, symbols=symbols)

    
    #print(f"Combined dataset shape: {combined_df.shape}")
    # Export processed combined data to CSV
    #combined_df.to_csv("symbol_csvs/TQQQ_footprint_with_targets.csv", index=False)
    #print("Exported processed data to symbol_csvs/TQQQ_footprint_with_targets.csv")
    #combined_df.drop(columns=["symbol", "time_bin"], inplace=True)

    features = [
        'total_ask_volume',
        'total_bid_volume',
        'volume_imbalance',
        'max_ask_bin',
        'max_bid_bin',
        'ask_dollar_vol_0', 'ask_dollar_vol_1', 'ask_dollar_vol_2', 'ask_dollar_vol_3', 'ask_dollar_vol_4',
        'ask_dollar_vol_5', 'ask_dollar_vol_6', 'ask_dollar_vol_7', 'ask_dollar_vol_8', 'ask_dollar_vol_9',
        'ask_dollar_vol_10', 'ask_dollar_vol_11', 'ask_dollar_vol_12', 'ask_dollar_vol_13', 'ask_dollar_vol_14',
        'ask_dollar_vol_15', 'ask_dollar_vol_16', 'ask_dollar_vol_17', 'ask_dollar_vol_18', 'ask_dollar_vol_19',
        'ask_dollar_vol_20', 'ask_dollar_vol_21', 'ask_dollar_vol_22', 'ask_dollar_vol_23', 'ask_dollar_vol_24',
        'ask_dollar_vol_25', 'ask_dollar_vol_26', 'ask_dollar_vol_27', 'ask_dollar_vol_28', 'ask_dollar_vol_29',
        'ask_dollar_vol_30', 'ask_dollar_vol_31', 'ask_dollar_vol_32', 'ask_dollar_vol_33', 'ask_dollar_vol_34',
        'ask_dollar_vol_35', 'ask_dollar_vol_36', 'ask_dollar_vol_37', 'ask_dollar_vol_38', 'ask_dollar_vol_39',
        'ask_dollar_vol_40', 'ask_dollar_vol_41', 'ask_dollar_vol_42', 'ask_dollar_vol_43', 'ask_dollar_vol_44',
        'ask_dollar_vol_45', 'ask_dollar_vol_46', 'ask_dollar_vol_47', 'ask_dollar_vol_48', 'ask_dollar_vol_49',
        'bid_dollar_vol_0', 'bid_dollar_vol_1', 'bid_dollar_vol_2', 'bid_dollar_vol_3', 'bid_dollar_vol_4',
        'bid_dollar_vol_5', 'bid_dollar_vol_6', 'bid_dollar_vol_7', 'bid_dollar_vol_8', 'bid_dollar_vol_9',
        'bid_dollar_vol_10', 'bid_dollar_vol_11', 'bid_dollar_vol_12', 'bid_dollar_vol_13', 'bid_dollar_vol_14',
        'bid_dollar_vol_15', 'bid_dollar_vol_16', 'bid_dollar_vol_17', 'bid_dollar_vol_18', 'bid_dollar_vol_19',
        'bid_dollar_vol_20', 'bid_dollar_vol_21', 'bid_dollar_vol_22', 'bid_dollar_vol_23', 'bid_dollar_vol_24',
        'bid_dollar_vol_25', 'bid_dollar_vol_26', 'bid_dollar_vol_27', 'bid_dollar_vol_28', 'bid_dollar_vol_29',
        'bid_dollar_vol_30', 'bid_dollar_vol_31', 'bid_dollar_vol_32', 'bid_dollar_vol_33', 'bid_dollar_vol_34',
        'bid_dollar_vol_35', 'bid_dollar_vol_36', 'bid_dollar_vol_37', 'bid_dollar_vol_38', 'bid_dollar_vol_39',
        'bid_dollar_vol_40', 'bid_dollar_vol_41', 'bid_dollar_vol_42', 'bid_dollar_vol_43', 'bid_dollar_vol_44',
        'bid_dollar_vol_45', 'bid_dollar_vol_46', 'bid_dollar_vol_47', 'bid_dollar_vol_48', 'bid_dollar_vol_49'
    ]
    #train(combined_df, features=features)










