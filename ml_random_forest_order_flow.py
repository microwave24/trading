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
    startdate = datetime(2025, 5, 12, 8, 0, 0)
    endDate = datetime(2025, 6, 12, 23, 55, 0)
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

def build_footprint_with_engineered_features(df, N=50, output_dir="symbol_csvs"):
    """
    Build 5-minute order flow footprint histograms with engineered features.

    Returns a DataFrame with:
    - ask/bid volume bins
    - bin center prices
    - dollar volume (volume × price) per bin
    - aggregate features (imbalance, total vol, etc.)
    """
    # Drop unused columns
    df = df.drop(columns=[
        'rtype', 'publisher_id', 'instrument_id', 'channel_id', 
        'order_id', 'flags', 'ts_in_delta', 'sequence'
    ], errors='ignore')

    # Filter for trades only and valid sides
    df_filtered = df[(df['action'] == 'T') and (df['side'].isin(['A', 'B']))].copy()

    # Parse timestamps and time bins
    df_filtered['ts_event'] = pd.to_datetime(df_filtered['ts_event'])
    df_filtered['time_bin'] = df_filtered['ts_event'].dt.floor('5T')

    features_list = []

    for symbol, sym_df in df_filtered.groupby('symbol'):
        min_price = sym_df['price'].min()
        max_price = sym_df['price'].max()
        bins = np.linspace(min_price, max_price, N + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for time_bin, group in sym_df.groupby('time_bin'):
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
                # Volume bins
                feature_row[f'ask_bin_{i}'] = ask_hist[i]
                feature_row[f'bid_bin_{i}'] = bid_hist[i]

                # Bin prices
                feature_row[f'ask_price_{i}'] = bin_centers[i]
                feature_row[f'bid_price_{i}'] = bin_centers[i]

                # Engineered dollar volume features
                feature_row[f'ask_dollar_vol_{i}'] = ask_hist[i] * bin_centers[i]
                feature_row[f'bid_dollar_vol_{i}'] = bid_hist[i] * bin_centers[i]

            features_list.append(feature_row)

    # Create DataFrame
    footprint_df = pd.DataFrame(features_list)
    # the model will be trained on the following features: 'total_ask_volume', 'total_bid_volume', 'volume_imbalance', 'max_ask_bin', 'max_bid_bin', 'bid_dollar_vol_' and 'ask_dollar_vol_'
    
    #normalize features
    cols_to_normalize = ['total_ask_volume', 'total_bid_volume', 'volume_imbalance', 'max_ask_bin', 'max_bid_bin']
    cols_to_normalize += [col for col in footprint_df.columns if col.startswith('ask_dollar_vol_') or col.startswith('bid_dollar_vol_')]

    scaler = MinMaxScaler()
    
    footprint_df[cols_to_normalize] = scaler.fit_transform(footprint_df[cols_to_normalize])

    # Save per-symbol CSVs
    os.makedirs(output_dir, exist_ok=True)
    for symbol in footprint_df['symbol'].unique():
        symbol_df = footprint_df[footprint_df['symbol'] == symbol]
        filepath = os.path.join(output_dir, f"{symbol}_footprint.csv")
        symbol_df.to_csv(filepath, index=False)
        print(f"Saved {len(symbol_df)} rows for '{symbol}' to {filepath}")

    footprint_df.dropna(inplace=True)
    return footprint_df
    
    
    

if __name__ == "__main__":
    #file_path = r"D:\Data\Extracted\xnas-itch-20250512.mbo.dbn"
    #df = dbn_to_df(file_path)
    #build_footprint_with_engineered_features(df)
    #print(f"csv files created")

    # Load footprint features
    df_tqqq = pd.read_csv("symbol_csvs/TQQQ_footprint.csv")
    df_tqqq['time_bin'] = pd.to_datetime(df_tqqq['time_bin'])

    # Load historical data
    historic_candle_data = get_historical_data(API_KEY, SECRET, "TQQQ", daily=False)
    historic_candle_data['timestamp'] = pd.to_datetime(historic_candle_data['timestamp'])
    historic_candle_data = historic_candle_data.sort_values('timestamp').reset_index(drop=True)

    X = 8  # lookahead

    # Calculate future % change
    historic_candle_data['close_X'] = historic_candle_data['close'].shift(-X)
    historic_candle_data['pct_change_to_close_X'] = (
        (historic_candle_data['close_X'] - historic_candle_data['close']) / historic_candle_data['close']
    ) * 100

    # Keep target columns only
    candlestick_df = historic_candle_data[['timestamp', 'low', 'high', 'pct_change_to_close_X']].copy()
    candlestick_df['target'] = (candlestick_df['pct_change_to_close_X'] > 0.5).astype(int)

    # Merge on timestamp: match df_tqqq['time_bin'] with candlestick_df['timestamp']
    combined_df = pd.merge(
        df_tqqq,
        candlestick_df,
        left_on='time_bin',
        right_on='timestamp',
        how='inner'  # only keep matching timestamps
    )

    # Optional: drop duplicate timestamp columns and NaNs
    combined_df = combined_df.drop(columns=['timestamp'])
    combined_df.dropna(inplace=True)

    print(f"Combined dataset shape: {combined_df.shape}")
    # Export processed combined data to CSV
    combined_df.to_csv("symbol_csvs/TQQQ_footprint_with_targets.csv", index=False)
    print("Exported processed data to symbol_csvs/TQQQ_footprint_with_targets.csv")







