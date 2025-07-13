# === Standard Libraries ===
import os
import glob
from datetime import datetime, timedelta
import warnings

# === Warning Suppression ===
warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === Data Handling ===
import polars as pl
import numpy as np

# === ML Tools ===
import tensorflow as tf

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
    df = pl.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    df.write_csv("historic/historic_data.csv")
    return df

def filter_rows(df):
    """Filter raw data to only have that are actual trades and have a labelled aggressor"""
    df = df.filter((pl.col('side').is_in(['A', 'B'])) & (pl.col('action') == 'T'))
    return df

def aggregate_rows(df, historic_df, bins=50, lookahead=5):
    """Aggregate rows into candles of a specified size, returning an array of (timestamp, footprint) tuples."""
    # Ensure timestamp is datetime and create 5-min bins
    df = df.with_columns(
    pl.col('ts_event').str.strptime(pl.Datetime)  # assume input string is UTC
    )
    df = df.with_columns(
        pl.col('ts_event').dt.truncate('5m').alias('time_bin')
    )


    results = []  # To store (time_bin, footprint) tuples

    # Group by time_bin and aggregate all columns
    grouped = df.group_by('time_bin').agg(pl.all()).sort('time_bin')

    # Wrap groupby iterator with tqdm for progress bar
    for row in tqdm(grouped.iter_rows(named=True), total=len(grouped), desc="Aggregating footprints"):
        time_bin = row['time_bin']
        group_df = pl.DataFrame(row).select(pl.exclude('time_bin'))

        if group_df.is_empty():
            continue

        # Get low and high prices from historic_df for the time_bin
        low_price = historic_df.filter(pl.col('timestamp') == time_bin)['low'][0]
        high_price = historic_df.filter(pl.col('timestamp') == time_bin)['high'][0]

        after_trend = historic_df.filter(
            (pl.col('timestamp') > time_bin) &
            (pl.col('timestamp') <= time_bin + timedelta(minutes=5 * lookahead))
        )
        label = 0

        if after_trend.is_empty():
            label = 0  # No data available to judge future trend
        else:
            future_close = after_trend['close'][-1]  # Closing price at end of lookahead
            current_close = historic_df.filter(pl.col('timestamp') == time_bin)['close'][0]

            # Define an uptrend: price increased by a threshold
            threshold = 0.005  # e.g., 0.6% move
            price_change = (future_close - current_close) / current_close

            label = 1 if price_change > threshold else 0

        if low_price == high_price:
            continue  # Skip flat candles

        # Create bin edges for price levels
        bin_edges = np.linspace(low_price, high_price, bins + 1)
        buy_volume = np.zeros(bins)
        sell_volume = np.zeros(bins)

        # Aggregate buy and sell volumes
        for row in group_df.iter_rows(named=True):
            price_level = row['price']
            volume = row['size']
            side = row['side']

            bin_idx = np.searchsorted(bin_edges, price_level, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)

            if side == 'A':
                buy_volume[bin_idx] += volume
            else:
                sell_volume[bin_idx] += volume

        # Calculate delta and create footprint
        delta = buy_volume - sell_volume
        footprint = np.stack((buy_volume, sell_volume, delta), axis=1)
        
        # Append tuple of (time_bin, footprint, label)
        results.append((time_bin, footprint, label))

    # Convert results to a NumPy array with structured dtype
    dtype = [('timestamp', 'datetime64[ms]'), ('footprint', float, (bins, 3)), ('label', 'i4')]
    return np.array(results, dtype=dtype)

if __name__ == "__main__":
    raw_data_path = "input"
    csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    print("Reading CSV files from:", raw_data_path)
    print(f'loading {len(csv_files)} CSV files from {raw_data_path}')
    df = pl.read_csv(csv_files[0], n_threads=os.cpu_count())
    print(f"Loaded data from {csv_files[0]}, shape: {df.shape}")

    historic_df = get_historical_data(API_KEY, SECRET, 'TQQQ', startdate=datetime(2025, 5, 12, 7, 55, 0),
                                      endDate=datetime(2025, 7, 9, 23, 55, 0))
    
    df = filter_rows(df)
    footprints = aggregate_rows(df, historic_df, bins=50)

    os.makedirs("output", exist_ok=True)
    np.save("output/footprints_TQQQ.npy", footprints)

    print("Done! :)")

    """ 'footprints.npy' will contain structured data with:
    - 'timestamp': datetime64[ms] type
    -  'footprint': a 2D array containing ask volume, bid volume, and the deltas at each price level between low and high (split into 50 buckets) 
    """

    
