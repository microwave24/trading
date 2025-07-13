# === Standard Libraries ===
import os
import glob
from datetime import datetime

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
    return df

def filter_rows(df):
    """Filter raw data to only have that are actual trades and have a labelled aggressor"""
    df = df.filter((pl.col('side').is_in(['A', 'B'])) & (pl.col('action') == 'T'))
    return df

def aggregate_rows(df, historic_df, bins=50):
    """Aggregate rows into candles of a specified size."""
    df = df.with_columns(pl.col('ts_event').cast(pl.Datetime))  # ensure timestamp is datetime
    df = df.with_columns(pl.col('ts_event').dt.truncate('5m').alias('time_bin'))  # create 5-min bins

    footprints = []

    grouped = df.group_by('time_bin').agg(pl.all()).sort('time_bin')

    # Wrap groupby iterator with tqdm for progress bar
    for row in tqdm(grouped.iter_rows(named=True), total=len(grouped), desc="Aggregating footprints"):
        time_bin = row['time_bin']
        group_df = pl.DataFrame(row).select(pl.exclude('time_bin'))

        if group_df.is_empty():
            continue

        low_price = historic_df.filter(pl.col('timestamp') == time_bin)['low'][0]
        high_price = historic_df.filter(pl.col('timestamp') == time_bin)['high'][0]

        if low_price == high_price:
            continue  # skip flat candles

        import numpy as np
        bin_edges = np.linspace(low_price, high_price, bins + 1)
        buy_volume = np.zeros(bins)
        sell_volume = np.zeros(bins)

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

        delta = buy_volume - sell_volume
        footprint = np.stack((buy_volume, sell_volume, delta), axis=1)
        footprints.append(footprint)

    return np.array(footprints)

if __name__ == "__main__":
    raw_data_path = "extracted_symbols"
    csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))

    print("loading data from csv files...")
    df = pl.read_csv(csv_files[0], n_threads=os.cpu_count() - 2)
    print(f"Loaded data from {csv_files[0]}, shape: {df.shape}")

    historic_df = get_historical_data(API_KEY, SECRET, 'TQQQ', startdate=datetime(2025, 5, 12, 7, 55, 0),
                                      endDate=datetime(2025, 7, 9, 23, 55, 0))
    
    df = filter_rows(df)
    footprints = aggregate_rows(df, historic_df, bins=50)

    np.save("processed/footprints.npy", footprints)
    print("Footprints saved to processed/footprints.npy")