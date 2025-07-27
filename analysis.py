# === Standard Libraries ===
import os
import glob
from datetime import datetime, timedelta
import warnings
import pytz
from datetime import time
import ast

# === Warning Suppression ===
warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === Data Handling ===
import polars as pl
import pandas as pd
import numpy as np

# === ML Tools ===
from sklearn.preprocessing import StandardScaler
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
# === Data Preperation and Analysis ===

def get_historical_data(api_key, secret_key, symbol, startDate, endDate, daily=False, t=5):
    client = StockHistoricalDataClient(api_key, secret_key)
    timeframe = TimeFrame.Day if daily else TimeFrame(t, TimeFrameUnit.Minute)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=startDate,
        end=endDate
    )
    bars = client.get_stock_bars(request)
    df = pl.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    df.write_csv(f"historic/{symbol}_historic_data_1min.csv")
    print("Done!")
    return df

def process(df, symbol):
    
    df_processed = pd.DataFrame({
        "delta": [0.0] * len(df),
        "avg_ema10_slope": [0.0] * len(df),
        "atr_spread": [0.0] * len(df),
    })

    window_length = 60

    for i in range(window_length, len(df)): 
        row = df.iloc[i]
        window = df.iloc[i-window_length:i]

        # === Slope ===
        ema = window["close"].ewm(span=10, adjust=False).mean()
        avg_slope = ema.diff().dropna().mean()

        high = window['high']
        low = window['low']
        close = window['close']

        # === ATR Spread ===

        # Previous close (shifted)
        prev_close = close.shift(1)

        # True Range (TR)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        spread = tr.std()

        # === Avg Delta ===
        deltas = ((window['close'] - window['open']) / window['open'].abs()) * 100
        delta_std = deltas.std()

        # adding to the processed df
        df_processed.at[i, "delta"] = delta_std
        df_processed.at[i, "avg_ema10_slope"] = avg_slope
        df_processed.at[i, "atr_spread"] = spread

    df_processed.dropna()
    df_processed = df_processed[(df_processed != 0).any(axis=1)]
    df_processed.to_csv(f"processed/processed_output_{symbol}.csv", index=False)
    return df_processed



def cluster(df):
    from sklearn.cluster import KMeans

    # === elbow method to find best num of clusters ===

    #inertia = []
    #K_range = range(1, 11)

    #for k in K_range:
        #kmeans = KMeans(n_clusters=k, random_state=42)
        #kmeans.fit(X_scaled)
        #inertia.append(kmeans.inertia_)

    #plt.figure(figsize=(8, 5))
    #plt.plot(K_range, inertia, 'bo-')
    #plt.xlabel('Number of clusters (k)')
    #plt.ylabel('Inertia')
    #plt.title('Elbow Method for Optimal k')
    #plt.xticks(K_range)
    #plt.show()

    # =================================================

    
    k = 3  # from elbow method, k = 3 is best
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)

    labels = kmeans.labels_
    df['cluster'] = labels

    for cluster_num in range(k):
        print(f"\nCluster {cluster_num} head:")
        print(df[df['cluster'] == cluster_num].head())
    df.to_csv(f"output/clustering_output_TQQQ.csv", index=False)
    

if __name__ == "__main__":
    # === RAW DATA RETRIEVAL ===
    
    #startdate=datetime(2020, 1, 8, 7, 55, 0)
    #endDate=datetime(2025, 7, 10, 23, 55, 0)
    #get_historical_data(API_KEY, SECRET, "TQQQ", startDate=startdate, endDate=endDate, t=1)

    # === DATA PROCESSING ===
    #df = pd.read_csv("historic/TQQQ_historic_data_1min.csv")
    #print("Starting data processing...")
    #process(df, 'TQQQ')
    #print("Data processed!")

    # === CLUSTERING ===
    #print("Clustering...")
    #df = pd.read_csv("processed/processed_output_TQQQ.csv")
    #cluster(df)
    #print("Clustering Successful!")

    """ From analysis:

    Three clusters:
    - 0: Mild uptrend, small deltas, with a gentle uptrend and low volatility
    - 1: Strong selloffs, small deltas, strong negative momentum and very high volatility
    - 2: Consolidation, moderate deltas, mild downtrend or almost stable, consistent volatility
    
    """

    
