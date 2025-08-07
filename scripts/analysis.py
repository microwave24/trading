# === Standard Libraries ===
import os
from datetime import datetime, timedelta
import warnings
from datetime import time
from tqdm import tqdm

# === Warning Suppression ===
warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === Data Handling ===
import polars as pl
import pandas as pd
import numpy as np

# === ML Tools ===
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
# === Utilities ===
from tqdm import tqdm
import matplotlib.pyplot as plt
import mplfinance as mpf

# === Alpaca & Databento ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from databento import DBNStore

# === Configuration ===
API_KEY = ""
SECRET = ""
# === Data Preperation and Analysis ===


def plot_candlestick(filepath):
    
    # Load CSV and parse dates
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    
    # Set Datetime as index
    df.set_index('timestamp', inplace=True)

    # Localize to UTC and convert to New York time
    df.index = df.index.tz_convert('America/New_York')

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']] # we keep only the necessary columns

    # Plot candlestick chart
    mpf.plot(df,
             type='candle',
             style='yahoo',
             title='TQQQ 5-Minute Candlestick Chart', # change title as needed
             ylabel='Price',
             ylabel_lower='Volume',
             volume=True)

def get_historical_data(api_key, secret_key, symbol, startDate, endDate, daily=False, t=1):
    """
    This function retrieves historical stock data for a given symbol from Alpaca's API. The raw data is saved to a CSV file.
    """
    client = StockHistoricalDataClient(api_key, secret_key) # Initialize the Alpaca client
    timeframe = TimeFrame.Day if daily else TimeFrame(t, TimeFrameUnit.Minute) # either multi-minute or daily data

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=startDate,
        end=endDate
    )

    bars = client.get_stock_bars(request) # actual data
    df = pl.DataFrame([bar.__dict__ for bar in bars[symbol]])

    print("Historical data retrieved")
    df.write_csv(f"historic/{symbol}_historic_data_1min.csv")
    print("Done!")
    return df

def process(df, symbol):
    """
    This function processes the raw stock data to extract features for analysis and clustering.:
    - Calculates the average delta (percentage change from open to close)
    - Computes the slope of the 10-period EMA (Exponential Moving Average)
    - Calculates the ATR (Average True Range) spread
    - Computes the ratio of bullish candles to total candles
    - Counts the number of peaks and troughs in the price data using a zigzag algorithm
    The processed data is saved to a CSV file.
    """
    df_processed = pd.DataFrame({
        "timestamp" : df["timestamp"],
        "delta": [0.0] * len(df),
        "avg_ema10_slope": [0.0] * len(df),
        "atr_spread": [0.0] * len(df),
        "candle_ratio": [0.0] * len(df),
        "peak_count": [0] * len(df),
        "trough_count": [0] * len(df)
    })

    window_length = 30 # 30 is arbitrary, can be changed --> will do hyperparameter tuning later

    for i in range(window_length, len(df)): 
        # Extract the window of data
        window = df.iloc[i-window_length:i].copy()

        # === Avg Delta ===
        open_price = window.iloc[0]['open']
        close_price = window.iloc[-1]['close']

        delta =((close_price - open_price) / open_price) * 100
        
        # === Slope ===
        ema = window["close"].ewm(span=10, adjust=False).mean()
        avg_slope = ema.diff().dropna().mean()

        # === ATR Spread ===
        high = window['high']
        low = window['low']
        close = window['close']

        # Previous close
        prev_close = close.shift(1)

        # True Range (TR)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        spread = tr.std()

        # === Candle Ratio
        total = len(window)
        positive = (window['close'] > window['open']).sum()
        candle_ratio = positive / total if total > 0 else 0

        # === number of peaks
        p, t = zigzag(window)

        # adding to the processed df

        df_processed.at[i, "delta"] = delta
        df_processed.at[i, "avg_ema10_slope"] = avg_slope
        df_processed.at[i, "atr_spread"] = spread
        df_processed.at[i, "candle_ratio"] = candle_ratio
        df_processed.at[i, "peak_count"] = p
        df_processed.at[i, "trough_count"] = t

    df_processed.dropna()
    df_processed = df_processed[(df_processed != 0).any(axis=1)]

    df_processed.to_csv(f"processed/processed_output_{symbol}.csv", index=False)
    return df_processed

def clear_bad(df):
    """
    This function removes rows with NaN values and rows where all specified features are zero.
    """
    df = df.dropna()
    features = ["delta", "avg_ema10_slope", "atr_spread", "candle_ratio", "peak_count", "trough_count"]
    df = df.loc[~(df[features] == 0).all(axis=1)]


def avg_atr(df, period=10, avg_window=30):
    """ 
    This function calculates the Average True Range (ATR) for a given DataFrame.
    The ATR is a measure of volatility, and is calculated as the average of the True Range over a specified period.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    atr = atr.dropna()

    return atr[-avg_window:].mean()

def zigzag(window):
    """ 
    This function implements a simple zigzag algorithm to count peaks and troughs in a given window.
    """
    trend = None # -1 down 1 up, None is intial state
    last_trend = None
    pivot_index = 0

    threshold = avg_atr(window) * 1.5 # arbitrary threshold, will do hyperparameter tuning later
    peaks = 0
    valleys = 0

    for i in range(1, len(window)):
        pivot = window.iloc[pivot_index]
        current = window.iloc[i]

        price_change = current["close"] - pivot["close"]

        # Determine trend direction
        if price_change > 0 and abs(price_change) > threshold:
            trend = 1
        elif price_change < 0 and abs(price_change) > threshold:
            trend = -1
        else:
            continue  # Not a significant move yet

        # Count peak or valley if trend changes, if not we continue without counting as we are still in the same trend
        if trend != last_trend:
            if trend == 1:
                peaks += 1
            elif trend == -1:
                valleys += 1
        pivot_index = i

        # Update last trend to current trend
        last_trend = trend

    return peaks, valleys

def elbow_method(df, features_to_scale, features_to_pass):
    """
    This function implements the Elbow Method to determine the optimal number of clusters (k) for KMeans clustering.
    """
    inertia = []
    K_range = range(1, 11)

    ct = ColumnTransformer([
        ('scale', StandardScaler(), features_to_scale),
        ('pass', 'passthrough', features_to_pass)
    ])


    X_scaled = ct.fit_transform(df)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(K_range)
    plt.show()

    # =================================================


def cluster(df, features_to_scale, features_to_pass):
    """
    This function performs KMeans clustering on the given DataFrame using specified features.
    """

    # === Scale and Transform Data ===
    ct = ColumnTransformer([
        ('scale', StandardScaler(), features_to_scale),
        ('pass', 'passthrough', features_to_pass)
    ])

    X_scaled = ct.fit_transform(df) 

    # === KMeans Clustering ===
    k = 3  # from elbow method, k = 3 is best
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    df['cluster'] = labels

    # Output the start of each cluster
    for cluster_num in range(k):
        print(f"\nCluster {cluster_num} head:")
        print(df[df['cluster'] == cluster_num].head())

    df.to_csv(f"output/clustering_output_TQQQ.csv", index=False)
    return df
    

def assign_clusters(original, clustered):
    """
    This function just copies over the cluster labels from the clustered DataFrame to the original DataFrame.
    This is done to keep the original/unscaled data intact while adding cluster information.
    """
    df_merged = original.merge(clustered[['cluster']], left_index=True, right_index=True, how='left')
    df_merged.to_csv(f"output/clustering_output_TQQQ.csv", index=False)

def retrieve_clusters(df, output_path, cluster_count=3):
    """
    Optional function to save each cluster's data to a separate CSV file.
    """
    for i in range(0,cluster_count):
        cluster = df[df['cluster'] == i]
        cluster.to_csv(f"{output_path}/cluster{i}.csv", index=False)

def get_targets(historic, clustered, window):
    """
    
    This function adds the information whether the future close price is higher than the current close price.
    As of current implementation, target is not used in any machine learning model, but added for future possible use.
    """
    # Map timestamps to their index positions in `historic`
    historic = historic.reset_index(drop=True)
    clustered = clustered.copy()
    clustered['target'] = 0

    # Create a mapping from timestamp to index
    timestamp_to_index = {ts: idx for idx, ts in enumerate(historic['timestamp'])}

    for i in tqdm(range(len(clustered))): # tqdm for progress bar
        time_stamp = clustered.at[i, 'timestamp']

        current_index = timestamp_to_index.get(time_stamp, None)
        if current_index is None:
            continue

        future_index = current_index + window
        if future_index >= len(historic):
            continue

        current_close = historic.at[current_index, 'close']
        future_close = historic.at[future_index, 'close']

        if future_close > current_close:
            clustered.at[i, 'target'] = 1

    clustered.to_csv("output/clustering_output_TQQQ.csv", index=False)

def get_averages(historic, clustered, window, std_n):
    """
    This function calculates the average close price, ATR, and standard deviations for each cluster in the clustered DataFrame.
    """

    # Map timestamps to their index positions in `historic`
    historic = historic.reset_index(drop=True)
    clustered = clustered.copy()

    clustered['avg_close'] = 0.0
    clustered['atr'] = 0.0
    clustered['pos_std'] = 0.0
    clustered['neg_std'] = 0.0

    # Create a mapping from timestamp to index
    timestamp_to_index = {ts: idx for idx, ts in enumerate(historic['timestamp'])} 

    for i in tqdm(range(len(clustered))): # tqdm for progress bar
        time_stamp = clustered.at[i, 'timestamp']

        current_index = timestamp_to_index.get(time_stamp, None)
        if current_index is None:
            continue

        future_index = current_index + window
        if future_index >= len(historic):
            continue

        avg_close = historic['close'][current_index:future_index].mean()
        clustered.at[i, 'avg_close'] = avg_close

        squared_diffs = (historic['close'][current_index:future_index] - avg_close) ** 2
        std = np.sqrt(1/ (window - 1) * squared_diffs.sum())
        pos_std = avg_close + std * std_n
        neg_std = avg_close - std * std_n

        clustered.at[i, 'pos_std'] = pos_std
        clustered.at[i, 'neg_std'] = neg_std

        clustered.at[i, 'atr'] = avg_atr(historic.iloc[current_index:future_index])

    clustered.to_csv("output/clustering_output_TQQQ.csv", index=False)




if __name__ == "__main__":
    # === RAW DATA RETRIEVAL ===
    
    #startdate=datetime(2018, 1, 1, 13, 30, 0)
    #endDate=datetime(2025, 7, 30, 20, 30, 0)
    #get_historical_data(API_KEY, SECRET, "TQQQ", startDate=startdate, endDate=endDate, t=1)

    # === DATA PROCESSING ===
    df = pd.read_csv("historic/TQQQ_historic_data_1min.csv")
    
    #print("Starting data processing...")
    #process(df, 'TQQQ')
    #print("Data processed!")

    # === CLUSTERING ===
    #print("Clustering...")
    #df = pd.read_csv("processed/processed_output_TQQQ.csv")
    #clear_bad(df)

    #features_to_scale = ['delta', 'avg_ema10_slope', 'candle_ratio', 'peak_count', 'trough_count']
    #features_to_pass = ['atr_spread']

    #elbow_method(df, features_to_scale, features_to_pass)
    #assign_clusters(pd.read_csv("processed/processed_output_TQQQ.csv"), cluster(df, features_to_scale, features_to_pass))
    #print("Clustering Successful!")

    # === CLUSTERING ===

    #progress = tqdm(total=len(steps))

    #df = pd.read_csv("output/clustering_output_TQQQ.csv")
    #progress.update(1)
    #retrieve_clusters(df, 'output', 3)

    # === Cluster centers ===
    #print("Retrieving cluster centers")
    #cluster_summary = df.drop(columns=['timestamp']).groupby('cluster').mean()
    #print(cluster_summary)
    #progress.update(1)

    # === PCA analysis ===
    #print("Starting PCA analysis")
    #X = df[['delta','avg_ema10_slope','atr_spread','candle_ratio','peak_count','trough_count']]
    #pca = PCA(n_components=2)
    #components = pca.fit_transform(X)
    #df['pca1'], df['pca2'] = components[:, 0], components[:, 1]
    #progress.update(1)

    #sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set1')
    #plt.title('PCA of Clustered Data')
    #plt.savefig("output/pca_clusters.png", dpi=300, bbox_inches='tight')  # Save the figure
    #progress.update(1)

    # === targets
    historic = pd.read_csv("historic/TQQQ_historic_data_1min.csv")
    clustered = pd.read_csv("output/clustering_output_TQQQ.csv")
    #get_targets(historic, clustered, 30)
    get_averages(historic, clustered, 30, 2)
    print("Done!")



    
