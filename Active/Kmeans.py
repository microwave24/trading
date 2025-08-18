# === Standard Libraries ===
import os
from datetime import datetime, timedelta
import warnings
from datetime import time
from tqdm import tqdm
import pytz
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.exceptions import InconsistentVersionWarning

# === Warning Suppression ===
warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

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



def plot_candlestick(df):
    # Load CSV and parse dates
    ## df = pd.read_csv(filepath, parse_dates=['timestamp'])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)  # ensures tz-aware UTC
    
    # Set Datetime as index
    df.set_index('timestamp', inplace=True)

    # Rename columns to OHLC format (only rename columns that exist)
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }

    df = df.rename(columns=column_mapping)
    


    # Keep only required OHLC columns
    df = df[['Open', 'High', 'Low', 'Close', 'prediction', 'signals']]
    apds = None

    # Create full-length series with NaN for non-matching predictions

    marker_0 = np.full(len(df), np.nan)
    marker_1 = np.full(len(df), np.nan)
    marker_2 = np.full(len(df), np.nan)

    marker_0 = pd.Series(index=df.index, dtype=float)
    marker_1 = pd.Series(index=df.index, dtype=float)
    marker_2 = pd.Series(index=df.index, dtype=float)

    # Only set values where predictions match
    cluster0 = df["prediction"] == 0
    cluster1 = df["prediction"] == 1
    cluster2 = df["prediction"] == 2

    buy_r = df["signals"] == 1
    sell_r = df["signals"] == -1
        
    marker_0[cluster0] = df['Open'][cluster0]   
    marker_1[cluster1] = df['Open'][cluster1] 
    marker_2[cluster2] = df['Open'][cluster2]  

    buy_marker = pd.Series(np.nan, index=df.index)
    sell_marker = pd.Series(np.nan, index=df.index)

    buy_marker[buy_r] = df['Open'][buy_r]
    sell_marker[sell_r] = df['Open'][sell_r]

    apds = [
        #mpf.make_addplot(marker_0, type='scatter', markersize=20, marker='o', color='g'),  
        #mpf.make_addplot(marker_1, type='scatter', markersize=20, marker='o', color='b'),
        #mpf.make_addplot(marker_2, type='scatter', markersize=20, marker='o', color='r'),
        mpf.make_addplot(buy_marker, type='scatter', markersize=50, marker='^', color='lime', panel=0),
        mpf.make_addplot(sell_marker, type='scatter', markersize=50, marker='v', color='red', panel=0)  
    ]
        
    # Remove prediction column before plotting (mplfinance only needs OHLC)
    plot_df = df[['Open', 'High', 'Low', 'Close']]

    # Plot candlestick chart
    plot_kwargs = {
        'type': 'candle',
        'style': 'yahoo',
        'title': 'TQQQ 5-Minute Candlestick Chart',
        'ylabel': 'Price',
        'volume': False
    }
    
    # Only add addplot if we have markers
    if apds is not None:
        plot_kwargs['addplot'] = apds
    
    mpf.plot(plot_df, **plot_kwargs)

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

    df.dropna(inplace=True)  # Ensure no NaN values before scaling
    # === Scale only the numeric features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_to_scale])

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
    df_clean = df.dropna(subset=features_to_scale).copy()
    # === Scale only the numeric features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_to_scale])

    # === KMeans Clustering ===
    k = 3  # from elbow method, k = 3 is best
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    df.loc[:, 'cluster'] = labels

    df.to_csv(f"output/clustering_output_TQQQ.csv", index=False)
    return df
    

def assign_clusters(historic, clustered):
    """
    This function just copies over the cluster labels from the clustered DataFrame to the original DataFrame.
    This is done to keep the original/unscaled data intact while adding cluster information.
    """
    df_merged = historic.merge(clustered[['cluster']], left_index=True, right_index=True, how='left')
    df_merged.to_csv(f"output/historic_clustering_output_TQQQ.csv", index=False)
    return df_merged


def retrieve_clusters(df, output_path, cluster_count=3):
    """
    Optional function to save each cluster's data to a separate CSV file.
    """
    for i in range(0,cluster_count):
        cluster = df[df['cluster'] == i]
        cluster.to_csv(f"{output_path}/cluster{i}.csv", index=False)





def trade_check(i, current_price, entry_price, in_pos, df, atr_threshold_tp=1.5, atr_threshold_sl=0.5, std_n=2):
    std = df.at[i, "roll_std"]
    mean = df.at[i, "mean"]

    pos_std = mean + std_n * std
    neg_std = mean - std_n * std
    atr = df.at[i, "avg_atr"]

    if pd.isna(pos_std) or pd.isna(atr):
        return 0  # skip until rolling windows are valid

    if in_pos:
        if current_price >= entry_price + atr * atr_threshold_tp:
            return -1
        elif current_price <= entry_price - atr * atr_threshold_sl:
            return -1
    else:
        if current_price <= neg_std:
            return 1
    return 0


def predict(window, cluster_centers, max_distances, sim_threshold):
    prediction = -1

    # === Avg Delta ===
    open_price = window.iloc[0]['open']
    close_price = window.iloc[-1]['close']

    delta = ((close_price - open_price) / open_price) * 100

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

    summerized = pd.DataFrame({
        "delta": [delta],
        "avg_ema10_slope": [avg_slope],
        "atr_spread": [spread],
        "candle_ratio": [candle_ratio],
        "peak_count": [p],
        "trough_count": [t]
    })

    # Find the closest cluster center using Euclidean distance
    
    prediction = -1
    min_dist = np.inf

    for cluster in cluster_centers.index:
        
        center_row = cluster_centers.loc[int(cluster), ["delta", "avg_ema10_slope", "atr_spread", 
                                           "candle_ratio", "peak_count", "trough_count"]]
        
        distance = np.linalg.norm(summerized.values.flatten() - center_row.values)

        if distance < min_dist:
            prediction = int(cluster)
            min_dist = distance

    similarity = 1 - (min_dist / max_distances[prediction])

    if similarity >= sim_threshold:
        return prediction
    return -1

def rolling_averages(df, processed, window_length, std_n=2, atr_period=10, atr_avg=30):
    # Rolling mean & std for close

    df["mean"] = df["close"].rolling(window_length).mean()
    df["roll_std"] = df["close"].rolling(window_length).std(ddof=1)  # sample std
    
    # Precompute pos/neg thresholds
    df["pos_std"] = df["mean"] + std_n * df["roll_std"]
    df["neg_std"] = df["mean"] - std_n * df["roll_std"]

    df["avg_atr"] = np.nan  # Initialize avg_atr column
    for i in tqdm(range(window_length, len(df))):
        window = df.iloc[i-window_length:i]

        atr = avg_atr(window)
        df.at[i, "avg_atr"] = atr

    
    processed["pos_std"] = df["pos_std"]
    processed["neg_std"] = df["neg_std"]
    processed["mean"] = df["mean"]
    processed["avg_atr"] = df["avg_atr"]


    df.to_csv("output/historic_clustered_w_avg.csv", index=False)
    processed.to_csv("output/processed_output_TQQQ_w_avg.csv", index=False)

    return df
    

def backtest(predicted, window_length, sl, tp, std_n=2):

    df = predicted.reset_index(drop=True).copy()

    capital = 10000.0
    quantity = 0
    wins = 0
    total_trades = 0
    
    peak_capital = capital
    max_drawdown = 0.0

    df = predicted.reset_index(drop=True).copy()
    n = len(df)

    # Extract numpy arrays for speed
    opens = df["open"].to_numpy()
    preds = df["prediction"].to_numpy()

    signals = np.zeros(n, dtype=np.int8)  # instead of df.loc per loop
    capital_history = np.empty(n, dtype=np.float64)

    in_pos = False
    entry_price = 0.0

    print(df.columns)

    for i in tqdm(range(window_length, n)):
        current_price = opens[i]
        prediction = preds[i]

        # portfolio value
        if in_pos:
            current_portfolio_value = capital + (quantity * current_price)
        else:
            current_portfolio_value = capital

        # update drawdown
        if current_portfolio_value > peak_capital:
            peak_capital = current_portfolio_value
        dd = ((peak_capital - current_portfolio_value) / peak_capital) * 100
        if dd > max_drawdown:
            max_drawdown = dd

        capital_history[i] = current_portfolio_value

        # trade decision
        trade = trade_check(i, current_price, entry_price, in_pos, df, tp, sl, std_n)
        
        # Entry
        if not in_pos:
            if trade == 1 and (prediction == 0 or prediction == 1):
                in_pos = True
                entry_price = current_price
                quantity = capital // current_price
                capital -= quantity * current_price
                total_trades += 1
                signals[i] = 1
        else:
            if trade == -1 and prediction != 0:
                capital += quantity * current_price
                if current_price > entry_price:
                    wins += 1
                quantity = 0
                in_pos = False
                entry_price = 0
                signals[i] = -1

    # final value
    final_price = opens[-1] if in_pos else 0
    final_portfolio_value = capital + (quantity * final_price if in_pos else 0)

    # metrics
    total_return = ((final_portfolio_value - 10000.0) / 10000.0) * 100
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0

    df["signals"] = signals

    return {
        "df": df,
        "final_capital": final_portfolio_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "winrate": winrate,
        "total_trades": total_trades,
        "capital_history": capital_history
    }

def findLargestDist(df, cluster_centers, k):

    out = np.zeros(k)
    print(cluster_centers.index)
    for c in cluster_centers.index:
        cluster = df[df['cluster'] == int(c)]
        if cluster.empty:
            continue

        cluster.reset_index(drop=True, inplace=True)
        center_row = cluster_centers.loc[int(c), ["delta", "avg_ema10_slope", "atr_spread", 
                                           "candle_ratio", "peak_count", "trough_count"]]
        max_dist = 0
        for i in tqdm(range(len(cluster))):
            data = cluster.iloc[i][["delta", "avg_ema10_slope", "atr_spread", 
                                           "candle_ratio", "peak_count", "trough_count"]]
            
            dist = np.linalg.norm(data.values.flatten() - center_row.values)

            if dist > max_dist:
                max_dist = dist
            # Calculate distance
        out[int(c)] = max_dist
    return out


def optimise(df):
    """
    We are optimising the parameters for the backtest function. Parameters include:
    - tp
    - sl 
    - std_n
    - similarity threshold
    
    """
    # TP & SL values: 0.2 → 5.0, step 0.05
    tp_values = np.arange(0.5, 3.0 + 0.001, 0.5).round(2).tolist()
    sl_values = np.arange(0.5, 3.0 + 0.001, 0.5).round(2).tolist()

    # std_n values: 0.1 → 5.0, step 0.1
    std_n_values = np.arange(1, 3.0 + 0.001, 0.5).round(2).tolist()

    # sim_thresholds: 0.5 → 0.98, step 0.02
    sim_thresholds = np.arange(0.6, 0.9 + 0.001, 0.1).round(2).tolist()

    historic = pd.read_csv("historic/TQQQ_historic_data_1min.csv")

    historic["timestamp"] = pd.to_datetime(historic["timestamp"], utc=True)

    max_dists = np.load("output/max_distances.npy")
    cluster_summary = df.drop(columns=['timestamp']).groupby('cluster').mean()

    startdate = datetime(2025, 6, 15, 13, 30, 0, tzinfo=pytz.UTC)
    enddate = datetime(2025, 6, 30, 20, 30, 0, tzinfo=pytz.UTC)

    historic = historic[
        (historic["timestamp"] >= startdate) &
        (historic["timestamp"] <= enddate)
    ]

    total_combinations = len(tp_values) * len(sl_values) * len(std_n_values)
    print(f"Total combinations to test: {total_combinations} for {len(sim_thresholds)} similarity thresholds")
    
    done = 0

    results = []

    for sim_threshold in sim_thresholds:
        
        historic['prediction'] = -1
        pred_col = historic.columns.get_loc('prediction')

        for i in tqdm(range(30, len(historic))):
            window = historic.iloc[i-30:i]
            prediction =  predict(window, cluster_summary, max_distances=max_dists, sim_threshold=sim_threshold)
            historic.iloc[i, pred_col] = prediction

        for tp in tp_values:
            for sl in sl_values:
                for std_n in std_n_values:
                    bt_result = backtest(historic, window_length=30, sl=sl,
                                         tp=tp, std_n=std_n)
                    results.append({
                        "tp": tp,
                        "sl": sl,
                        "std_n": std_n,
                        "sim_threshold": sim_threshold,
                        "total_return": bt_result["total_return"],
                        "max_drawdown": bt_result["max_drawdown"],
                        "final_capital": bt_result["final_capital"],
                        "winrate": bt_result["winrate"],
                        "total_trades": bt_result["total_trades"]
                    })

                    done += 1
                    print(f"Completed {done}/{total_combinations}")
                     
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/optimisation_results.csv", index=False)
    print("Saved results to output/optimisation_results.csv")
    
def precompute_predictions(historic, cluster_centers, max_distances, startdate, enddate, window_size, sim_threshold=0.8):
    historic["timestamp"] = pd.to_datetime(historic["timestamp"], utc=True)

    historic = historic[
            (historic["timestamp"] >= startdate) &
            (historic["timestamp"] <= enddate)
        ]
    
    historic.set_index("timestamp", inplace=True)

    # == PREDICTION ==
    print("Starting prediction...")
    historic['prediction'] = -1
    pred_col = historic.columns.get_loc('prediction')

    print(len(historic))

    for i in tqdm(range(window_size, len(historic))):
        window = historic.iloc[i-window_size:i]
        prediction = predict(window, cluster_centers, max_distances=max_distances, sim_threshold=sim_threshold)
        historic.iloc[i, pred_col] = prediction
    historic.to_csv("output/historic_clustered_w_avg_predicted_TQQQ.csv", index=True)

    print(f'number of predictions: {len(historic)}')
    print(f'number of predictions with -1: {len(historic[historic["prediction"] == -1])}')
    print(f'number of predictions with 0: {len(historic[historic["prediction"] == 0])}')
    print(f'number of predictions with 1: {len(historic[historic["prediction"] == 1])}')
    print(f'number of predictions with 2: {len(historic[historic["prediction"] == 2])}')


    

    
def compute_cluster_centers_chunked(df, chunk_size=10000):
    """Compute cluster centers with chunked processing for very large datasets"""
    
    df_no_timestamp = df.drop(columns=['timestamp'])
    unique_clusters = df_no_timestamp['cluster'].unique()
    
    cluster_centers = {}

    print("Computing cluster centers...")
    
    for cluster_id in tqdm(unique_clusters, desc="Processing clusters"):
        cluster_data = df_no_timestamp[df_no_timestamp['cluster'] == cluster_id]
        cluster_centers[cluster_id] = cluster_data.drop(columns=['cluster']).mean()
    
    return pd.DataFrame(cluster_centers).T
                    
def delete_garbage_cluster(clustered, original_processed, cluster_id):
    """
    Deletes a cluster from the processed DataFrame.
    This is useful if a cluster is found to be garbage or not useful.
    """
    # Find all timestamps in clustered that belong to that cluster
    timestamps_to_remove = clustered[clustered['cluster'] == cluster_id]['timestamp']

    # Filter original_processed by removing those timestamps
    df_filtered = original_processed[~original_processed['timestamp'].isin(timestamps_to_remove)].reset_index(drop=True)
    df_filtered.to_csv(f"processed/processed_output_TQQQ.csv", index=False)

            
if __name__ == "__main__":
    # == RAW DATA RETRIEVAL ==
    #startdate = datetime(2025, 6, 1, 13, 30, 0, tzinfo=pytz.UTC)
    #enddate = datetime(2025, 6, 30, 20, 30, 0, tzinfo=pytz.UTC)

    #historic = get_historical_data(API_KEY, SECRET, "TQQQ", startDate=startdate,endDate=enddate, daily=False, t=1)
    # == PROCESSING RAW DATA ==
    historic = pd.read_csv("historic/TQQQ_historic_data_1min.csv")
    #processed_historic = process(historic, "TQQQ")
    processed_historic = pd.read_csv("processed/processed_output_TQQQ.csv")
    

    # == CLUSTERING ==
    print("Clustering...")
    
    #clear_bad(processed_historic)

    split_index = int(len(processed_historic) * 0.8)
    train_df = processed_historic[:split_index]
    test_df = processed_historic[split_index:]

    features_to_scale = ["delta", "avg_ema10_slope", "atr_spread", "candle_ratio", "peak_count", "trough_count"]
    features_to_pass = ["timestamp"]

    #elbow_method(train_df, features_to_scale, features_to_pass)

    clustered_df = cluster(train_df, features_to_scale, features_to_pass)
    processed_clustered = pd.read_csv("output/clustering_output_TQQQ.csv")


    #for c in range(3):
        #count = (processed_clustered['cluster'] == c).sum()
        #print(f"Cluster {c} row count: {count}")

    #delete_garbage_cluster(processed_clustered, processed_historic, 2)

    historic_clustered = assign_clusters(historic, processed_clustered)
    historic_clustered = pd.read_csv("output/historic_clustering_output_TQQQ.csv")
    


    # == CLUSTER CENTERS ==
    cluster_centers = processed_clustered.drop(columns=['timestamp']).groupby('cluster').mean()
    #print(cluster_centers)

    # == AVERAGES ==
    window_length = 30
    std_n = 2
    print("Calculating rolling averages...")
    print("last timestamp:", historic_clustered["timestamp"].iloc[-1])
    #rolling_averages(df=historic_clustered, processed=processed_clustered, window_length=window_length, std_n=std_n, 
                     #atr_period=10, atr_avg=30)
    historic_clustered_w_avg = pd.read_csv("output/historic_clustered_w_avg.csv")
    processed_historic_w_avg = pd.read_csv("output/processed_output_TQQQ_w_avg.csv")

    print("Rolling averages calculated and saved to output/historic_clustered_w_avg.csv")

    
    # == Prediction ==

    #max_dists = findLargestDist(processed_historic_w_avg, cluster_centers, k=3)
    max_dists = np.load("output/max_distances.npy")
    print("Max distances for each cluster:", max_dists)

    #precompute_predictions(historic_clustered_w_avg, cluster_centers,
                           #max_distances=max_dists,
                           #startdate=datetime(2025, 6, 1, 13, 30, 0, tzinfo=pytz.UTC),
                           #enddate=datetime(2025, 6, 30, 20, 30, 0, tzinfo=pytz.UTC),
                           #window_size=window_length, sim_threshold=0.8)
    predicted = pd.read_csv("output/historic_clustered_w_avg_predicted_TQQQ.csv")

    # == BACKTESTING ==
    backtest_result = backtest(predicted, window_length=window_length, sl=1, tp=1, std_n=0.1)
    backtest_result["df"].to_csv("output/backtest_result_TQQQ.csv", index=False)

    print("Backtest Results:")
    print(f"Final Capital: {backtest_result['final_capital']:.2f}")
    print(f"Total Return: {backtest_result['total_return']:.2f}%")
    print(f"Max Drawdown: {backtest_result['max_drawdown']:.2f}%")
    print(f"Winrate: {backtest_result['winrate']:.2f}%")
    print(f"Total Trades: {backtest_result['total_trades']}")

    backtest_result = pd.read_csv("output/backtest_result_TQQQ.csv")

    plot_candlestick(backtest_result)
    
    








    




    
