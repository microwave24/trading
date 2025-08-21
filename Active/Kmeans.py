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
API_KEY = "PKHLDVUZSBAVJ0GWCR14"
SECRET = "bK4CterukecekQpcg2ElddIW90MrRvggmYU36Ehg"
# === Data Preperation and Analysis ===



def plot_candlestick(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index('timestamp', inplace=True)

    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }
    df = df.rename(columns=column_mapping)
    df = df[['Open', 'High', 'Low', 'Close', 'prediction', 'signals']]

    # Prepare markers
    marker_0 = pd.Series(np.nan, index=df.index)
    marker_1 = pd.Series(np.nan, index=df.index)
    marker_2 = pd.Series(np.nan, index=df.index)

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

    # Only include non-empty addplots
    apds = []
    for series, kwargs in [
        (buy_marker, dict(type='scatter', markersize=100, marker='^', color='lime', panel=0)),
        (sell_marker, dict(type='scatter', markersize=100, marker='v', color='red', panel=0)),
        (marker_0, dict(type='scatter', markersize=5, marker='o', color="#15FF00")),
        (marker_1, dict(type='scatter', markersize=5, marker='o', color="#0E12EB")),
        (marker_2, dict(type='scatter', markersize=5, marker='o', color="#FF0000"))
        
    ]:
        if series.notna().any():  # only add if there's at least one valid point
            apds.append(mpf.make_addplot(series, **kwargs))

    plot_df = df[['Open', 'High', 'Low', 'Close']]

    mpf.plot(
        plot_df,
        type='candle',
        style='yahoo',
        title='Candlestick Chart',
        ylabel='Price',
        volume=False,
        addplot=apds if apds else None
    )


def get_historical_data(api_key, secret_key, symbol, startDate, endDate, t_type="m", t=1):
    """
    This function retrieves historical stock data for a given symbol from Alpaca's API. The raw data is saved to a CSV file.
    """

    if t_type == "m":
        timeframe = TimeFrame(t, TimeFrameUnit.Minute)
    elif t_type == "h":
        timeframe = TimeFrame(t, TimeFrameUnit.Hour)
    elif t_type == "d":
        timeframe = TimeFrame.Day
    else:
        raise ValueError("Invalid t_type. Use 'm' for minutes, 'h' for hours, or 'd' for days.")

    client = StockHistoricalDataClient(api_key, secret_key) # Initialize the Alpaca client

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

def process(df, symbol, window_length):
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

    for i in tqdm(range(window_length, len(df))): 
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
    df_processed = df_processed.iloc[window_length:].reset_index(drop=True)

    df_processed.to_csv(f"processed/processed_output_{symbol}.csv", index=False)
    return df_processed

def clear_bad(df):
    """
    This function removes rows with NaN values and rows where all specified features are zero.
    """
    df = df.dropna()
    features = ["delta", "avg_ema10_slope", "atr_spread", "candle_ratio", "peak_count", "trough_count"]
    df = df.loc[~(df[features] == 0).all(axis=1)]


def avg_atr(df, period=10, avg_window=15):
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


def cluster(df, features_to_scale, features_to_pass, k, symbol):
    """
    This function performs KMeans clustering on the given DataFrame using specified features.
    """
    # === Scale only the numeric features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_to_scale])

    # === KMeans Clustering ===
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    df.loc[:, 'cluster'] = labels
    print(k)

    df.to_csv(f"output/clustering_output_{symbol}.csv", index=False)
    return df
    

def assign_clusters(historic, clustered, symbol):
    """
    This function just copies over the cluster labels from the clustered DataFrame to the original DataFrame.
    This is done to keep the original/unscaled data intact while adding cluster information.
    """
    df_merged = historic.merge(clustered[['cluster']], left_index=True, right_index=True, how='left')
    df_merged.to_csv(f"output/historic_clustering_output_{symbol}.csv", index=False)
    return df_merged


def retrieve_clusters(df, output_path, cluster_count=3):
    """
    Optional function to save each cluster's data to a separate CSV file.
    """
    for i in range(0,cluster_count):
        cluster = df[df['cluster'] == i]
        cluster.to_csv(f"{output_path}/cluster{i}.csv", index=False)





def trade_check(i, current_price, entry_price, tp_price, sl_price, in_pos, df, prediction,
                atr_threshold_tp=1, atr_threshold_sl=1, std_n=2):

    bullish_c = 0
    kangaroo_c = 1
    bear_c = 2


    none = -1

    hyper_bear = -99
    hyper_bull = -99

    std = df.at[i, "roll_std"]
    mean = df.at[i, "mean"]
    diff = df.at[i, "ema_diff"]

    pos_std = mean + std_n * std
    neg_std = mean - std_n * std
    atr = df.at[i, "avg_atr"]

    neg_std_dynamic = neg_std / (1 + diff)

    if pd.isna(pos_std) or pd.isna(atr):
        return 0, tp_price, sl_price
    
    if in_pos:
        if current_price >= tp_price:
            return -1, tp_price, sl_price
        elif current_price <= sl_price:
            return -1, tp_price, sl_price
    else:
        if (prediction == kangaroo_c ) and current_price <= neg_std:
            tp_price = mean + atr * atr_threshold_tp
            sl_price = mean - atr * atr_threshold_sl
            return 1, tp_price, sl_price
        if prediction == bullish_c and diff > np.inf and current_price <= neg_std + std:
            tp_price = current_price + 2*atr * atr_threshold_tp
            sl_price = current_price - atr * atr_threshold_sl
            return 1, tp_price, sl_price

    return 0, tp_price, sl_price


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

def rolling_averages(df, processed, window_length, symbol):
    # Rolling mean & std for close
    # Shift rolling calculations to use only historical data
    df["mean"] = df["close"].rolling(window_length).mean().shift(1)
    df["roll_std"] = df["close"].rolling(window_length).std(ddof=1).shift(1)
    


    # Calculate 12-period and 5-period EMA
    df["ema_l"] = df["close"].ewm(span=120, adjust=False).mean().shift(1)
    df["ema_s"] = df["close"].ewm(span=50, adjust=False).mean().shift(1)
    # Difference between 12 EMA and 5 EMA
    df["ema_diff"] =  df["ema_s"] - df["ema_l"]

    df["avg_atr"] = np.nan  # Initialize avg_atr column
    for i in tqdm(range(window_length, len(df))):
        window = df.iloc[i-window_length:i]

        atr = avg_atr(window)
        df.at[i, "avg_atr"] = atr

    processed["mean"] = df["mean"]
    processed["avg_atr"] = df["avg_atr"]


    df.to_csv("output/historic_clustered_w_avg.csv", index=False)
    processed.to_csv(f"output/processed_output_{symbol}_w_avg.csv", index=False)

    return df
    

def backtest(predicted, window_length, sl, tp, std_n=2):

    df = predicted.reset_index(drop=True).copy()
    df.dropna()

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
    tp_price = np.inf
    sl_price = -np.inf

    for i in range(window_length, n):
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
        trade, tp_price, sl_price = trade_check(i=i, current_price=current_price,entry_price=entry_price, tp_price=tp_price, sl_price=sl_price,in_pos=in_pos,df=df, prediction=prediction, std_n=std_n, atr_threshold_tp=tp, atr_threshold_sl=sl )
        
        # Entry
        if not in_pos and capital > current_price:
            if trade == 1:
                capital -= 0.99

                in_pos = True
                entry_price = current_price
                quantity = capital // current_price
                capital -= quantity * current_price
                total_trades += 1
                signals[i] = 1
                
        else:
            if trade == -1:
                capital += quantity * current_price
                capital -= 0.99
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


def optimise(symbol, window_length):
    """
    We are optimising the parameters for the backtest function. Parameters include:
    - tp
    - sl 
    - std_n
    - similarity threshold
    
    """
    # TP & SL values: 0.2 → 5.0, step 0.05
    tp_values = np.arange(0.5, 6 + 0.001, 0.5).round(2).tolist()
    sl_values = np.arange(0.5, 6 + 0.001, 0.5).round(2).tolist()

    # std_n values: 0.1 → 5.0, step 0.1
    std_n_values = np.arange(1, 5 + 0.001, 1).round(2).tolist()

    # sim_thresholds: 0.5 → 0.98, step 0.02
    sim_thresholds = np.arange(0.1, 0.8 + 0.001, 0.1).round(2).tolist()
    processed_clustered = pd.read_csv(f"output/clustering_output_{symbol}.csv")


    cluster_centers = processed_clustered.drop(columns=['timestamp']).groupby('cluster').mean()

    #rolling_averages(df=historic_clustered, processed=processed_clustered, window_length=window_length)
    historic_clustered_w_avg = pd.read_csv("output/historic_clustered_w_avg.csv")

    max_dists = np.load("output/max_distances.npy")


    startdate = datetime(2024, 7, 1, 13, 30, 0, tzinfo=pytz.UTC)
    enddate = datetime(2025, 7, 30, 20, 30, 0, tzinfo=pytz.UTC)

    historic_clustered_w_avg["timestamp"] = pd.to_datetime(historic_clustered_w_avg["timestamp"], utc=True)
    historic_clustered_w_avg = historic_clustered_w_avg[
        (historic_clustered_w_avg["timestamp"] >= startdate) &
        (historic_clustered_w_avg["timestamp"] <= enddate)
    ]

    results = []
    total = len(sl_values) * len(tp_values) * len(std_n_values) * len(sim_thresholds)
    done = 0
    print(f'Starting optimisation for {total} combinations of parameters!')
    for sim_threshold in sim_thresholds:
        precompute_predictions(historic_clustered_w_avg, cluster_centers,
                           max_distances=max_dists,
                           startdate=startdate,
                           enddate=enddate,
                           window_size=window_length, sim_threshold=sim_threshold, symbol=symbol)
        predicted = pd.read_csv(f"output/historic_clustered_w_avg_predicted_{symbol}.csv")

        predicted["timestamp"] = pd.to_datetime(predicted["timestamp"], utc=True)
        predicted = predicted[
            (predicted["timestamp"] >= startdate) &
            (predicted["timestamp"] <= enddate)
        ]

        for sl in sl_values:
            for tp in tp_values:
                for stds in std_n_values:
                    backtest_result = backtest(predicted, window_length=window_length, sl=sl, tp=tp, std_n=stds)
                    if backtest_result == -1:
                        continue
                    results.append({
                        "return": backtest_result['total_return'],
                        "drawdown": backtest_result['max_drawdown'],
                        "sl": sl,
                        "tp": tp,
                        "std_n": stds,
                        "sim_threshold": sim_threshold
                    })
                    done += 1
                    print(f"Completed {done} out of {total} combinations: return {backtest_result['total_return']}, drawdown: {backtest_result['max_drawdown']}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/optimisation_results.csv", index=False)
    print("Saved results to output/optimisation_results.csv")

    
def precompute_predictions(historic, cluster_centers, max_distances, startdate, enddate, window_size, symbol, sim_threshold=0.8):
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
    historic.to_csv(f"output/historic_clustered_w_avg_predicted_{symbol}.csv", index=True)

    

    print(f'number of predictions: {len(historic)}')
    print(f'number of predictions with -1: {len(historic[historic["prediction"] == -1])}')
    print(f'number of predictions with 0: {len(historic[historic["prediction"] == 0])}')
    print(f'number of predictions with 1: {len(historic[historic["prediction"] == 1])}')
    print(f'number of predictions with 2: {len(historic[historic["prediction"] == 2])}')


                    
def delete_garbage_cluster(clustered, original_processed, cluster_id, symbol):
    """
    Deletes a cluster from the processed DataFrame.
    This is useful if a cluster is found to be garbage or not useful.
    """
    # Find all timestamps in clustered that belong to that cluster
    timestamps_to_remove = clustered[clustered['cluster'] == cluster_id]['timestamp']

    # Filter original_processed by removing those timestamps
    df_filtered = original_processed[~original_processed['timestamp'].isin(timestamps_to_remove)].reset_index(drop=True)
    df_filtered.to_csv(f"processed/processed_output_{symbol}.csv", index=False)

def optimal_front_plot(optimised_values, baseline_return):
    # Filter values above baseline return
    optimised_values = optimised_values[optimised_values["return"] >= baseline_return]

    x = optimised_values["drawdown"]
    y = optimised_values["return"]

    # Simple scatter plot
    plt.scatter(x, y, s=20, c='blue', marker='o', alpha=1.0)
    plt.xlabel('Drawdown')
    plt.ylabel('Return')
    plt.title('Optimised Values Scatter Plot')
    plt.grid(True)
    plt.show()
  
if __name__ == "__main__":
    optimising = 2
    window_length = 20
    symbol ='PCG'
    if optimising == 1:
        optimise(symbol=symbol,window_length=window_length)
        optimal_vals = pd.read_csv("output/optimisation_results.csv")
        optimal_front_plot(optimal_vals, -99)
        print("done")
    elif optimising == 0:
        # == RAW DATA RETRIEVAL ==
    
        startdate = datetime(2021, 1, 1, 13, 30, 0, tzinfo=pytz.UTC)
        enddate = datetime(2025, 7, 30, 20, 30, 0, tzinfo=pytz.UTC)

        historic = get_historical_data(API_KEY, SECRET, f"{symbol}", startDate=startdate,endDate=enddate, t_type="h", t=2)
        # == PROCESSING RAW DATA ==
        
        historic = pd.read_csv(f"historic/{symbol}_historic_data_1min.csv")
        processed_historic = process(historic, f"{symbol}", window_length)
        

        processed_historic = pd.read_csv(f"processed/processed_output_{symbol}.csv")
        clear_bad(processed_historic)
        

        # == CLUSTERING == #optimise
        print("Clustering...")
        
        split_index = int(len(processed_historic) * 0.8)
        train_df = processed_historic[:split_index]
        test_df = processed_historic[split_index:]

        features_to_scale = ["delta", "avg_ema10_slope", "atr_spread", "candle_ratio", "peak_count", "trough_count"]
        features_to_pass = ["timestamp"]

        
        elbow_method(train_df, features_to_scale, features_to_pass)
        
        clustered_df = cluster(train_df, features_to_scale, features_to_pass, k=3, symbol=symbol)
        processed_clustered = pd.read_csv(f"output/clustering_output_{symbol}.csv")

        ##delete_garbage_cluster(processed_clustered, processed_historic, 2, symbol=symbol)

        historic_clustered = assign_clusters(historic, processed_clustered, symbol=symbol)
        historic_clustered = pd.read_csv(f"output/historic_clustering_output_{symbol}.csv")
        


        # == CLUSTER CENTERS ==
        cluster_centers = processed_clustered.drop(columns=['timestamp']).groupby('cluster').mean()
        print(cluster_centers)


        ## === 4 clusters
        # Cluster 0 = Consolidation
        # CLuster 1 = Bullish
        # Cluster 2 = Heavy Bearish
        # Cluster 3 = Bearish
        ## === 3 clusters
        # 0 = bull
        # 1 = consol
        # 2 = bear

        # == AVERAGES ==
        std_n = 2
        print("Calculating rolling averages...")
        rolling_averages(df=historic_clustered, processed=processed_clustered, window_length=window_length, symbol=symbol)
        historic_clustered_w_avg = pd.read_csv("output/historic_clustered_w_avg.csv")
        processed_historic_w_avg = pd.read_csv(f"output/processed_output_{symbol}_w_avg.csv")

        print("Rolling averages calculated and saved to output/historic_clustered_w_avg.csv")

        # == Prediction ==

        max_dists = findLargestDist(processed_historic_w_avg, cluster_centers, k=4)
        np.save("output/max_distances.npy", max_dists)
        max_dists = np.load("output/max_distances.npy")
        print("Max distances for each cluster:", max_dists)

        precompute_predictions(historic_clustered_w_avg, cluster_centers,
                            max_distances=max_dists,
                            startdate=datetime(2024, 7, 1, 13, 30, 0, tzinfo=pytz.UTC),
                            enddate=datetime(2025, 7, 30, 20, 30, 0, tzinfo=pytz.UTC),
                            window_size=window_length, sim_threshold=0.0, symbol=symbol)
        predicted = pd.read_csv(f"output/historic_clustered_w_avg_predicted_{symbol}.csv")\

    historic_clustered = pd.read_csv(f"output/historic_clustering_output_{symbol}.csv")
    processed_clustered = pd.read_csv(f"output/clustering_output_{symbol}.csv")
    cluster_centers = processed_clustered.drop(columns=['timestamp']).groupby('cluster').mean()    
    historic_clustered_w_avg = pd.read_csv("output/historic_clustered_w_avg.csv")
    processed_historic_w_avg = pd.read_csv(f"output/processed_output_{symbol}_w_avg.csv")
    max_dists = np.load("output/max_distances.npy")
    
    precompute_predictions(historic_clustered_w_avg, cluster_centers,
                            max_distances=max_dists,
                            startdate=datetime(2024, 7, 1, 13, 30, 0, tzinfo=pytz.UTC),
                            enddate=datetime(2025, 7, 30, 20, 30, 0, tzinfo=pytz.UTC),
                            window_size=window_length, sim_threshold=0.8, symbol=symbol)
    
    
    predicted = pd.read_csv(f"output/historic_clustered_w_avg_predicted_{symbol}.csv")
        #== BACKTESTING == 4.0,1.0,1.0,0.8
    backtest_result = backtest(predicted, window_length=window_length, sl=4, tp=1, std_n=1)
    backtest_result["df"].to_csv(f"output/backtest_result_{symbol}.csv", index=False)
        
    print("Backtest Results:")
    print(f"Final Capital: {backtest_result['final_capital']:.2f}")
    print(f"Total Return: {backtest_result['total_return']:.2f}%")
    print(f"Max Drawdown: {backtest_result['max_drawdown']:.2f}%")
    print(f"Winrate: {backtest_result['winrate']:.2f}%")
    print(f"Total Trades: {backtest_result['total_trades']}")

    backtest_result = pd.read_csv(f"output/backtest_result_{symbol}.csv")

    plot_candlestick(backtest_result)

    

    


    
    
    








    




    