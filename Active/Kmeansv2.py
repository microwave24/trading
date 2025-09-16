
# === Standard Libraries ===
import os
import math
from datetime import datetime
import warnings
from tqdm import tqdm
import pytz
import matplotlib.pyplot as plt
import mplfinance as mpf
# === Utilities ===
from tqdm import tqdm
# === Data Handling ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import product


# === Alpaca & Databento ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# === Variables ===
WINDOW_LENGTH = 20
TIMEFRAME = 'h'
TIMEFRAME_LENGTH = 1
SYMBOL = 'TPET'

API_KEY = "PKKI3LTTYAXQTD08267W"
SECRET = "eO6dkZWeBXetQyU71WPQNh1sAx7pc9IChmt4Fig7"

START_DATE = datetime(2022, 1, 30, 13, 30, 0, tzinfo=pytz.UTC)
END_DATE = datetime(2025, 8, 30, 20, 30, 0, tzinfo=pytz.UTC)


def get_data():
    """
    This function retrieves historical stock data for a given symbol from Alpaca's API. The raw data is saved to a CSV file.
    """
    if TIMEFRAME == "m":
        timeframe = TimeFrame(TIMEFRAME_LENGTH, TimeFrameUnit.Minute)
    elif TIMEFRAME == "h":
        timeframe = TimeFrame(TIMEFRAME_LENGTH, TimeFrameUnit.Hour)
    elif TIMEFRAME == "d":
        timeframe = TimeFrame.Day
    else:
        raise ValueError("Invalid t_type. Use 'm' for minutes, 'h' for hours, or 'd' for days.")

    client = StockHistoricalDataClient(API_KEY, SECRET) # Initialize the Alpaca client

    request = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=timeframe,
        start=START_DATE,
        end=END_DATE
    )

    bars = client.get_stock_bars(request) # actual data
    df = pd.DataFrame([bar.__dict__ for bar in bars[SYMBOL]])

    # Convert 'timestamp' column to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # Convert to New York time
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    # Ensure the "historic" folder exists
    if not os.path.exists("historic"):
        os.makedirs("historic") 

    # we use dollar volume because stock price change the volume value, so we could accidently label periods of high price as "low volume"
    df["dollar_volume"] = df["volume"] * df["close"]

    df.to_csv(f"historic/{SYMBOL}_historic_data.csv", index=False)
    return df

def analyze_volume_data():
    # purpose of this function is to find bad rows: such as low volume

    df = pd.read_csv(f"historic/{SYMBOL}_historic_data.csv")
    
    # Plotting the distribution of volume
    volume = df["dollar_volume"]
    mean = volume.mean()
    std = volume.std()

    # we create custom binning below the mean because occassionally there are high volume counts that skew the mean
    bins = np.concatenate([
        np.linspace(volume.min(), mean, 900),
        np.linspace(mean, volume.max(), 1) 
    ])
    sns.histplot(volume, bins=bins, kde=True, stat="density")
    x = np.linspace(volume.min(), volume.max(), 1000)

    # Normal distribution curve (PDF)
    pdf = norm.pdf(x, mean, std)
    plt.plot(x, pdf, 'r-', linewidth=2, label="Normal distribution")

    # Mean line
    plt.axvline(mean, color="blue", linestyle="--", label=f"Mean = {mean:.2f}")

    plt.legend()
    plt.title("Histogram with Normal Distribution")
    plt.show()

    """
    What I found (on symbol --> PCG):
    - there seems to be a very large spike in frequency of candles that had volumes between 0 to 40000 dollar volume
    - This corresponds to 24/7 market, premarket and post market trading periods and thus should be culled from the data
    """

def mean_reversion_analysis(df):

    # lets check if price likes to revert to mean after dropping more than 2stds 
    # we will use the last "look_size" candle average as the mean to revert, then find candles that deviate by n*stds or more and then check if it

    reverts = 0
    non_reverts = 0
    total = 0
    look_size = 10
    stdev = 0.5

    
    active = False

    past_window_mean_price = 0
    stop_price = 0
    take_price = 0
    price_std = 0

    bars = 0
    times_for_reversion = []



    for i in tqdm(range(look_size, len(df))):
        current_candle = df.iloc[i]
        past_window = df[i-look_size:i]

        past_window_mean_price = past_window['close'].mean()
        price_std = past_window['close'].std()

        # Long setup: price dips below -stdev * sigma
        if not active and current_candle['open'] <= past_window_mean_price - stdev*price_std:
            active = True
            direction = "long"

            take_price = past_window_mean_price  # mean reversion target
            stop_price = current_candle['open'] - (take_price - current_candle['open'])  # symmetric stop

            bars = 0

        # Short setup: price spikes above +stdev * sigma
        elif not active and current_candle['open'] >= past_window_mean_price + stdev*price_std:
            active = True
            direction = "short"

            take_price = past_window_mean_price  # mean reversion target
            stop_price = current_candle['open'] + (current_candle['open'] - take_price)  # symmetric stop

            bars = 0

        if active:
            bars += 1

            if bars > 480:
                non_reverts += 1
                active = False
                bars = 0
                continue

            if direction == "long":
                if current_candle['high'] >= take_price:  # hit mean
                    reverts += 1
                    active = False
                    times_for_reversion.append(bars)
                    bars = 0
                elif current_candle['low'] <= stop_price:  # stop-out
                    non_reverts += 1
                    active = False
                    bars = 0

            elif direction == "short":
                if current_candle['low'] <= take_price:  # reverted down to mean
                    reverts += 1
                    active = False
                    times_for_reversion.append(bars)
                    bars = 0
                elif current_candle['high'] >= stop_price:  # stop-out
                    non_reverts += 1
                    active = False
                    bars = 0
            


    total = reverts + non_reverts

    # Plot bar chart for reverts and non-reverts
    labels = ['Reverts', 'Non-Reverts']
    counts = [reverts, non_reverts]

    plt.bar(labels, counts, color=['green', 'red'])
    plt.title(f"Mean Reversion Outcomes (Total: {total})")
    print(f"Reversion rate: {reverts}/{non_reverts} = {reverts/(non_reverts + reverts):.2%}")
    plt.ylabel("Count")
    plt.show()


    bins = np.concatenate([
        np.linspace(0, 120, 480),
        np.linspace(480, 1000, 1) 
    ])

    plt.figure(figsize=(8, 5))
    sns.histplot(times_for_reversion, bins=bins, color='skyblue')

    mean_time = np.mean(times_for_reversion)
    median_time = np.median(times_for_reversion)

    plt.axvline(median_time, color='green', linestyle='--', label=f"Median = {np.median(times_for_reversion):.2f}")
    plt.axvline(mean_time, color='red', linestyle='--', label=f"Mean = {np.median(mean_time):.2f}") 

    plt.title("Distribution of Times to Reversion")
    plt.xlabel("Bars to Reversion")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    """
    What I found:
        - There does seem to be true that the price does mean revert, 50% to 65% of the time. Obv when we lower the stdev it reverts much
          more often --> about 60% when stdev is about 0.33, but realistically that wouldn't be an amazing return target, so anything less 0.5
          stdev is rejected

        - Thus there is an inverse relation between stdev and reversion rate
          
    """


def vwap_reversion_analysis(
    df,
    look_size=120,
    stdev=0.5,
    max_bars=480
):
    """
    Expects df with columns:
      ['open','high','low','close','vwap'] (vwap should be from prior bar to avoid lookahead).
    Symmetric long/short triggers off VWAP ± stdev * rolling_close_std.
    """
    # Counters
    reverts = 0
    non_reverts = 0
    times_for_reversion = []
    returns = []

    # State
    active = False
    direction = None
    take_price = 0.0
    stop_price = 0.0
    bars = 0

    

    n = len(df)
    if n <= look_size:
        print("Not enough data.")
        return {"reverts": 0, "non_reverts": 0, "reversion_rate": 0.0, "times": []}

    for i in tqdm(range(look_size, n), desc="VWAP reversion"):
        current = df.iloc[i]
        prev    = df.iloc[i-1]
        window  = df.iloc[i-look_size:i]

        # Use previous bar's VWAP to avoid lookahead
        vwap_now = prev["vwap"]
        price_std = window["close"].std(ddof=0)

        # --- Entry logic ---
        if not active:
            # Long: open falls below VWAP - k*sigma
            if current["open"] <= vwap_now - stdev * price_std:
                active = True
                direction = "long"
                take_price = vwap_now
                # symmetric stop: distance from entry to target
                dist = take_price - current["open"]
                stop_price = current["open"] - dist
                bars = 0

            # Short: open rises above VWAP + k*sigma
            elif current["open"] >= vwap_now + stdev * price_std:
                active = True
                direction = "short"
                take_price = vwap_now
                dist = current["open"] - take_price
                stop_price = current["open"] + dist
                bars = 0

        # --- Management / Exit logic ---
        if active: 
            bars += 1
            

            # Timeout
            if bars > max_bars:
                non_reverts += 1
                active = False
                bars = 0
                continue

            if direction == "long":
                entry_price = (take_price + stop_price) / 2
                return_ = (current["open"] - entry_price) / entry_price
                returns.append(return_)
                # target hit if high reaches VWAP
                if current["high"] >= take_price:
                    reverts += 1
                    times_for_reversion.append(bars)
                    active = False
                    bars = 0
                    
                # stop-out if low breaches stop
                elif current["low"] <= stop_price:
                    non_reverts += 1
                    active = False
                    bars = 0

            elif direction == "short":
                entry_price = (take_price + stop_price) / 2

                # target hit if low reaches VWAP
                if current["low"] <= take_price:
                    reverts += 1
                    times_for_reversion.append(bars)
                    active = False
                    bars = 0
                # stop-out if high breaches stop
                elif current["high"] >= stop_price:
                    non_reverts += 1
                    active = False
                    bars = 0

    total = reverts + non_reverts
    rate = (reverts / total) if total else 0.0

    # ---- Plots ----
    # Bar chart of outcomes
    plt.figure()
    plt.bar(["Reverts", "Non-Reverts"], [reverts, non_reverts], color=['green', 'red'])
    plt.title(f"VWAP Reversion Outcomes (Total: {total})")
    plt.ylabel("Count")
    print(f"Reversion rate: {reverts}/{total} = {rate:.2%}")
    print(f"med return: {np.median(returns)*100}")
    plt.show()

    # Histogram of bars-to-reversion
    if times_for_reversion:
        plt.figure()
        bins = np.concatenate([
            np.linspace(0, max_bars, min(max_bars, 400)),
            np.linspace(max_bars, max(times_for_reversion)+1, 1)
        ])
        sns.histplot(times_for_reversion, bins=bins, color='skyblue')
        mean_time = float(np.mean(times_for_reversion))
        median_time = float(np.median(times_for_reversion))
        plt.axvline(median_time, color='green', linestyle='--', label=f"Median = {median_time:.2f}")
        plt.axvline(mean_time, color='red', linestyle='--', label=f"Mean = {mean_time:.2f}")
        plt.title("Distribution of Bars to VWAP Reversion")
        plt.xlabel("Bars to Reversion")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    return {
        "reverts": reverts,
        "non_reverts": non_reverts,
        "reversion_rate": rate,
        "times": times_for_reversion
    }


def avg_atr(df, period=14):
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

    return atr.iloc[-1]

def atr_reversion_analysis(df):
    # purpose of this function is to find bad rows: such as low volume
    window_size = 120
    n_std = 0.5
    n_atr = 0.5

    df = df.copy()


    reversion_price = np.nan
    stop_price = np.nan

    in_pos = False

    reversion = 0
    non_reversion = 0

    time = 0

    for i in tqdm(range(window_size, len(df))):
        window = df[i-window_size:i]

        current_price = df.iloc[i]['open']
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        
        mean = window['close'].mean()
        std = window['close'].std()
        atr = avg_atr(window[-3:], period=3)

        if not in_pos:
            if current_price < mean - n_std * std:
                reversion_price = current_price + atr*n_atr
                stop_price = current_price - atr*n_atr
                in_pos = True
            
        if in_pos:
            time += 1
            if time > 240:
                in_pos = False
                non_reversion += 1
                time = 0
            elif current_high >= reversion_price:
                in_pos = False
                reversion += 1
            elif current_low <= stop_price:
                in_pos = False
                non_reversion += 1
    
    print(f"Reversion rate: {reversion}/{non_reversion} = {reversion/(non_reversion + reversion):.2%}")
        

def cull_data():
    df = pd.read_csv(f"historic/{SYMBOL}_historic_data.csv")
    df = df[df['dollar_volume'] > 40000]
    df.dropna()

    return df

def get_highs(df):
    df = df.copy()
    df['p'] = 0
    rolling_max = df['high'].rolling(window=5, center=True).max()
    df['p'] = np.where(df['high'] == rolling_max, 1, 0)

    return sum(df['p'])

def get_lows(df):
    df = df.copy()
    df['v'] = 0
    rolling_max = df['low'].rolling(window=5, center=True).min()
    df['v'] = np.where(df['low'] == rolling_max, 1, 0)

    return sum(df['v'])



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

def skewness(x, bias=False):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan

    m  = x.mean()
    d  = x - m
    m2 = np.mean(d*d)
    if m2 <= 0:
        return 0.0  # all equal
    m3 = np.mean(d*d*d)

    g1 = m3 / (m2**1.5)  # moment skewness
    if bias:
        return g1
    # Fisher–Pearson (unbiased) sample skewness
    return np.sqrt(n * (n - 1)) / (n - 2) * g1

def process_data(df, save=False):
    df = df.copy()
    df.dropna()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df_processed = pd.DataFrame({
        "timestamp" : df["timestamp"],
        "delta": [np.nan] * len(df),
        "ema_spread": [np.nan] * len(df),
        "ema_mean": [np.nan] * len(df),
        "atr_spread": [np.nan] * len(df),
        "pv_ratio": [np.nan] * len(df)
    })

    

    LARGE_WINDOW = WINDOW_LENGTH * 3
    for i in tqdm(range(LARGE_WINDOW, len(df))):

        window = df[i-WINDOW_LENGTH:i]

        slopes = window["close"].ewm(span=10, adjust=False).mean()
        slopes = slopes * 100

        # delta
        delta = ((window.iloc[-1]['close'] - window.iloc[0]['close'])/window.iloc[0]['close']) * 100

        # ema mean
        ema_mean = slopes.diff().dropna().mean() 
        # ema varience
        ema_spread = slopes.diff().dropna().std()

        # ATR spread
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

        atr_spread = tr.std()

        # peaks and valleys
        p = get_highs(window)
        v = get_lows(window)

        if v == 0:
            v = 1  # to avoid division by zero

        ratio = p / v  # +1 to avoid division by zero

        # adding to the processed df

        ts = df.iloc[i]["timestamp"]  # or df.index[i] if your index is the time

        df_processed.loc[i, [
            "timestamp","delta","ema_spread","ema_mean","atr_spread","pv_ratio"
        ]] = [
            ts,            # timestamp
            delta,         
            ema_spread,
            ema_mean,
            atr_spread,
            ratio
        ]
    if save == True:
        if not os.path.exists("processed"):
            os.makedirs("processed") 

        df_processed = df_processed[LARGE_WINDOW:]
        df_processed = df_processed[LARGE_WINDOW:].reset_index(drop=True)

        df_processed.to_csv(f"processed/{SYMBOL}_processed_data.csv", index=False)
    

def run_kmeans(df, columns, k=3, random_state=42):
    """
    This function performs KMeans clustering on the given DataFrame using specified features.
    """

    # === KMeans Clustering ===
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[columns])

    labels = kmeans.labels_
    df.loc[:, 'cluster'] = labels

    if not os.path.exists("clustered"):
            os.makedirs("clustered") 

    df.to_csv(f"clustered/{SYMBOL}_clustered_process.csv", index=False)
    return df
        

    
    
def elbow_method(df, columns):
    """
    This function implements the Elbow Method to determine the optimal number of clusters (k) for KMeans clustering.
    """
    inertia = []
    K_range = range(1, 12)

    df.dropna(inplace=True)  # Ensure no NaN values

    X = df[columns].copy()

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(K_range)
    plt.show()

    # =================================================

def run_kmedians(df, columns, k=3, max_iter=500, tol=1e-4, random_state=42):
    rng = np.random.default_rng(random_state)
    X = df[columns].to_numpy()
    n_samples, n_features = X.shape

    # initialize medians
    medians = X[rng.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        # assign each row to nearest median (L1 distance)
        dists = np.abs(X[:, None, :] - medians[None, :, :]).sum(axis=2)
        labels = np.argmin(dists, axis=1)

        new_medians = np.copy(medians)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_medians[j] = np.median(cluster_points, axis=0)

        # check convergence
        if np.all(np.abs(new_medians - medians) < tol):
            break
        medians = new_medians

    df["cluster"] = labels
    df.to_csv(f"clustered/{SYMBOL}_clustered_process.csv", index=False)
    return df, medians

def elbow_method_kmedians(df, columns, max_k=11, random_state=42):
    costs = []
    ks = range(1, max_k + 1)

    for k in ks:
        df_tmp = df.copy()
        df_tmp, medians = run_kmedians(df_tmp, columns, k=k, random_state=random_state)

        X = df_tmp[columns].to_numpy()
        labels = df_tmp["cluster"].to_numpy()

        # compute total L1 cost for this k
        cost = 0
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                cost += np.abs(cluster_points - medians[j]).sum()
        costs.append(cost)

    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), costs, 'bo-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Total L1 Cost")
    plt.title("Elbow Method for Optimal k (K-medians)")
    plt.xticks(list(ks))
    plt.show()

    return costs

def assign_clusters(df, df_p):
    df['timestamp']  = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
    df_p['timestamp'] = pd.to_datetime(df_p['timestamp']).dt.tz_convert(None)
    df['cluster'] = df['timestamp'].map(df_p.set_index('timestamp')['cluster'])

    if not os.path.exists("clustered"):
        os.makedirs("clustered") 
    df.to_csv(f"clustered/{SYMBOL}_clustered_historic.csv", index=False)

def predict(window, cluster_centers):
    slopes = window["close"].ewm(span=3, adjust=False).mean()
    slopes = slopes * 100

    # delta
    delta = ((window.iloc[-1]['close'] - window.iloc[0]['close'])/window.iloc[0]['close']) * 100

    # ema_3 mean
    ema_mean = slopes.diff().dropna().mean() 
    # ema_3 varience
    ema_spread = slopes.diff().dropna().std()

    # ATR spread
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

    atr_spread = tr.std()


    # peaks and valleys
    p = get_highs(window)
    v = get_lows(window)


    # adding to the processed df
    vector = pd.DataFrame({
        "delta": [delta],
        "ema_spread": [ema_spread],
        "ema_mean": [ema_mean],
        "atr_spread": [atr_spread],
        "pv_ratio": [p / (v if v != 0 else 1)]
    })

    
    prediction = -1
    min_dist = np.inf
    for cluster in cluster_centers.index:
        
        center_row = cluster_centers.loc[int(cluster), ["delta", "ema_spread", "ema_mean", "atr_spread", "pv_ratio"]]
        
        distance = np.linalg.norm(vector.values.flatten() - center_row.values)

        if distance < min_dist:
            prediction = int(cluster)
            min_dist = distance

    return prediction

def trade_check(open_price, mean, std, atr, tp_atr, sl_atr, stdev_n,
                in_pos, take_price, stop_price):
    if not in_pos:
        # enter position
        if open_price < mean - stdev_n * std:
            take_price = open_price + tp_atr * atr
            stop_price = open_price - sl_atr * atr
            in_pos = True
            return 1, take_price, stop_price, in_pos  # buy signal
    else:
        # exit position
        if open_price >= take_price or open_price <= stop_price:
            in_pos = False
            return -1, take_price, stop_price, in_pos  # sell signal


    return 0, take_price, stop_price, in_pos  # hold signal

def backtest(df, stdev_n=2, tp_atr=1, sl_atr=1, trade_cluster=0,
             initial_capital=10000.0, save=False):
    mean = df['mean_close'].to_numpy()
    std = df['std_close'].to_numpy()
    atr = df['atr'].to_numpy()
    open_price = df['open'].to_numpy()
    prediction = df['prediction'].to_numpy()

    volume = df['dollar_volume'].to_numpy()

    signals = np.zeros(len(df))
    equity = np.full(len(df), np.nan, dtype=float)
    drawdown_pct = np.full(len(df), np.nan, dtype=float)
    trade_pnl = np.full(len(df), np.nan, dtype=float)

    in_pos = False
    take_price = 0.0
    stop_price = 0.0

    capital = float(initial_capital)
    peak_equity = capital
    max_drawdown = 0.0

    entry_price = np.nan
    entry_capital = capital
    qty = 0.0  # full-port position size in units when in a trade

    wins = 0
    total_trades = 0

    for i in range(WINDOW_LENGTH, len(df)):
        # mark-to-market before state change
        equity_i = qty * open_price[i] if in_pos and np.isfinite(entry_price) else capital
        if volume[i] < 40000:
            continue  # skip low volume bars

        # generate signal only when inputs are valid and cluster matches
        if prediction[i] == trade_cluster and not (
            np.isnan(mean[i]) or np.isnan(std[i]) or np.isnan(atr[i])
        ):
            signal, take_price, stop_price, in_pos_next = trade_check(
                open_price[i], mean[i], std[i], atr[i],
                tp_atr, sl_atr, stdev_n,
                in_pos, take_price, stop_price
            )
        else:
            signal = 0
            in_pos_next = in_pos

        # handle transitions
        if signal == 1:
            # enter with full capital
            entry_price = open_price[i]
            entry_capital = capital - 0.99
            qty = 0.0 if entry_price <= 0 or not np.isfinite(entry_price) else entry_capital / entry_price

        elif signal == -1:
            # exit and realize PnL
            if np.isfinite(entry_price) and qty > 0:
                exit_val = qty * open_price[i]
                pnl = exit_val - entry_capital
                capital = exit_val - 0.99
                trade_pnl[i] = pnl
                total_trades += 1
                if pnl > 0:
                    wins += 1
            entry_price = np.nan
            entry_capital = capital
            qty = 0.0

        # after applying signal, recompute equity mark
        equity_i = qty * open_price[i] if in_pos_next and np.isfinite(entry_price) else capital
        equity[i] = equity_i

        # drawdown tracking
        if equity_i > peak_equity:
            peak_equity = equity_i
        dd = 0.0 if peak_equity <= 0 else (peak_equity - equity_i) / peak_equity
        drawdown_pct[i] = dd
        if dd > max_drawdown:
            max_drawdown = dd

        # finalize for next loop
        in_pos = in_pos_next
        signals[i] = signal

    # optional forced close on last bar
    if in_pos and np.isfinite(entry_price) and qty > 0:
        exit_val = qty * open_price[-1]
        pnl = exit_val - entry_capital
        capital = exit_val
        trade_pnl[-1] = pnl
        total_trades += 1
        if pnl > 0:
            wins += 1
        in_pos = False
        qty = 0.0
        equity[-1] = capital
        if capital > peak_equity:
            peak_equity = capital
        dd = 0.0 if peak_equity <= 0 else (peak_equity - capital) / peak_equity
        drawdown_pct[-1] = dd
        if dd > max_drawdown:
            max_drawdown = dd

    winrate = 0.0 if total_trades == 0 else (wins / total_trades) * 100.0

    # write outputs
    df = df.copy()
    df['signals'] = signals
    df['equity'] = equity
    df['drawdown_pct'] = drawdown_pct
    df['trade_pnl'] = trade_pnl
    if save:
        if not os.path.exists("backtest"):
            os.makedirs("backtest")
        df.to_csv(f"backtest/{SYMBOL}_backtest.csv", index=False)

    
    stats = {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "net_return_pct": 0.0 if initial_capital == 0 else (capital / initial_capital - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "wins": wins,
        "total_trades": total_trades,
        "winrate_pct": winrate,
    }
    return stats



def plot(df):
    
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
    marker_3 = pd.Series(np.nan, index=df.index)

    cluster0 = df["prediction"] == 0
    cluster1 = df["prediction"] == 1
    cluster2 = df["prediction"] == 2
    cluster3 = df["prediction"] == 3

    buy_r = df["signals"] == 1
    sell_r = df["signals"] == -1

    marker_0[cluster0] = df['Open'][cluster0]   
    marker_1[cluster1] = df['Open'][cluster1] 
    marker_2[cluster2] = df['Open'][cluster2]
    marker_3[cluster3] = df['Open'][cluster3]  
     

    buy_marker = pd.Series(np.nan, index=df.index)
    sell_marker = pd.Series(np.nan, index=df.index)

    buy_marker[buy_r] = df['Open'][buy_r]
    sell_marker[sell_r] = df['Open'][sell_r]

    # Only include non-empty addplots
    apds = []
    for series, kwargs in [
        #(buy_marker, dict(type='scatter', markersize=100, marker='^', color='lime', panel=0)),
        #(sell_marker, dict(type='scatter', markersize=100, marker='v', color='red', panel=0)),
        (marker_0, dict(type='scatter', markersize=5, marker='o', color="#FF0000")),
        (marker_1, dict(type='scatter', markersize=5, marker='o', color="#0E24EB")),
        (marker_2, dict(type='scatter', markersize=5, marker='o', color="#15FF00")),
        (marker_3, dict(type='scatter', markersize=5, marker='o', color="#A200FF"))

        
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


def pretty_print(stats):
    fnum = lambda v: f"{float(v):,.2f}"
    fpct = lambda v: f"{float(v):,.2f}%"
    print(
        f"{'Initial capital:':<18}{fnum(stats['initial_capital'])}\n"
        f"{'Final capital:':<18}{fnum(stats['final_capital'])}\n"
        f"{'Net return:':<18}{fpct(stats['net_return_pct'])}\n"
        f"{'Max drawdown:':<18}{fpct(stats['max_drawdown_pct'])}\n"
        f"{'Wins:':<18}{stats['wins']:,}\n"
        f"{'Total trades:':<18}{stats['total_trades']:,}\n"
        f"{'Win rate:':<18}{fpct(stats['winrate_pct'])}"
    )


    
def generate_predictions(df, cluster_centers):
    df = df.reset_index(drop=True).copy()
    df["prediction"] = -1
    pred_col = df.columns.get_loc("prediction")

    for i in tqdm(range(WINDOW_LENGTH, len(df))):
        window = df.iloc[i - WINDOW_LENGTH:i]
        pred = predict(window, cluster_centers)
        df.iloc[i, pred_col] = pred  # replaced .at with .iloc

    os.makedirs("historic_predicted", exist_ok=True)
    df.to_csv(f"historic_predicted/{SYMBOL}_historic_predicted.csv", index=False)

def generate_rolling_data(df):
    w = WINDOW_LENGTH
    out = df.copy()

    # rolling mean/std in C
    out["mean_close"] = out["close"].rolling(w).mean().shift(1)
    out["std_close"]  = out["close"].rolling(w).std(ddof=1).shift(1)

    # ATR (vectorized True Range)
    prev_close = out["close"].shift(1)
    tr = np.maximum.reduce([
        (out["high"] - out["low"]).to_numpy(),
        (out["high"] - prev_close).abs().to_numpy(),
        (out["low"]  - prev_close).abs().to_numpy()
    ])
    # simple ATR = rolling mean of TR
    out["atr"] = pd.Series(tr, index=out.index).rolling(w).mean().shift(1)
    # Wilder alternative (uncomment if needed)

    # drop warmup rows without using dropna
    out = out.iloc[w:].copy()

    os.makedirs("ready", exist_ok=True)
    out.to_csv(f"ready/{SYMBOL}_test_data.csv", index=False)
    return out

def optimise(df_test):
    stdevs  = np.arange(0.0, 3.0, 0.2)
    tp_atrs = np.arange(0.5, 10.0, 0.5)
    sl_atrs = np.arange(0.5, 10.0, 0.5)

    dtype = [
        ("return", "f8"), ("drawdown", "f8"), ("winrate", "f8"),
        ("trades", "i4"), ("tp", "f8"), ("sl", "f8"), ("stdev", "f8"),
    ]

    total = stdevs.size * tp_atrs.size * sl_atrs.size
    results = np.empty(total, dtype=dtype)

    i = 0
    for stdev, tp, sl in product(stdevs, tp_atrs, sl_atrs):
        stats = backtest(df_test, stdev_n=stdev, tp_atr=tp, sl_atr=sl, trade_cluster=2, save=False)
        results[i] = (
            float(stats["net_return_pct"]),
            float(stats["max_drawdown_pct"]),
            float(stats["winrate_pct"]),
            int(stats["total_trades"]),
            float(tp), float(sl), float(stdev),
        )
        i += 1
        print(f"Completed {i}/{total}", end="\r")
            

    df = pd.DataFrame.from_records(results)
    os.makedirs("optimisation", exist_ok=True)
    df.to_csv(f"optimisation/{SYMBOL}_optimisation.csv", index=False)
    plot_optimisation()



def plot_optimisation():
    df = pd.read_csv(f"optimisation/{SYMBOL}_optimisation.csv")

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df["drawdown"], df["return"], c=df["winrate"], cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Win Rate (%)')
    plt.xlabel('Max Drawdown (%)')
    plt.ylabel('Net Return (%)')
    plt.title('Optimization Results')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    get_data()
    #analyze_volume_data()
    df = cull_data()
    #atr_reversion_analysis(df[int(len(df)*0.9):])
    #mean_reversion_analysis(df)
    #vwap_reversion_analysis(df)
    process_data(df, save=True)
    df_p = pd.read_csv(f"processed/{SYMBOL}_processed_data.csv")

    col = ["delta","ema_spread","ema_mean", "atr_spread","pv_ratio"]

    #print(len(df_p))
    split = int(len(df_p)*0.75)
    df_train = df_p[:split]
    df_test = df[split:]
    #elbow_method_kmedians(df_train, col, max_k=10)
    #elbow_method(df_train, col)
    run_kmeans(df_train, col, k=3)
    
    #elbow_method_kmedians(df_train, col, max_k=10)
    #run_kmedians(df_train, col, k=5)

    df_c = pd.read_csv(f"clustered/{SYMBOL}_clustered_process.csv")


    cluster_centers = df_c.drop(columns=['timestamp']).groupby('cluster').mean()

    print("Cluster counts:")
    print(df_c["cluster"].value_counts())

    print(cluster_centers)
    d = generate_rolling_data(df_test)
    generate_predictions(d[(int)(len(d)*0):], cluster_centers)
    
    df_test = pd.read_csv(f"historic_predicted/{SYMBOL}_historic_predicted.csv")
    #stats = backtest(df_test, stdev_n=2, tp_atr=4, sl_atr=4, trade_cluster=1)

    optimise(df_test)
    
    


    #df_backtest = pd.read_csv(f"backtest/{SYMBOL}_backtest.csv")
    #plot(df_backtest)


    """
    possible features for each window:
    - z_score of the change in price of the window in reference of a much larger window frame
    - ema_5 slope varience 
    - average_volitility z_score compared to a much larger window
    - dollar_volume_averge z_score compared to a much larger window
    - peak to valley ratio
    """