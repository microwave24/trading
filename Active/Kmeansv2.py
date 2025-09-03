
# === Standard Libraries ===
import os
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
# === Alpaca & Databento ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# === Variables ===
WINDOW_LENGTH = 30
TIMEFRAME = 'm'
TIMEFRAME_LENGTH = 1
SYMBOL = 'PCG'

API_KEY = ""
SECRET = ""

START_DATE = datetime(2025, 7, 1, 13, 30, 0, tzinfo=pytz.UTC)
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
    # we will use the last 40 candle average as the mean to revert, then find candles that deviate by 2stds or more and then check if it
    # reverts to it within the next 40 candles

    reverts = 0
    non_reverts = 0
    total = 0
    look_size = 60
    stdev = 0.68

    
    active = False

    past_window_mean_price = 0
    stop_price = 0
    take_price = 0
    price_std = 0

    bars = 0


    # s
    for i in tqdm(range(look_size, len(df))):
        current_candle = df.iloc[i]
        past_window = df[i-look_size:i]

        past_window_mean_price = past_window['close'].mean()
        price_std = past_window['close'].std()
    
        if current_candle['open'] <= past_window_mean_price - stdev*price_std and active == False:
            active = True

            take_price = past_window_mean_price
            stop_price = current_candle['open'] - (take_price - current_candle['open'])

        if active == True:
            bars += 1
            if current_candle['high'] >= take_price:
                reverts += 1
                active = False
                bars = 0
            elif current_candle['low'] <= stop_price:
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

    """
    What I found:
        - There does seem to be true that the price does mean revert, 50% to 65% of the time. Obv when we lower the stdev it reverts much
          more often --> about 60% when stdev is about 0.33, but realistically that wouldn't be an amazing return target, so anything less 0.5
          stdev is rejected

        - Thus there is an inverse relation between stdev and reversion rate
          
    """


def vwap_reversion_analysis(df):

    # lets check if price likes to revert to mean after dropping more than 2stds 
    # we will use the last 40 candle average as the mean to revert, then find candles that deviate by 2stds or more and then check if it
    # reverts to it within the next 40 candles

    reverts = 0
    non_reverts = 0
    total = 0
    look_size = 60
    stdev = 4

    
    active = False

    past_window_vwap_price = 0
    stop_price = 0
    take_price = 0
    price_std = 0

    bars = 0


    # s
    for i in tqdm(range(look_size, len(df))):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        past_window = df[i-look_size:i]

        vwap_now = prev_candle['vwap']
        price_std = past_window['close'].std()
    
        if current_candle['open'] <= vwap_now - stdev*price_std and active == False:
            active = True

            take_price = vwap_now
            stop_price = current_candle['open'] - (take_price - current_candle['open'])

        if active == True:
            bars += 1
            if current_candle['high'] >= take_price:
                reverts += 1
                active = False
                bars = 0
            elif current_candle['low'] <= stop_price:
                non_reverts += 1
                active = False
                bars = 0


    total = reverts + non_reverts
    # Plot bar chart for reverts and non-reverts
    labels = ['Reverts', 'Non-Reverts']
    counts = [reverts, non_reverts]

    plt.bar(labels, counts, color=['green', 'red'])
    plt.title(f"VWAP Reversion Outcomes (Total: {total})")
    print(f"Reversion rate: {reverts}/{non_reverts} = {reverts/(non_reverts + reverts):.2%}")
    plt.ylabel("Count")
    plt.show()

    """
    What I found:
        - Price does seem to also revert to vwap
        - however as stdev rises, the reversion rate increases aswell!
        - although at the cost of lower frequency of trade possibilities
    """
            

def cull_data():
    df = pd.read_csv(f"historic/{SYMBOL}_historic_data.csv")
    df = df[df['dollar_volume'] > 40000]
    df.dropna()

    return df

def avg_atr(df, period=1, avg_window=1):
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

def process_data(df, save=False):
    df = df.copy()
    df.dropna()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df_processed = pd.DataFrame({
        "timestamp" : df["timestamp"],
        "delta": [np.nan] * len(df),
        "ema3_var": [np.nan] * len(df),
        "ema3_mean": [np.nan] * len(df),
        "atr_z": [np.nan] * len(df),
        "vol_skew": [np.nan] * len(df),
        "extrema_delta": [np.nan] * len(df),
        "pv_ratio": [np.nan] * len(df)
    })

    

    LARGE_WINDOW = WINDOW_LENGTH * 3
    for i in tqdm(range(LARGE_WINDOW, len(df))):

        window = df[i-WINDOW_LENGTH:i]
        large_window =df[i-LARGE_WINDOW:i-WINDOW_LENGTH]

        slopes = window["close"].ewm(span=3, adjust=False).mean()
        dollar_volumes = window["dollar_volume"]


        # delta
        delta = (window.iloc[-1]['close'] - window.iloc[0]['close'])/window.iloc[0]['close'] * 100

        # ema_3 mean
        ema_mean = slopes.diff().mean()
        # ema_3 varience
        ema_var = slopes.diff().std()

        # ATR z_score
        mL, mW = avg_atr(large_window), avg_atr(window)
        atr_z_score = (mW - mL)/1.68 if np.isfinite(mW) and np.isfinite(mL) else 0.0

        # Dollar Volume Skewness
        vol = dollar_volumes.to_numpy()
        m = vol.mean()
        var = np.mean((vol - m)**2)
        den = var**1.5
        skewness = np.mean((vol - m)**3) / den if den > 0 else 0.0

        # extrema_delta
        extrema_delta = (window["high"].max() - window["low"].min())/window["low"].min() * 100


        # peaks and valleys
        p,v = zigzag(window)
        pv_ratio = p/(v+1)

        # adding to the processed df

        ts = df.iloc[i]["timestamp"]  # or df.index[i] if your index is the time

        df_processed.loc[i, [
            "timestamp","delta","ema3_var","ema3_mean","atr_z","vol_skew","extrema_delta","pv_ratio"
        ]] = [
            ts,            # timestamp
            delta,         
            ema_var,
            ema_mean,
            atr_z_score,
            skewness,
            extrema_delta,
            pv_ratio
        ]
    if save == True:
        if not os.path.exists("processed"):
            os.makedirs("processed") 

        df_processed = df_processed[LARGE_WINDOW:]
        df_processed = df_processed[LARGE_WINDOW:].reset_index(drop=True)

        df_processed.to_csv(f"processed/{SYMBOL}_processed_data.csv", index=False)
    

    
    
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

def run_kmeans(df, columns, k=3):
    X = df[columns].copy()

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    df['cluster'] = kmeans.labels_

    print("Cluster Centers:")
    for idx, center in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {idx} center:")
        for col_name, value in zip(columns, center):
            print(f"  {col_name}: {value}")

    if not os.path.exists("clustered"):
        os.makedirs("clustered") 
    df.to_csv(f"clustered/{SYMBOL}_clustered_process.csv", index=False)



def assign_clusters(df, df_p):
    df['timestamp']  = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
    df_p['timestamp'] = pd.to_datetime(df_p['timestamp']).dt.tz_convert(None)
    df['cluster'] = df['timestamp'].map(df_p.set_index('timestamp')['cluster'])

    if not os.path.exists("clustered"):
        os.makedirs("clustered") 
    df.to_csv(f"clustered/{SYMBOL}_clustered_historic.csv", index=False)

def predict(large_window, cluster_centers):
    LARGE_WINDOW = WINDOW_LENGTH * 3
    window = large_window[LARGE_WINDOW-WINDOW_LENGTH:-1]
    large_window =df[0:LARGE_WINDOW-WINDOW_LENGTH]

    slopes = window["close"].ewm(span=3, adjust=False).mean()
    dollar_volumes = window["dollar_volume"]

    # delta
    delta = (window.iloc[-1]['close'] - window.iloc[0]['close'])/window.iloc[0]['close'] * 100

    # ema_3 mean
    ema_mean = slopes.diff().mean()
    # ema_3 varience
    ema_var = slopes.diff().std()
    # ATR z_score
    mL, mW = avg_atr(large_window), avg_atr(window)
    atr_z_score = (mW - mL)/1.68 if np.isfinite(mW) and np.isfinite(mL) else 0.0

    # Dollar Volume Skewness
    vol = dollar_volumes.to_numpy()
    m = vol.mean()
    var = np.mean((vol - m)**2)
    den = var**1.5
    skewness = np.mean((vol - m)**3) / den if den > 0 else 0.0

    # extrema_delta
    extrema_delta = (window["high"].max() - window["low"].min())/window["low"].min() * 100

    # peaks and valleys
    p,v = zigzag(window)
    pv_ratio = p/(v+1)# ATR z_score
    mL, mW = avg_atr(large_window), avg_atr(window)
    atr_z_score = (mW - mL)/1.68 if np.isfinite(mW) and np.isfinite(mL) else 0.0

    # Dollar Volume Skewness
    vol = dollar_volumes.to_numpy()
    m = vol.mean()
    var = np.mean((vol - m)**2)
    den = var**1.5
    skewness = np.mean((vol - m)**3) / den if den > 0 else 0.0

    # extrema_delta
    extrema_delta = (window["high"].max() - window["low"].min())/window["low"].min() * 100


    # peaks and valleys
    p,v = zigzag(window)
    pv_ratio = p/(v+1)

    vector = pd.DataFrame({
        "delta": [delta],
        "ema3_var": [ema_var],
        "ema3_mean": [ema_mean],
        "atr_z": [atr_z_score],
        "vol_skew": [skewness],
        "extrema_delta": [extrema_delta],
        "pv_ratio": [pv_ratio]
    })
    
    prediction = -1
    min_dist = np.inf
    for cluster in cluster_centers.index:
        
        center_row = cluster_centers.loc[int(cluster), ["delta", "ema3_var", "ema3_mean", "atr_z", 
                                           "vol_skew", "extrema_delta", "pv_ratio"]]
        
        distance = np.linalg.norm(vector.values.flatten() - center_row.values)

        if distance < min_dist:
            prediction = int(cluster)
            min_dist = distance
    return prediction


def backtest(df_test, df_c):
    LARGE_WINDOW = WINDOW_LENGTH * 3
    df_test = df_test.reset_index(drop=True)
    df_test["prediction"] = [-1] * len(df_test)

    cluster_centers = df_c.drop(columns=['timestamp']).groupby('cluster').mean()

    pred_col = df_test.columns.get_loc("prediction")

    for i in tqdm(range(LARGE_WINDOW ,len(df_test))):
        total_window = df_test[i-LARGE_WINDOW:i+1]

        prediction = predict(total_window, cluster_centers)
        df_test.iloc[i, pred_col] = prediction
    
    if not os.path.exists("backtest"):
        os.makedirs("backtest")
    df_test.to_csv(f"backtest/{SYMBOL}_backtest_results.csv", index=False)

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
    df = df[['Open', 'High', 'Low', 'Close', 'prediction']]

    # Prepare markers
    marker_0 = pd.Series(np.nan, index=df.index)
    marker_1 = pd.Series(np.nan, index=df.index)
    marker_2 = pd.Series(np.nan, index=df.index)
    marker_3 = pd.Series(np.nan, index=df.index)

    cluster0 = df["prediction"] == 0
    cluster1 = df["prediction"] == 1
    cluster2 = df["prediction"] == 2
    cluster3 = df["prediction"] == 3

    #buy_r = df["signals"] == 1
    #sell_r = df["signals"] == -1

    marker_0[cluster0] = df['Open'][cluster0]   
    marker_1[cluster1] = df['Open'][cluster1] 
    marker_2[cluster2] = df['Open'][cluster2]
    marker_3[cluster3] = df['Open'][cluster3]  
     

    #buy_marker = pd.Series(np.nan, index=df.index)
    #sell_marker = pd.Series(np.nan, index=df.index)

    #buy_marker[buy_r] = df['Open'][buy_r]
    #sell_marker[sell_r] = df['Open'][sell_r]

    # Only include non-empty addplots
    apds = []
    for series, kwargs in [
        #(buy_marker, dict(type='scatter', markersize=100, marker='^', color='lime', panel=0)),
        #(sell_marker, dict(type='scatter', markersize=100, marker='v', color='red', panel=0)),
        (marker_0, dict(type='scatter', markersize=5, marker='o', color="#4400FF")),
        (marker_1, dict(type='scatter', markersize=5, marker='o', color="#54EB0E")),
        (marker_2, dict(type='scatter', markersize=5, marker='o', color="#FBFF00")),
        (marker_3, dict(type='scatter', markersize=5, marker='o', color="#00751D"))

        
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





if __name__ == "__main__":
    get_data()
    #analyze_volume_data()
    df = cull_data()
    #mean_reversion_analysis(df)
    #vwap_reversion_analysis(df)
    #process_data(df, save=True)
    df_p = pd.read_csv(f"processed/{SYMBOL}_processed_data.csv")

    col = ["delta","ema3_var","ema3_mean", "atr_z","vol_skew","extrema_delta","pv_ratio"]

    split = int(len(df_p) * 0.80) 
    df_train = df_p[:split]
    df_test = df[split:]

    #elbow_method(df_train, col)
    run_kmeans(df_train, col, 4)

    df_c = pd.read_csv(f"clustered/{SYMBOL}_clustered_process.csv")
    
    backtest(df_test, df_c)
    cluster_centers = df_c.drop(columns=['timestamp']).groupby('cluster').mean()
    print(cluster_centers)

    df_backtest = pd.read_csv(f"backtest/{SYMBOL}_backtest_results.csv")

    plot(df_backtest)


    """
    possible features for each window:
    - z_score of the change in price of the window in reference of a much larger window frame
    - ema_5 slope varience 
    - average_volitility z_score compared to a much larger window
    - dollar_volume_averge z_score compared to a much larger window
    - peak to valley ratio
    """