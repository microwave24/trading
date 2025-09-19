# === Standard Libraries ===
import os
from datetime import datetime, timedelta
import warnings
from datetime import time
from tqdm import tqdm
import pytz
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
import websocket
import threading

import json
# === Data Handling ===
import polars as pl
import pandas as pd
import numpy as np

# === Configuration ===
API_KEY = "PK7LLGFUSPUL7PRPME8M"
SECRET = "yvT87Fi7AbxM1xbF48gCbhxlsNNQPKF3wV33iSLw"
# === Data Preperation and Analysis ===
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SOCKET = "wss://stream.data.alpaca.markets/v2/delayed_sip"
SYMBOL = 'TEM'

# === BOT PARAMETERS ===
CAPITAL = 10000  # starting capital
WINDOW_LENGTH = 20  # number of bars in each window
TP_n = 1
SL_n = 1
ENTRY_n = 2
TRADING_CLUSTER = 0  # which clusters to trade
MIN_DOLLAR_VOL = CAPITAL * 100

# === BOT STATE VARIABLES ===
take_price = 0.0
stop_price = 0.0
in_position = False
technique = "meds"

# === Alpaca & Databento ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit



df = pd.DataFrame(columns=["symbol", "timestamp", "close_price"])

bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "signal"])

if os.path.exists("realtime/{SYMBOL}_realtime_bars.csv"):
    bars = pd.read_csv(f"realtime/{SYMBOL}_realtime_bars.csv")
    bars['timestamp'] = pd.to_datetime(bars['timestamp'], utc=True).dt.tz_convert('America/New_York')


current_bar = {}


# model loading
df_c = pd.read_csv(f"clustered/{SYMBOL}_{technique}_clustered_process.csv")

cluster_centers = df_c.drop(columns=['timestamp']).groupby('cluster').mean()


def get_data(df):
    """
    This function retrieves historical stock data for a given symbol from Alpaca's API. The raw data is saved to a CSV file.
    """
    timeframe = TimeFrame(1, TimeFrameUnit.Hour)

    client = StockHistoricalDataClient(API_KEY, SECRET) # Initialize the Alpaca client

    request = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=timeframe,
        start=datetime.today() - timedelta(days=3),
        end=datetime.today() - timedelta(hours=12)
    )

    bars = client.get_stock_bars(request) # actual data
    df = pd.DataFrame([bar.__dict__ for bar in bars[SYMBOL]])

    # Convert 'timestamp' column to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # Convert to New York time
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

    df.drop(columns=['symbol','vwap', 'trade_count'], inplace=True)  # drop unnecessary columns

    # Ensure the "historic" folder exists
    if not os.path.exists("historic"):
        os.makedirs("historic") 

    return df


def on_open(ws):
    print("opened")
    ws.send(json.dumps({"action": "auth", "key": API_KEY, "secret": SECRET}))
    ws.send(json.dumps({
        "action": "subscribe",
        "trades": [SYMBOL],
        "quotes": [SYMBOL],
        "bars":   [SYMBOL]
    }))

def on_error(ws, error):
    print("ERROR:", error)

def on_close(ws, code, msg):
    print("closed connection", code, msg)

def on_message(ws, message):
    data = json.loads(message)
    # Alpaca sends acks like {"T":"success","msg":"connected"} and {"T":"subscription", ...}
    if isinstance(data, dict):
        data = [data]
    for msg in data:
        t = msg.get("T")
        if t == "success":
            print(">>", msg.get("msg"))
            continue
        if t == "subscription":
            print(">> subscribed:", msg)
            continue
        if t == "error":
            print(">> SERVER ERROR:", msg)
            continue
        if t == "t":  # trade
            ts = pd.to_datetime(msg["t"], utc=True).tz_convert("America/New_York")
            price = float(round(msg["p"], 3))
            size = int(msg["s"])
            update_bar(ts, price, size)

            
def update_bar(ts, price, size):
    global bars, current_bar

    bar_time = ts.floor("h")

    new_bar = current_bar.get("timestamp") != bar_time

    # finalize old bar
    if new_bar and current_bar:
        if bars.empty:
            bars = pd.DataFrame([current_bar])
        else:
            bars.loc[len(bars)] = current_bar  # <- no concat, no warning

    if new_bar:
        window = bars.tail(WINDOW_LENGTH)
        signal = trade_logic(window, price, cluster_centers) if len(window) == WINDOW_LENGTH else 0
        current_bar = {
            "timestamp": bar_time,
            "open": price, "high": price, "low": price, "close": price,
            "volume": size,
            "signal": signal,
        }
    else:
        current_bar["high"] = max(current_bar["high"], price)
        current_bar["low"]  = min(current_bar["low"], price)
        current_bar["close"] = price
        current_bar["volume"] += size

    # --- Write bars + current bar together ---
    combined = bars.copy()

    if current_bar:
        # drop last row if it has the same timestamp as current_bar
        if not combined.empty and combined.iloc[-1]["timestamp"] == current_bar["timestamp"]:
            combined = combined.iloc[:-1]

        if combined.empty:
            combined = pd.DataFrame([current_bar])
        else:
            combined.loc[len(combined)] = current_bar  # no FutureWarning
    if not os.path.exists("realtime"):
        os.makedirs("realtime") 

    combined.to_csv(f"realtime/{SYMBOL}_realtime_bars.csv", index=False)


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
    cols = ["delta","ema_spread","ema_mean","atr_spread","pv_ratio"]
    for i in range(len(cluster_centers)):
        center_row = cluster_centers.iloc[i][cols]
        distance = np.linalg.norm(vector.values.flatten() - center_row.values)
        if distance < min_dist:
            prediction = cluster_centers.index[i]
            min_dist = distance

    return prediction

def generate_rolling(window, include_current=False): #window = df.iloc[i-W+1:i+1]  
    if window.empty or len(window) < 2:
        return float('nan'), float('nan'), float('nan')

    hist = window if include_current else window.iloc[:-1]

    # mean/std of CLOSE only (scalars)
    mean_close = float(hist["close"].mean())
    std_close  = float(hist["close"].std(ddof=1))
    mean_dollar_vol = float((hist["close"] * hist["volume"]).mean())

    # ATR (simple SMA of True Range over the slice)
    prev_close = hist["close"].shift(1)
    tr = np.maximum.reduce([
        (hist["high"] - hist["low"]).to_numpy(),
        (hist["high"] - prev_close).abs().to_numpy(),
        (hist["low"]  - prev_close).abs().to_numpy()
    ])
    atr = float(np.nanmean(tr))  # average TR over the slice

    return mean_close, std_close, atr, mean_dollar_vol

def trade_check(open_price, mean, std, atr, dollar_vol ,tp_atr, sl_atr, stdev_n,
                in_pos, take_price, stop_price, prediction, trade_cluster=0):
    # default: no trade, keep current state
    signal = 0

    if dollar_vol < MIN_DOLLAR_VOL:
        return signal, take_price, stop_price, in_pos

    if not in_pos and prediction == trade_cluster:
        # only enter if numbers are valid
        if np.isfinite([open_price, mean, std, atr]).all() and std > 0 and atr > 0:
            if open_price < mean - stdev_n * std:
                take_price = open_price + tp_atr * atr
                stop_price = open_price - sl_atr * atr
                in_pos = True
                signal = 1  # buy
    elif in_pos:
        if open_price >= take_price or open_price <= stop_price:
            in_pos = False
            signal = -1  # exit

    return signal, take_price, stop_price, in_pos

def trade_logic(window, open_price, cluster_centers):
    global in_position, take_price, stop_price

    prediction = predict(window, cluster_centers)
    mean, std, atr, mean_dollar_vol = generate_rolling(window)

    signal, take_price, stop_price, in_position = trade_check(
            open_price, mean, std, atr, mean_dollar_vol,
            TP_n, SL_n, ENTRY_n,
            in_position, take_price, stop_price, prediction, TRADING_CLUSTER
        )
        
    return signal  # no trade signal




def run_ws():
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    # pings help keep some networks happy
    ws.run_forever()

if __name__ == "__main__":
    # EITHER run on main thread:

    if(len(bars) < WINDOW_LENGTH):
        print("Fetching initial data...")
        bars = get_data(bars)

        if not os.path.exists("realtime"):
            os.makedirs("realtime")

        bars.to_csv(f"realtime/{SYMBOL}_realtime_bars.csv", index=False)
        print("Initial data fetched.")
    print(cluster_centers)
    run_ws()
    print("WS thread terminated")