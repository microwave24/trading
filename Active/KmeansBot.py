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
API_KEY = "PKKI3LTTYAXQTD08267W"
SECRET = "eO6dkZWeBXetQyU71WPQNh1sAx7pc9IChmt4Fig7"
# === Data Preperation and Analysis ===
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SOCKET = "wss://stream.data.alpaca.markets/v2/delayed_sip"
SYMBOL = 'PCG'

df = pd.DataFrame(columns=["symbol", "timestamp", "close_price"])
bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
current_bar = {}

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

    bar_time = ts.floor("H")

    # new hour → finalize old bar
    if current_bar.get("timestamp") != bar_time:
        if current_bar:  # append finalized old bar
            bars = pd.concat([bars, pd.DataFrame([current_bar])], ignore_index=True)
        # start a new bar
        current_bar = {
            "timestamp": bar_time,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": size
        }
    else:
        # update existing bar
        current_bar["high"] = max(current_bar["high"], price)
        current_bar["low"]  = min(current_bar["low"], price)
        current_bar["close"] = price
        current_bar["volume"] += size

    # --- Write bars + current bar together ---
    combined = bars.copy()
    if current_bar:  # ensure it's included
        # if last row is same timestamp, drop it first
        if not combined.empty and combined.iloc[-1]["timestamp"] == current_bar["timestamp"]:
            combined = combined.iloc[:-1]
        combined = pd.concat([combined, pd.DataFrame([current_bar])], ignore_index=True)

    combined.to_csv(f"{SYMBOL}_realtime_bars.csv", index=False)

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
    run_ws()
    print("WS thread terminated")