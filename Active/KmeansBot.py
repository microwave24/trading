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
API_KEY = "PKHLDVUZSBAVJ0GWCR14"
SECRET = "bK4CterukecekQpcg2ElddIW90MrRvggmYU36Ehg"
# === Data Preperation and Analysis ===
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SOCKET = "wss://stream.data.alpaca.markets/v2/iex"
SYMBOL = 'NVDA'

df = pd.DataFrame(columns=["symbol", "timestamp", "close_price"])

def on_open(ws):
    print("opened")

    auth_data = {
        "action": "auth",
        "key": API_KEY,
        "secret": SECRET
    }
    ws.send(json.dumps(auth_data))

    # subscribe to TSLA trades, quotes, or bars
    listen_message = {
        "action": "subscribe",
        "trades": [SYMBOL],    # trade prints
        "quotes": [SYMBOL],    # bid/ask
        "bars": [SYMBOL]       # OHLCV bars
    }
    ws.send(json.dumps(listen_message))

current_bar = {}
def on_message(ws, message):
    global df, current_bar
    data = json.loads(message)

    for msg in data:
        if msg.get("T") == "t":  # trade
            ts = pd.to_datetime(msg["t"]).tz_convert("America/New_York")
            minute = ts.floor("s")
            print(minute)
            price = round(msg["p"], 3)

            if current_bar.get("timestamp") != minute:
                # save old bar if exists
                if current_bar:
                    df = pd.concat([df, pd.DataFrame([current_bar])], ignore_index=True)
                    df = df.tail(200).reset_index(drop=True)

                # start a new bar
                current_bar = {
                    "timestamp": minute,
                    "close_price": price
                }
            else:
                # update existing bar's close
                current_bar["close_price"] = price
            


            



def on_close(ws, close_status_code, close_msg):
    print("closed connection")


def run_ws():
    ws = websocket.WebSocketApp(SOCKET,
                                on_open=on_open,
                                on_message=on_message,
                                on_close=on_close)
    ws.run_forever()

def animate(i):
    if not df.empty:
        ax.clear()
        ax.plot(df["timestamp"], df["close_price"], marker="o")
        ax.set_title("Live Price Stream")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        fig.autofmt_xdate()


if __name__ == "__main__":
    
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, interval=1000,
                              save_count=200)
    plt.show()