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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers, models, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, Dense

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def get_historical_data(api_key, secret_key, symbol, startdate, endDate, daily=False, t=5):
    client = StockHistoricalDataClient(api_key, secret_key)
    timeframe = TimeFrame.Day if daily else TimeFrame(t, TimeFrameUnit.Week)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=startdate,
        end=endDate
    )
    bars = client.get_stock_bars(request)
    df = pl.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    df.write_csv(f"historic/{symbol}_historic_data_1week.csv")
    return df

def process(file, name):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    ny_tz = pytz.timezone('America/New_York')
    df['timestamp'] = df['timestamp'].dt.tz_convert(ny_tz)

    start_time = time(8, 0, 0)
    end_time = time(17, 0, 0)

    #mask = df['timestamp'].dt.time.between(start_time, end_time)
    #df = df.loc[mask].reset_index(drop=True)

    # Create a new DataFrame with the timestamp and empty columns
    df_processed = pd.DataFrame({
        "timestamp": df["timestamp"],
        "p_change": [0.0] * len(df),
        "vwap_change": [0.0] * len(df),
        "vwap_distance": [0.0] * len(df),
        "volume_change": [0.0] * len(df)
    })

    print(len(df))
    for i in range(1, len(df)):  # start from 1 to avoid index error
        
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        p_change = ((row['close'] - row['open']) / row['open']) * 100
        vwap_change = ((row['vwap'] - prev_row['vwap']) / prev_row['vwap']) * 100
        vwap_distance = ((row['close'] - row['vwap']) / row['vwap']) * 100
        volume_change = ((row['volume'] - prev_row['volume']) / prev_row['volume']) * 100

        df_processed.at[i, "p_change"] = p_change
        df_processed.at[i, "vwap_change"] = vwap_change
        df_processed.at[i, "vwap_distance"] = vwap_distance
        df_processed.at[i, "volume_change"] = volume_change

    features = ['p_change', 'vwap_change', 'vwap_distance', 'volume_change']
    scaler = StandardScaler()
    df_processed[features] = scaler.fit_transform(df_processed[features])


    df_training_data = pd.DataFrame({
        "target": [0.0] * len(df_processed),
        "p_change_series": [None] * len(df_processed),
        "vwap_change_series": [None] * len(df_processed),
        "vwap_distance_series": [None] * len(df_processed),
        "volume_change_series": [None] * len(df_processed)
    })

    window_size = 20

    for i in range(window_size - 1, len(df_processed) - 1):
        # Target is next candle's p_change
        df_training_data.at[i, "target"] = df_processed.iloc[i + 1]["p_change"]

        # Last 20 candles including current
        df_training_data.at[i, "p_change_series"] = df_processed.iloc[i - window_size + 1 : i + 1]["p_change"].tolist()
        df_training_data.at[i, "vwap_change_series"] = df_processed.iloc[i - window_size + 1 : i + 1]["vwap_change"].tolist()
        df_training_data.at[i, "vwap_distance_series"] = df_processed.iloc[i - window_size + 1 : i + 1]["vwap_distance"].tolist()
        df_training_data.at[i, "volume_change_series"] = df_processed.iloc[i - window_size + 1 : i + 1]["volume_change"].tolist()

    # drop rows before window_size - 1 since those have None series
    df_training_data = df_training_data.iloc[window_size - 1 : -1].reset_index(drop=True)

    df_training_data.to_csv(f"processed/processed_output_{name}.csv", index=False)
    return df_processed

def train(df):
    for col in ['p_change_series', 'vwap_change_series', 'vwap_distance_series', 'volume_change_series']:
        df[col] = df[col].apply(ast.literal_eval)

    X = np.stack([
        df['p_change_series'].to_list(),
        df['vwap_distance_series'].to_list(),
    ], axis=-1)  # shape: (samples, 20, 4)

    
        #df['vwap_change_series'].to_list(),
        #df['vwap_distance_series'].to_list(),
        #df['volume_change_series'].to_list()

    y = df['target'].values

    # === FIX: Standardize target ===
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(128, input_shape=(20, 2)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'), 
        Dropout(0.1),
        Dense(1)
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mae']
    )

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2
    )

    # === Evaluate on scaled and inverse the result ===
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred_scaled = model.predict(X_test)

    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    




if __name__ == "__main__":
    startdate=datetime(2020, 1, 8, 7, 55, 0)
    #endDate=datetime(2025, 7, 10, 23, 55, 0)

    #get_historical_data(API_KEY, SECRET, "TQQQ", startdate, endDate, t=1)
    df_train_data = process("historic/TQQQ_historic_data_1day.csv", "TQQQ_1day")
    df_train_data = pd.read_csv("processed/processed_output_TQQQ_1day.csv")
    train(df_train_data)


