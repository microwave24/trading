import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import joblib

API_KEY = ""
SECRET = ""

def get_historical_data(api_key, secret_key, symbol, startdate=None, endDate=None, Hourly=False):
    # Create the client
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = None

    # Create the request
    request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=startdate,
            end=endDate
        )

    # Fetch bars
    bars = client.get_stock_bars(request_params)

    # Convert the list of bars to a DataFrame
    data = pd.DataFrame([bar.__dict__ for bar in bars[symbol]])

    # Convert 'timestamp' to UTC-4
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Etc/GMT+4')  # UTC-4 

    print("Historical data retrieved")
    return data

def retrieve_data(symbols):

    start_date = datetime.now() - timedelta(days=3600)  # X years worth of data, in this case 10 years
    end_date = datetime.now() - timedelta(days=1)  # up to yesterday

    for symbol in symbols:
        print(f"Processing {symbol}...")
        data = get_historical_data(API_KEY, SECRET, symbol, startdate=start_date, endDate=end_date)

        # Ensure 'timestamp' is the index
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)

        # Drop any rows with NaN values
        data.dropna(inplace=True)

        # Save the cleaned DataFrame to a CSV file
        csv_filename = f"{symbol}_historical_data.csv"
        data.to_csv(csv_filename)
        print(f"Saved {symbol} data to {csv_filename}")

        # Calculate EMAs for the previous candle
        for period in [10, 20, 50, 100, 150, 200]:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean().shift(1)

        # Calculate MACD and MACD histogram for the previous candle
        exp12 = data['close'].ewm(span=12, adjust=False).mean()
        exp26 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - signal).shift(1)
        data['macd_hist'] = macd_hist

        # Previous open, close, high, low
        data['prev_open'] = data['open'].shift(1)
        data['prev_close'] = data['close'].shift(1)
        data['prev_high'] = data['high'].shift(1)
        data['prev_low'] = data['low'].shift(1)

        # Target variable
        def get_target(row):
            if (row['high'] > row['open'] * 1.01):
                return 1
            else:
                return -1

        data['target'] = data.apply(get_target, axis=1)

        # Drop rows with NaN values from shifting/indicators
        data.dropna(inplace=True)

        # Save the feature set to a new CSV
        features = [
            'prev_open', 'prev_close', 'prev_high', 'prev_low',
            'macd_hist',
            'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_150', 'ema_200',
            'target'
        ]
        feature_csv_filename = f"{symbol}_features.csv"
        data[features].to_csv(feature_csv_filename)
        print(f"Saved {symbol} features to {feature_csv_filename}")

        print

    print("All symbols processed successfully.")
    
def combine(symbols):
    # Combine all processed feature CSVs into one DataFrame
    combined_df = pd.DataFrame()
    for symbol in symbols:
        feature_csv_filename = f"{symbol}_features.csv"
        df_symbol = pd.read_csv(feature_csv_filename)
        df_symbol['symbol'] = symbol
        combined_df = pd.concat([combined_df, df_symbol], ignore_index=True)

    # Save the combined DataFrame to a single CSV
    combined_csv_filename = "all_symbols_features.csv"
    combined_df.to_csv(combined_csv_filename, index=False)
    print(f"Combined all features into {combined_csv_filename}")

    return combined_df

def train(df):
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=1000,       # Number of trees
        max_depth=None,         # Allow full depth
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'               
    )

    features = [
            'prev_open', 'prev_close', 'prev_high', 'prev_low',
            'macd_hist',
            'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_150', 'ema_200'
        ]
    
    df['target'] = df['target'].astype(int)

    # We split the data into training and testing sets
    # 80% for training, 20% for testing
    split_point = int(len(df) * 0.8)

    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    X_train = train_df[features]
    y_train = train_df['target']

    X_test = test_df[features]
    y_test = test_df['target']

    # Fit the model and evaluate
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)
    y_pred = rf_model.predict(X_test)

    joblib.dump(rf_model, 'random_forest_trading_model.pkl')

    print("Model trained and saved as 'random_forest_trading_model.pkl'")
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, y_pred, labels=[-1, 1]))

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    # Visualize it
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

if __name__ == "__main__":

    # Use combined_df for model training
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "AMD", "INTC", "SPY", "QQQ"]
    retrieve_data(symbols)
    df = combine(symbols)
    df.dropna(inplace=True)  # Ensure no NaN values before training
    train(df)



    
    


