import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
import mplfinance as mpf
import numpy as np
import warnings

API_KEY = ""
SECRET = ""


TARGET_SYMBOL = "SOXL"
SHORT_MA_PERIOD = 12
LONG_MA_PERIOD = 26

bollinger_window = 200  # Bollinger Bands window size
bollinger_std_dev = 1.5 # Standard deviation for Bollinger Bands

MACD_UNCERTAINTY_THRESHOLD = 0.0  # percentage threshold for uncertainty in MACD crossovers
PRICE_UNCERTAINTY_THRESHOLD = 0.0  # percentage threshold for uncertainty in price movements

MACD_SENSITIVITY = 0  # percentage threshold for sensitivity in MACD crossovers
PRICE_SENSITIVITY = 0  # percentage threshold for sensitivity in price movements

MINUTES = 1
HOURS = 1

PERCENTAGE_PROFIT = 1.01  # profit threshold
PERCENTAGE_LOSS = 0.95  # loss threshold


START_DAY_AGO = 220
END_DAY_AGO = 0.5  # 0.5 days ago, so we get the (almost) latest data

def bollinger_bands(data, window=bollinger_window, std_dev=bollinger_std_dev):
    """
    Calculate Bollinger Bands and add 'SMA', 'Upper_Band', 'Lower_Band' columns to the DataFrame.
    """
    sma = data['close'].rolling(window=window).mean()
    rstd = data['close'].rolling(window=window).std()

    data['SMA'] = sma
    data['Upper_Band'] = sma + std_dev * rstd
    data['Lower_Band'] = sma - std_dev * rstd

    return data

def MACD(data, short_window=SHORT_MA_PERIOD, long_window=LONG_MA_PERIOD):
    """
    Calculate MACD and Signal line.
    """
    data['Short_MA'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['Long_MA'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['Short_MA'] - data['Long_MA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    return data



def get_historical_data(api_key, secret_key, symbol, startdate=None, endDate=None, Hourly=False):
    # Create the client
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = None

    # Create the request
    if Hourly:  # If hourly data is requested
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(HOURS, TimeFrameUnit.Hour),
            start=startdate,
            end=endDate
        )
    else:
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(MINUTES, TimeFrameUnit.Minute),
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

def detect_swing(period, column, bloom, window=30):
    """
    Use the derivative to detect the general trend
    """
    ma = period[column].rolling(window=window).mean()
    deltas = ma.diff()
    avg_deltas = deltas.mean()

    dir = 0
    if avg_deltas > bloom:
        dir = 1
    elif avg_deltas < bloom:
        dir = -1
    return dir


def backtest(data):
    """
    Forward test the strategy using the historical data.
    This function should implement the logic to simulate trades based on the historical data.

    1. Identify entry points based on MACD crossovers and Bollinger Bands.
    2. Move a window through the data to simulate trades.
    3. Apply the strategy logic to determine when to buy/sell on said window.
    4. Track performance metrics such as profit/loss, win rate, etc.
    5. Return a summary of the performance metrics.
    """
    in_position = False
    starting_capital = 10000  # Starting capital for the strategy
    entry_price = 0
    units = 0

    macd_df = data[['MACD_Histogram']].copy()
    macd_df['bearish_h'] = data['MACD_Histogram'].apply(lambda x: x if x < 0 else 0)
    macd_df['bullish_h'] = data['MACD_Histogram'].apply(lambda x: x if x > 0 else 0)

    signals = [0] * len(data)
    
    
    for i in range(bollinger_window + 1, len(data)):
        # Skip if timestamp is not between 8:30 am and 3:30 pm
        current_time = data.index[i].time()
        prev_close = data['close'].iloc[i-1]
        current_open = data['open'].iloc[i]

        if((current_open > entry_price * PERCENTAGE_PROFIT and in_position == True) or (in_position == True and current_open < entry_price * PERCENTAGE_LOSS) or (i == len(data) - 1 and in_position == True)):
            # Sell signal
            starting_capital += units * current_open
            signals[i] = -1
            in_position = False
        
        if not (current_time >= datetime.strptime("08:30", "%H:%M").time() and current_time <= datetime.strptime("15:30", "%H:%M").time() and not in_position):
            continue

        

        #first requirement: outside the Bollinger Bands
        if (prev_close < data['Lower_Band'].iloc[i] and in_position == False): # bullish signal

            if data['MACD_Histogram'].iloc[i - 1] <= data['MACD_Histogram'].iloc[i - 2]:  # MACD must be bullish
                continue

            MACD_direction = detect_swing(macd_df.iloc[i-80:i], 'bullish_h',bloom=MACD_SENSITIVITY , window=20)  # Detect bullish divergence
            Price_directrion = detect_swing(data.iloc[i-80:i], 'close',bloom=PRICE_SENSITIVITY, window=20)  # Detect price direction
            

            if(MACD_direction == 1 and Price_directrion == -1 and in_position == False):
                units = starting_capital // current_open  # Calculate how many units we can buy
                starting_capital -= units * current_open
                entry_price = current_open
                signals[i] = 1  # Buy signal
                in_position = True
        
                
    print(f"Final capital: {starting_capital}")
    print(f"Total trades: {signals.count(1)}")
    return signals
            

if __name__ == "__main__":
    # Define the start and end dates for the historical data
    start_date = datetime.now() - timedelta(days=START_DAY_AGO)
    end_date = datetime.now() - timedelta(days=END_DAY_AGO)


    data = get_historical_data(
        API_KEY,
        SECRET,
        TARGET_SYMBOL,
        startdate=start_date,
        endDate=end_date,
        Hourly=False
    )

    # Ensure datetime index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    MACD(data, SHORT_MA_PERIOD, LONG_MA_PERIOD)
    bollinger_bands(data, bollinger_window, bollinger_std_dev)

    data['Signals'] = backtest(data)
    data.to_csv(f"{TARGET_SYMBOL}_historical_data.csv")

    # Rename columns to match mplfinance
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    
    # Prepare MACD and Signal Line for plotting
    macd_plot = mpf.make_addplot(data['MACD'], panel=1, color='b', ylabel='MACD')
    signal_plot = mpf.make_addplot(data['Signal_Line'], panel=1, color='r')

    macd_hist = data['MACD'] - data['Signal_Line']
    # Color MACD histogram green for positive, red for negative
    hist_colors = ['g' if val >= 0 else 'r' for val in macd_hist]
    hist_plot = mpf.make_addplot(macd_hist, panel=1, type='bar', color=hist_colors, alpha=0.5)

    bb_mid = mpf.make_addplot(data['SMA'], color='blue', width=1)
    bb_upper = mpf.make_addplot(data['Upper_Band'], color='red', width=0.75)
    bb_lower = mpf.make_addplot(data['Lower_Band'], color='green', width=0.75)

    # Create aligned Series for buy/sell signals
    buy_marker = pd.Series(np.nan, index=data.index)
    sell_marker = pd.Series(np.nan, index=data.index)

    # Only plot buy/sell signals if there are any nonzero signals
    if (data['Signals'] == 1).any():
        buy_marker[data['Signals'] == 1] = data['Open']
        buy_plot = mpf.make_addplot(
            buy_marker,
            type='scatter',
            markersize=100,
            marker='^',
            color='lime',
            panel=0
        )
    else:
        buy_plot = None

    if (data['Signals'] == -1).any():
        sell_marker[data['Signals'] == -1] = data['Open']
        sell_plot = mpf.make_addplot(
            sell_marker,
            type='scatter',
            markersize=100,
            marker='v',
            color='red',
            panel=0
        )
    else:
        sell_plot = None

    # Filter out None plots from addplots
    addplots = [macd_plot, signal_plot, hist_plot, bb_mid, bb_upper, bb_lower]
    if buy_plot is not None:
        addplots.append(buy_plot)
    if sell_plot is not None:
        addplots.append(sell_plot)
    
    # Ensure all required columns are present and numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with NaNs in any of those columns
    data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    # Export data to CSV
    
    # Plot using mplfinance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpf.plot(
            data,
            type='candle',
            style='charles',
            title=f"{TARGET_SYMBOL} Candlestick Chart",
            volume=False,
            ylabel='Price',
            ylabel_lower='Volume',
            addplot=addplots
        )