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
bollinger_std_dev = 2  # Standard deviation for Bollinger Bands

MACD_UNCERTAINTY_THRESHOLD = 0.0  # percentage threshold for uncertainty in MACD crossovers
MACD_SENSITIVITY = 1.0  # percentage threshold for sensitivity in MACD crossovers
PRICE_UNCERTAINTY_THRESHOLD = 0.0  # percentage threshold for uncertainty in price movements

MINUTES = 1
HOURS = 1

PERCENTAGE_PROFIT = 1.07  # profit threshold


START_DAY_AGO = 4
END_DAY_AGO = 2  # 0.5 days ago, so we get the (almost) latest data

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

    return data



def get_historical_data(api_key, secret_key, symbol, startdate=None, endDate=None, Hourly=False):
    # Create the client
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = None

    # Create the request
    if Hourly: # If daily data is requested, we get daily bars
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(HOURS, TimeFrameUnit.Hour),  # Daily bars
            start=startdate,
            end=endDate
        )
    else:
        request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(MINUTES, TimeFrameUnit.Minute),  # variable minute bars
        start=startdate,
        end= endDate 
    )
    

    # Fetch bars
    bars = client.get_stock_bars(request_params)
    # Convert the list of bars to a DataFrame
    data = pd.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    return data

def get_MACD_clusters(data):
    """
    Identify MACD clusters based on the MACD histogram.
    This function will return a list of arrays where each array contains the indices of the MACD histogram values that are part of a cluster.
    """
    clusters = []
    in_cluster = False
    current_cluster = []

    for i in range(len(data['MACD']) - 1, -1, -1):
        if len(clusters) >= 2:
           break

        if data['MACD'].iloc[i] != 0:
            if not in_cluster:
                in_cluster = True
                current_cluster = [(data.index[i], data['MACD'].iloc[i])]
            else:
                current_cluster.append((data.index[i], data['MACD'].iloc[i]))
        else:
            if in_cluster:
                clusters.append(current_cluster)
                in_cluster = False
                current_cluster = []
    return clusters
    


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
    
    for i in range(bollinger_window + 1, len(data)):
        
        window = data.iloc[i - bollinger_window:i].copy()
        prev_close = window['close'].iloc[-2]
        current_open = data['open'].iloc[i]

        window_MACD = MACD(window, SHORT_MA_PERIOD, LONG_MA_PERIOD)
        window_bollinger = bollinger_bands(window, bollinger_window, bollinger_std_dev)

        # Copy window_MACD and replace any negative values with 0
        # Keep only the index (timestamp) and MACD column
        bullish_MACD = window_MACD[['MACD']].copy()
        bullish_MACD[bullish_MACD['MACD'] < 0] = 0

        bearish_MACD = window_MACD[['MACD']].copy()
        bearish_MACD[bearish_MACD['MACD'] > 0] = 0
        
        bullish_MACD_clusters = get_MACD_clusters(bullish_MACD)
        bearish_MACD_clusters = get_MACD_clusters(bearish_MACD)

        #first requirement: outside the Bollinger Bands
        if (prev_close < window_bollinger['Lower_Band'].iloc[-1]): # bullish signal
            #second requirement: MACD divergence
            #if bearish_MACD['MACD'].iloc[-1] < bearish_MACD['MACD'].iloc[-2]: # the current MACD cluster has not formed a peak peak
                #continue
            
            cluster_lows = [min(cluster, key=lambda x: x[1])[1] for cluster in bearish_MACD_clusters]

            
            MACD_Direction = (cluster_lows[0] - cluster_lows[1])**MACD_SENSITIVITY #[0] is the most recent cluster, [1] is the previous cluster
            period = bearish_MACD_clusters[1][-1][0] # period between the two clusters

            print(f'{period} bullish')
            
            price_difference = data.loc[data.index[i - 1], 'low'] - data.loc[period, 'low']


            if MACD_Direction > MACD_UNCERTAINTY_THRESHOLD and price_difference < PRICE_UNCERTAINTY_THRESHOLD: #MACD is increasing and price is moving down:
                in_position = True
        elif (prev_close > window_bollinger['Upper_Band'].iloc[-1]): # bearish signal

            #if bullish_MACD['MACD'].iloc[-1] > bullish_MACD['MACD'].iloc[-2]: # the current MACD cluster might of formed a peak
                #continue

            cluster_highs = [max(cluster, key=lambda x: x[1])[1] for cluster in bullish_MACD_clusters]

            MACD_Direction = (cluster_highs[0] - cluster_highs[1])**MACD_SENSITIVITY #[0] is the most recent cluster, [1] is the previous cluster
            period = bullish_MACD_clusters[1][-1][0]# period between the two clusters
            print(f'{period} bearish')

            price_difference = data.loc[data.index[i - 1], 'high'] - data.loc[period, 'high']

            if MACD_Direction < -MACD_UNCERTAINTY_THRESHOLD and price_difference > PRICE_UNCERTAINTY_THRESHOLD: #MACD is decreasing and price is moving up:
                in_position = True
        
        if in_position:
            # Check if the price has reached the profit threshold
            if (current_open >= prev_close * PERCENTAGE_PROFIT or current_open > window_bollinger['Upper_Band'].iloc[-2]):
                print(f"Profit taken at {current_open} on {data.index[i]}")
                in_position = False







        


if __name__ == "__main__":
    # Define the start and end dates for the historical data
    start_date = datetime.now() - timedelta(days=START_DAY_AGO)
    end_date = datetime.now() - timedelta(days=END_DAY_AGO)

    

    # Get historical data
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

    backtest(data)
        

    

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



    


    addplots = [macd_plot, signal_plot, hist_plot, bb_mid, bb_upper, bb_lower]
    
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