import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
import mplfinance as mpf
import numpy as np
import pytz
from pytz import timezone

API_KEY = ""
SECRET = ""

TARGET_SYMBOL = "SOXL"
SHORT_MA_PERIOD = 5
LONG_MA_PERIOD = 20
MIN_SHORT_DELTA = -0.1  # Minimum short delta to consider a buy signal
MAX_SHORT_DELTA = 0.1
DELTA_DISTANCE = 1 # the number of bars we are looking back
MAX_MA_DISTANCE = 0.1  # Maximum distance between short and long MA to consider a buy signal
MINUTES = 59

PERCENTAGE_PROFIT = 1.05  # profit threshold
PERCENTAGE_LOSS = 0.95  # loss threshold

START_DAY_AGO = 220
END_DAY_AGO = 0.5  # 0.5 days ago, so we get the (almost) latest data

# REAL TIME DATA 
def start_quote_stream(api_key, secret_key, symbol):
    # Create the WebSocket client
    wss_client = StockDataStream(api_key, secret_key)

    # async handler
    async def quote_data_handler(data):
        print(data)

    # Subscribe to quotes
    wss_client.subscribe_quotes(quote_data_handler, symbol)

    # Run the WebSocket client
    wss_client.run()


# HISTORICAL DATA

def get_historical_data(api_key, secret_key, symbol, startdate=None, endDate=None, daily=False):
    # Create the client
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = None

    # Create the request
    if daily: # If daily data is requested, we get daily bars
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,  # Daily bars
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

def get_long_posiitions(data, longWindow=LONG_MA_PERIOD):
    # Identify long positions
    long_positions = [0] * len(data)
    for i in range(longWindow, len(data)):
        if (data['Short MA'].iloc[i] > data['Long MA'].iloc[i] and
            data['Short MA'].iloc[i - 1] >= data['Long MA'].iloc[i - 1]):
            long_positions[i] = 1  # Long position entry

    return long_positions

def derivative(current, previous, current_x, previous_x):
    return (current - previous) / (current_x - previous_x)

def get_short_deltas(data):
    # Calculate short deltas
    short_deltas = [0] * len(data)
    for i in range(SHORT_MA_PERIOD + DELTA_DISTANCE, len(data)):
        short_deltas[i] = derivative(data['Short MA'].iloc[i], data['Short MA'].iloc[i - DELTA_DISTANCE], i, i-DELTA_DISTANCE)
    return short_deltas

def get_MA_distance(data):
    # Calculate the distance between short and long moving averages
    ma_distance = [999] * len(data)
    for i in range(LONG_MA_PERIOD, len(data)):
        ma_distance[i] = abs(data['Short MA'].iloc[i] - data['Long MA'].iloc[i])
    return ma_distance
    
    
def get_signals(data):
    capital = 10000  # Starting capital for the strategy
    wins = 0
    total_trades = 0
    units = 0
    in_position = False
    entry_price = 0
    signals = [0] * len(data)
    for i in range(LONG_MA_PERIOD, len(data)):
        if data['Long Positions'].iloc[i - 1] == 1 and data['Short Deltas'].iloc[i - 1] > MIN_SHORT_DELTA and data['Short Deltas'].iloc[i - 1] < MAX_SHORT_DELTA and data['MA Distance'].iloc[i - 1] < MAX_MA_DISTANCE:
            if not in_position:
                signals[i] = 1  # Buy signal
                capital-=1 # small fee for buying
                units = capital // data['open'].iloc[i] # this needs to change to realtime price
                capital -= units * data['open'].iloc[i]
                

                entry_price = data['open'].iloc[i]
                in_position = True
                total_trades += 1
        elif in_position:
            # Check for exit conditions
            if data['high'].iloc[i] >= entry_price * PERCENTAGE_PROFIT or data['open'].iloc[i] <= entry_price * PERCENTAGE_LOSS: # this needs to change to realtime price
                #print(f"Trade closed at {data['open'].iloc[i]} with entry price {entry_price}")
                if data['high'].iloc[i] >= entry_price * PERCENTAGE_PROFIT:
                    capital += units * entry_price * 1.005 # this as well
                    wins += 1
                else:
                    capital += units * data['open'].iloc[i]
                capital-= 1 # small fee for selling
                
                units = 0

                signals[i] = -1
                in_position = False
        if i == len(data) - 1 and in_position:
            capital = units * entry_price

    print(f"Final capital: {capital}, winrate: {wins / total_trades if total_trades > 0 else 0:.2f}, total trades: {total_trades}, daily trades: {(total_trades / START_DAY_AGO):.2f}")
    return signals



# MAIN EXECUTION
def main():
    historical_data = get_historical_data(API_KEY, SECRET, TARGET_SYMBOL, datetime.now() - timedelta(days=START_DAY_AGO), datetime.now() - timedelta(days=END_DAY_AGO))
    #daily_data = get_historical_data(API_KEY, SECRET, TARGET_SYMBOL, datetime.now() - timedelta(days=START_DAY_AGO), datetime.now() - timedelta(days=0.5), daily=True)

    if historical_data is None:
        return

    # Ensure datetime index
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    historical_data.set_index('timestamp', inplace=True)

    #daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])
    #daily_data.set_index('timestamp', inplace=True)
    # Export daily data to CSV
    #daily_data.to_csv(f"{TARGET_SYMBOL}_daily_data.csv")

    historical_data['Short MA'] = historical_data['close'].rolling(window=SHORT_MA_PERIOD).mean().fillna(np.nan)
    historical_data['Long MA']  = historical_data['close'].rolling(window=LONG_MA_PERIOD).mean().fillna(np.nan)
    
    MAshort = mpf.make_addplot(historical_data['Short MA'], color='blue', width=1, panel=0)
    MAlong = mpf.make_addplot(historical_data['Long MA'], color='red', width=1, panel=0)

    historical_data['Long Positions'] = get_long_posiitions(historical_data)
    historical_data['Short Deltas'] = get_short_deltas(historical_data)

    max_high = historical_data['high'].max()

    # Highlight green markers where 'Long Positions' == 1
    long_entry_prices = pd.Series(np.nan, index=historical_data.index)
    long_entry_prices[historical_data['Long Positions'] == 1] = max_high

    historical_data['MA Distance'] = get_MA_distance(historical_data)

    historical_data['Signals'] = get_signals(historical_data)

    # Create green arrows for buy signals (Signals == 1)
    buy_arrows = pd.Series(np.nan, index=historical_data.index)
    buy_arrows[historical_data['Signals'] == 1] = historical_data['open']

    # Create red arrows for sell signals (Signals == -1)
    sell_arrows = pd.Series(np.nan, index=historical_data.index)
    sell_arrows[historical_data['Signals'] == -1] = historical_data['open']

    buy_marker = mpf.make_addplot(
        buy_arrows,
        type='scatter',
        markersize=100,
        marker='^',
        color='green',
        panel=0
    )
    sell_marker = mpf.make_addplot(
        sell_arrows,
        type='scatter',
        markersize=100,
        marker='v',
        color='red',
        panel=0
    )

    

    long_markers = mpf.make_addplot(
        long_entry_prices,
        type='scatter',
        markersize=100,
        marker='s',
        color='green',
        panel=0
    )
    # Export historical data to CSV
    historical_data.to_csv(f"{TARGET_SYMBOL}_historical_data.csv")


    # Rename columns to match mplfinance
    historical_data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    addplots = [MAshort, MAlong, long_markers, buy_marker, sell_marker]

    import warnings
    # Suppress warnings from mplfinance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Plot with mplfinance (no volume)
        mpf.plot(
            historical_data,
            type='candle',
            style='charles',
            title=f"{TARGET_SYMBOL} Candlestick Chart",
            addplot=addplots,
            volume=False
        )

if __name__ == "__main__":
    main()
