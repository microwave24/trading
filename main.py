import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
import mplfinance as mpf
import numpy as np

API_KEY = "PKGJ4AHVH5N96O2V29QV"
SECRET = "T8TcUvDr9CbO9y5g9i6ecPkgNxddR6fGf5gcLUQH"

TARGET_SYMBOL = "SOXL"
SHORT_MA_PERIOD = 8
LONG_MA_PERIOD = 30
MIN_SHORT_DELTA = 0  # Minimum short delta to consider a buy signal
MAX_SHORT_DELTA = 10
DELTA_DISTANCE = 8 # the number of bars we are looking back
MAX_MA_DISTANCE = 0.05  # Maximum distance between short and long MA to consider a buy signal
MINUTES = 10

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

def get_historical_data(api_key, secret_key, symbol, startdate=None, endDate=None):
    # Create the client
    client = StockHistoricalDataClient(api_key, secret_key)

    # Create the request
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(MINUTES, TimeFrameUnit.Minute),  # 10m bars
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

def get_short_deltas(data):
    # Calculate short deltas

    short_deltas = [0] * len(data)
    for i in range(SHORT_MA_PERIOD + DELTA_DISTANCE, len(data)):
        short_deltas[i] = data['Short MA'].iloc[i - DELTA_DISTANCE] - data['Short MA'].iloc[i]
    return short_deltas

def get_MA_distance(data):
    # Calculate the distance between short and long moving averages
    ma_distance = [999] * len(data)
    for i in range(LONG_MA_PERIOD, len(data)):
        ma_distance[i] = abs(data['Short MA'].iloc[i] - data['Long MA'].iloc[i])
    return ma_distance
    
    
def get_buy_signal(data):
    # Check for buy signal
    buy = [0] * len(data)
    for i in range(LONG_MA_PERIOD - 1, len(data)):
        if data['Long Positions'].iloc[i] == 1 and data['Short Deltas'].iloc[i] > MIN_SHORT_DELTA and data['Short Deltas'].iloc[i] < MAX_SHORT_DELTA and data['MA Distance'].iloc[i] < MAX_MA_DISTANCE:
            buy[i] = 1
    return buy




# MAIN EXECUTION
def main():
    historical_data = get_historical_data(API_KEY, SECRET, TARGET_SYMBOL, datetime.now() - timedelta(days=4), datetime.now() - timedelta(days=0.5))

    if historical_data is None:
        return

    # Ensure datetime index
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    historical_data.set_index('timestamp', inplace=True)

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

    # Highlight growing deltas WHEN short ma is above long ma
    historical_data['Buy Signal'] = get_buy_signal(historical_data)

    # Plot blue markers at buy signal points (at close price)
    buy_signal_prices = pd.Series(np.nan, index=historical_data.index)
    buy_signal_prices[historical_data['Buy Signal'] == 1] = historical_data['open']

    

    buy_markers = mpf.make_addplot(
        buy_signal_prices,
        type='scatter',
        markersize=50,
        marker='o',
        color='blue',
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

    if historical_data['Buy Signal'].sum() < 1:
        print("No buy signals found in the data.")
        addplots = [MAshort, MAlong, long_markers]
    else:
        print("At least one buy signal found.")
        addplots = [MAshort, MAlong, long_markers, buy_markers]
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
