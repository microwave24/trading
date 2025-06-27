import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
import mplfinance as mpf

API_KEY = "xxxx"
SECRET = "xxxx"

TARGET_SYMBOL = "SOXL"
SHORT_MA_PERIOD = 8
LONG_MA_PERIOD = 30

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
        timeframe=TimeFrame(10, TimeFrameUnit.Minute),  # 10m bars
        start=startdate,
        end= endDate 
    )

    # Fetch bars
    bars = client.get_stock_bars(request_params)
    # Convert the list of bars to a DataFrame
    data = pd.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    return data

# MAIN EXECUTION
def main():
    historical_data = get_historical_data(API_KEY, SECRET, TARGET_SYMBOL, datetime.now() - timedelta(days=2), datetime.now() - timedelta(days=1))

    if historical_data is None:
        return

    # Ensure datetime index
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    historical_data.set_index('timestamp', inplace=True)

    historical_data['Short MA'] = historical_data['close'].rolling(window=SHORT_MA_PERIOD).mean()
    historical_data['Long MA'] = historical_data['close'].rolling(window=LONG_MA_PERIOD).mean()
    MAshort = mpf.make_addplot(historical_data['Short MA'], color='blue', width=1, panel=0)
    MAlong = mpf.make_addplot(historical_data['Long MA'], color='red', width=1, panel=0)
    

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
        addplot=[MAshort, MAlong],
        volume=False
    )

    

if __name__ == "__main__":
    main()
