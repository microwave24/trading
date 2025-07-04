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

UNCERTAINTY_THRESHOLD = 0.0  # percentage threshold for uncertainty in MACD crossovers

MINUTES = 1
HOURS = 1

PERCENTAGE_PROFIT = 1.07  # profit threshold


START_DAY_AGO = 14
END_DAY_AGO = 0.5  # 0.5 days ago, so we get the (almost) latest data


def MACD(data, short_window=SHORT_MA_PERIOD, long_window=LONG_MA_PERIOD):
    """
    Calculate MACD and Signal line.
    """
    data['Short_MA'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['Long_MA'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['Short_MA'] - data['Long_MA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data

def get_signals(data):
    """
    Generate buy/sell signals based on MACD crossovers.
    Only marks the **first** point where MACD crosses Signal Line.
    Only trades between market open (9:30) and market close (15:30) Eastern Time.
    """

    data['Signal'] = 0
    in_position = False
    capital = 10000  # Initial capital for trading simulation
    units = 0
    wins = 0
    total_trades = 0
    entry_price = 0

    prev_relation = None  # Tracks whether MACD was above or below Signal Line

    for i in range(LONG_MA_PERIOD, len(data)):
        

        price = data['open'].iloc[i]
        high = data['high'].iloc[i]
        macd_prev = data['MACD'].iloc[i - 1]
        signal_prev = data['Signal_Line'].iloc[i - 1]

        macd_signal_diff = abs(macd_prev - signal_prev)

        if macd_signal_diff < UNCERTAINTY_THRESHOLD:
            current_relation = "neutral"
        elif macd_prev > signal_prev:
            current_relation = "bullish"
        else:
            current_relation = "bearish"

        # Detect first bullish crossover
        if current_relation == "bullish" and prev_relation == "bearish" and not in_position:
            entry_price = price
            data.at[data.index[i], 'Signal'] = 1
            in_position = True
            units = capital // price
            capital -= units * price
            total_trades += 1

        # Detect first bearish crossover after being bullish
        elif current_relation == "bearish" and prev_relation == "bullish" and in_position:
            if price > entry_price:
                wins += 1
            data.at[data.index[i], 'Signal'] = -1
            in_position = False
            capital += units * price
            units = 0

        # Optional profit take
        elif high > entry_price * PERCENTAGE_PROFIT and in_position:
            if price > entry_price:
                wins += 1
            data.at[data.index[i], 'Signal'] = -1
            in_position = False
            capital += units * entry_price * PERCENTAGE_PROFIT
            units = 0

        prev_relation = current_relation  # Update for next iteration

    # Final liquidation if still holding
    if in_position:
        capital += units * data['close'].iloc[-1]
        data.at[data.index[-1], 'Signal'] = -1

    print(f"Total trades: {total_trades}, Wins: {wins}, Win rate: {wins / total_trades if total_trades > 0 else 0:.2%}")
    print(f"Final capital: ${capital:.2f}")

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
    get_signals(data)

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

    buy_signal_plot = data['Open'].copy()
    buy_signal_plot[~(data['Signal'] == 1)] = np.nan  # Set non-buy signals to NaN

    sell_signal_plot = data['Open'].copy()
    sell_signal_plot[~(data['Signal'] == -1)] = np.nan  # Set non-sell signals to NaN

    buy_plot = mpf.make_addplot(
        buy_signal_plot,
        type='scatter',
        markersize=50,
        marker='^',
        color='g'
    )

    sell_plot = mpf.make_addplot(
        sell_signal_plot,
        type='scatter',
        markersize=50,
        marker='v',
        color='r'
    )

    addplots = [macd_plot, signal_plot, buy_plot, sell_plot]
    
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