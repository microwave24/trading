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

API_KEY = "PKAXTN4AN4K04RUYOH45"
SECRET = "vyFgaaKkid9sjZpyqcPafiUeQ5Fqkvgzp7KZga7a"


TARGET_SYMBOL = "SOXL"
SHORT_MA_PERIOD = 12
LONG_MA_PERIOD = 28
MINUTES = 10

SAR_CYCLE = 100
SAR_STEP = 0.5
SAR_LIMIT = 5

PERCENTAGE_PROFIT = 1.005  # profit threshold
PERCENTAGE_LOSS = 0.995  # loss threshold

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

def MACD_signal(data, short_window=SHORT_MA_PERIOD, long_window=LONG_MA_PERIOD):
    current_macd = data['MACD'].iloc[-1]
    current_signal = data['Signal_Line'].iloc[-1]

    return current_macd > current_signal

def SAR(data, cycle=SAR_CYCLE, step=SAR_STEP, limit=SAR_LIMIT):
    high = data['high'].values
    low = data['low'].values
    sar = [0.0] * len(data)
    direction = [1] * len(data)  # 1 for uptrend, -1 for downtrend
    trend = 1

    #init SAR and EP
    sar[cycle] = min(low[:cycle])
    ep = high[cycle]  # Extreme Point
    af = step  # Acceleration Factor

    for i in range(cycle + 1, len(data)):
        #calculate SAR
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if trend == 1:
            sar[i] = min(sar[i], low[i - 1], low[i - 2])  # SAR cannot be above the lowest low
        else:
            sar[i] = max(sar[i], high[i - 1], high[i - 2])

        #check for trend reversal
        if trend == 1:
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]  # Reset EP to the lowest low
                af = step  # Reset AF
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, limit)  # Increase AF but cap it at limit
        else:
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = step  # Reset AF
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, limit)
        direction[i] = trend
    data['SAR'] = sar
    data['SAR_Direction'] = direction
    return data

def SAR_signal(data, cycle=SAR_CYCLE, step=SAR_STEP, limit=SAR_LIMIT):
    prev_direction = data['SAR_Direction'].iloc[-2]
    current_direction = data['SAR_Direction'].iloc[-1]

    return prev_direction == -1 and current_direction == 1  # Downtrend to Uptrend (BUY) signal




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


def get_signals(data):
    """
    Generate buy and sell signals based on MACD.
    Simulates full-port trading with $10,000 starting capital.
    Tracks win rate, capital growth, and auto-liquidates at end.
    """

    capital = 10000
    in_position = False
    entry_price = 0.0
    units = 0
    wins = 0
    total_trades = 0

    signals = [0] * len(data)

    for i in range(LONG_MA_PERIOD, len(data)):
        price = data['open'].iloc[i]

        # Entry Condition
        if not in_position and MACD_signal(data.iloc[:i]):
            signals[i] = 1
            entry_price = price
            units = capital // price
            capital -= units * price
            in_position = True
            total_trades += 1
            continue

        # Exit Condition (TP or SL)
        if in_position:
            target_price = entry_price * PERCENTAGE_PROFIT
            stop_price = entry_price * PERCENTAGE_LOSS

            if price >= target_price or price <= stop_price:
                signals[i] = -1
                sell_price = price
                capital += units * sell_price
                units = 0
                in_position = False

                if sell_price >= target_price:
                    wins += 1

    # Final liquidation if still holding
    if in_position:
        final_price = data['open'].iloc[-1]
        capital += units * final_price
        signals[-1] = -1  # Mark final sell
        units = 0
        in_position = False

    data['Signal'] = signals

    # Summary
    print(f"Final capital: ${capital:.2f}")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Win rate: {(wins / total_trades * 100):.2f}%" if total_trades > 0 else "Win rate: N/A")

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
        daily=False
    )

    # Ensure datetime index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    MACD(data, SHORT_MA_PERIOD, LONG_MA_PERIOD)
    SAR(data, SAR_CYCLE, SAR_STEP, SAR_LIMIT)

    # Generate buy/sell signals and add to DataFrame
    get_signals(data)


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

    buy_signal_plot = data['Close'].copy()
    buy_signal_plot[~(data['Signal'] == 1)] = np.nan  # Set non-buy signals to NaN

    sell_signal_plot = data['Close'].copy()
    sell_signal_plot[~(data['Signal'] == -1)] = np.nan  # Set non-sell signals to NaN

    buy_plot = mpf.make_addplot(
        buy_signal_plot,
        type='scatter',
        markersize=50,
        marker='o',
        color='g'
    )

    sell_plot = mpf.make_addplot(
        sell_signal_plot,
        type='scatter',
        markersize=50,
        marker='o',
        color='r'
    )
    
    
    # Ensure all required columns are present and numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with NaNs in any of those columns
    data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    addplots = [macd_plot, signal_plot, buy_plot, sell_plot]


    # Export data to CSV
    data.to_csv(f"{TARGET_SYMBOL}_historical_data.csv")
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