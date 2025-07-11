# btc_data.py

import requests
import pandas as pd
import time
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

BINANCE_BASE_URL = "https://api.binance.com"

def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=100):
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from Binance.

    Args:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").
        interval (str): The candlestick interval (e.g., "1h", "4h", "1d").
        limit (int): The number of data points to fetch.

    Returns:
        pandas.DataFrame: DataFrame containing OHLCV data with timestamp as index.
                          Returns an empty DataFrame on failure.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if not data:
            print(f"No data received for {symbol} with interval {interval}.")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        # Convert relevant columns to numeric, coercing errors to NaN
        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["high"] = pd.to_numeric(df["high"], errors='coerce')
        df["low"] = pd.to_numeric(df["low"], errors='coerce')
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
        df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
        
        # Drop rows with any NaN values that resulted from coercion
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

        return df[["open", "high", "low", "close", "volume"]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OHLCV data from Binance: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error parsing JSON data from Binance: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred in fetch_ohlcv: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """
    Adds Relative Strength Index (RSI), EMA 20, and EMA 50 to the DataFrame.
    Also adds candle body size and wick ratio for volatility analysis.

    Args:
        df (pandas.DataFrame): DataFrame with 'close' prices.

    Returns:
        pandas.DataFrame: DataFrame with added technical indicator columns.
    """
    if df.empty or 'close' not in df.columns or 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        print("DataFrame is empty or missing required OHLCV columns for indicator calculation.")
        return df

    # Ensure relevant columns are numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    if df.empty:
        return df

    # Calculate RSI
    if len(df) >= 14:
        rsi = RSIIndicator(df["close"], window=14).rsi()
        df["RSI"] = rsi
    else:
        df["RSI"] = float('nan')

    # Calculate EMA 20
    if len(df) >= 20:
        ema_20 = EMAIndicator(df["close"], window=20).ema_indicator()
        df["EMA_20"] = ema_20
    else:
        df["EMA_20"] = float('nan')

    # Calculate EMA 50
    if len(df) >= 50:
        ema_50 = EMAIndicator(df["close"], window=50).ema_indicator()
        df["EMA_50"] = ema_50
    else:
        df["EMA_50"] = float('nan')

    # --- New: Candle Volatility Metrics ---
    # Candle body size (absolute difference between open and close)
    df['body_size'] = abs(df['close'] - df['open'])

    # Total candle range (High - Low)
    df['candle_range'] = df['high'] - df['low']

    # Upper wick: High - max(Open, Close)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    # Lower wick: min(Open, Close) - Low
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Total wick size
    df['total_wick_size'] = df['upper_wick'] + df['lower_wick']

    # Wick to body ratio (avoid division by zero)
    df['wick_to_body_ratio'] = df.apply(
        lambda row: row['total_wick_size'] / row['body_size'] if row['body_size'] > 0 else (1.0 if row['total_wick_size'] > 0 else 0.0),
        axis=1
    )
    # --- End New Metrics ---

    return df

def get_current_price(symbol="BTCUSDT"):
    """
    Fetches the current price of a given symbol from Binance.

    Args:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").

    Returns:
        float: The current price. Returns 0.0 on failure.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=5) # Add timeout
        response.raise_for_status()
        return float(response.json()["price"])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current price from Binance: {e}")
        return 0.0
    except (ValueError, KeyError) as e:
        print(f"Error parsing current price data: {e}")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred in get_current_price: {e}")
        return 0.0

if __name__ == "__main__":
    print("--- Testing btc_data.py ---")
    df = fetch_ohlcv()
    if not df.empty:
        df = add_technical_indicators(df)
        print("\nLast 5 OHLCV with Indicators and Volatility Metrics:")
        print(df[['close', 'RSI', 'EMA_20', 'EMA_50', 'body_size', 'candle_range', 'wick_to_body_ratio']].tail())
    else:
        print("Failed to fetch OHLCV data for testing.")

    current_price = get_current_price()
    print(f"\nCurrent BTC Price: ${current_price}")

