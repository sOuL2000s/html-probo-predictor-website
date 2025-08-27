from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import pandas as pd
import pytz
import numpy as np
import requests
import feedparser
from textblob import TextBlob
import urllib.parse
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from prophet import Prophet

# --- Gemini API Integration ---
import google.generativeai as genai
from PIL import Image # For image handling
import io # For file processing

app = Flask(__name__)
CORS(app)

# --- Global Data Storage ---
# Data for Bitcoin
btc_market_data_df = None
btc_current_price = 0.0
btc_sentiment_score = 0.0
btc_market_conditions = {}

# Data for Ethereum
eth_market_data_df = None
eth_current_price = 0.0
eth_sentiment_score = 0.0
eth_market_conditions = {}

# --- Configuration ---
BINANCE_BASE_URL = "https://api.binance.com"
# Telegram Bot Token and User ID are removed as per request.

# Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCzx6ReMk8ohPJcCjGwHHzu7SvFccJqAbA") # Your provided key
genai.configure(api_key=GEMINI_API_KEY)
# Using gemini-1.5-flash-latest for stability, but user requested "gemini-2.5-flash-preview-05-20"
# Trying with the user's specified model, if it fails, a more stable one might be needed.
GEMINI_PREDICTION_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_CHAT_MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Chat is good with latest flash

# Ensure models are initialized
try:
    gemini_prediction_model = genai.GenerativeModel(GEMINI_PREDICTION_MODEL_NAME)
    gemini_chat_model = genai.GenerativeModel(GEMINI_CHAT_MODEL_NAME)
    print(f"Gemini Prediction Model '{GEMINI_PREDICTION_MODEL_NAME}' initialized.")
    print(f"Gemini Chat Model '{GEMINI_CHAT_MODEL_NAME}' initialized.")
except Exception as e:
    print(f"Error initializing Gemini models: {e}")
    print("Please check your API key and model names. Proceeding without Gemini functionality.")
    gemini_prediction_model = None
    gemini_chat_model = None

# --- Crypto Data Functions ---

def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=200): # Increased limit for better indicator calculation
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from Binance for a given symbol.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
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
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

        return df[["open", "high", "low", "close", "volume"]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OHLCV data from Binance for {symbol}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error parsing JSON data from Binance for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred in fetch_ohlcv for {symbol}: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """
    Adds RSI, EMA 20, EMA 50, MACD, Stochastic Oscillator, Bollinger Bands,
    and candle volatility metrics to the DataFrame.
    """
    if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        print("DataFrame is empty or missing required OHLCV columns for indicator calculation.")
        return df

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

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

    # Calculate MACD
    if len(df) >= 34: # MACD needs 12 (fast EMA) + 26 (slow EMA) periods to stabilize, 34 is safer.
        macd_indicator = MACD(df["close"])
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Hist"] = macd_indicator.macd_diff()
    else:
        df["MACD"] = float('nan')
        df["MACD_Signal"] = float('nan')
        df["MACD_Hist"] = float('nan')

    # Calculate Stochastic Oscillator
    if len(df) >= 14: # Stochastic needs at least 14 periods
        stoch_indicator = StochasticOscillator(df["high"], df["low"], df["close"])
        df["STOCH_K"] = stoch_indicator.stoch()
        df["STOCH_D"] = stoch_indicator.stoch_signal()
    else:
        df["STOCH_K"] = float('nan')
        df["STOCH_D"] = float('nan')

    # Calculate Bollinger Bands
    if len(df) >= 20: # Bollinger Bands typically use 20 periods
        bb_indicator = BollingerBands(df["close"])
        df["BB_Upper"] = bb_indicator.bollinger_hband()
        df["BB_Lower"] = bb_indicator.bollinger_lband()
        df["BB_Mid"] = bb_indicator.bollinger_mavg()
        df["BB_Width"] = bb_indicator.bollinger_wband()
        df["BB_Percent"] = bb_indicator.bollinger_pband() # %B
    else:
        df["BB_Upper"] = float('nan')
        df["BB_Lower"] = float('nan')
        df["BB_Mid"] = float('nan')
        df["BB_Width"] = float('nan')
        df["BB_Percent"] = float('nan')

    # Candle Volatility Metrics
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_wick_size'] = df['upper_wick'] + df['lower_wick']

    df['wick_to_body_ratio'] = df.apply(
        lambda row: row['total_wick_size'] / row['body_size'] if row['body_size'] > 0 else (1.0 if row['total_wick_size'] > 0 else 0.0),
        axis=1
    )
    return df

def get_current_price(symbol="BTCUSDT"):
    """
    Fetches the current price of a given symbol from Binance.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return float(response.json()["price"])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current price for {symbol} from Binance: {e}")
        return 0.0
    except (ValueError, KeyError) as e:
        print(f"Error parsing current price data for {symbol}: {e}")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred in get_current_price for {symbol}: {e}")
        return 0.0

# --- Sentiment Functions (consolidated) ---

def fetch_news_sentiment(query="bitcoin", max_items=20):
    """
    Fetches news headlines for a given query from Google News RSS and calculates
    an average sentiment polarity using TextBlob.
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}"
    
    try:
        feed = feedparser.parse(url)
        headlines = [entry.title for entry in feed.entries[:max_items]]

        if not headlines:
            print(f"No news headlines found for query: '{query}'")
            return 0.0

        sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
        return round(sum(sentiments) / len(sentiments), 3)
    except Exception as e:
        print(f"Error fetching or analyzing news sentiment for '{query}': {e}")
        return 0.0

def get_crypto_sentiment(crypto_name="bitcoin"):
    """
    Gets the sentiment score for a specific cryptocurrency.
    """
    return fetch_news_sentiment(f"{crypto_name} OR {crypto_name[:3].upper()}")

# --- Market Interpretation (consolidated and enhanced) ---

def interpret_market_conditions(df: pd.DataFrame):
    """
    Interprets market conditions based on technical indicators and volatility metrics.
    Incorporates RSI, EMA, MACD, Stochastic, Bollinger Bands, and candle volatility.
    """
    required_cols = [
        'RSI', 'EMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'STOCH_K', 'STOCH_D', 'BB_Upper', 'BB_Lower', 'BB_Mid', 'BB_Width', 'BB_Percent',
        'body_size', 'candle_range', 'wick_to_body_ratio', 'close', 'volume'
    ]
    if df.empty or not all(col in df.columns for col in required_cols):
        print("Warning: DataFrame is empty or missing required columns for market interpretation. Returning default conditions.")
        # Initialize with None to correctly reflect missing data, rather than default values
        return {k: None for k in [
            "bullish_trend", "oversold", "overbought", "rsi", "ema_20", "ema_50",
            "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
            "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_percent",
            "massive_move_recent", "candle_volatility_high", "candle_bodies_stable",
            "macd_bullish_crossover", "macd_bearish_crossover",
            "stoch_oversold", "stoch_overbought",
            "price_near_bb_lower", "price_near_bb_upper",
            "bb_contracting", "bb_expanding", "volume_spike_recent"
        ]}

    latest = df.iloc[-1]

    # Extract values, handling potential NaN
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    ema_20 = float(latest["EMA_20"]) if pd.notna(latest["EMA_20"]) else None
    ema_50 = float(latest["EMA_50"]) if pd.notna(latest["EMA_50"]) else None
    macd = float(latest["MACD"]) if pd.notna(latest["MACD"]) else None
    macd_signal = float(latest["MACD_Signal"]) if pd.notna(latest["MACD_Signal"]) else None
    macd_hist = float(latest["MACD_Hist"]) if pd.notna(latest["MACD_Hist"]) else None
    stoch_k = float(latest["STOCH_K"]) if pd.notna(latest["STOCH_K"]) else None
    stoch_d = float(latest["STOCH_D"]) if pd.notna(latest["STOCH_D"]) else None
    bb_upper = float(latest["BB_Upper"]) if pd.notna(latest["BB_Upper"]) else None
    bb_lower = float(latest["BB_Lower"]) if pd.notna(latest["BB_Lower"]) else None
    bb_width = float(latest["BB_Width"]) if pd.notna(latest["BB_Width"]) else None
    bb_percent = float(latest["BB_Percent"]) if pd.notna(latest["BB_Percent"]) else None
    latest_close = float(latest["close"]) if pd.notna(latest["close"]) else None
    latest_body_size = float(latest["body_size"]) if pd.notna(latest["body_size"]) else None
    latest_candle_range = float(latest["candle_range"]) if pd.notna(latest["candle_range"]) else None
    latest_wick_to_body_ratio = float(latest["wick_to_body_ratio"]) if pd.notna(latest["wick_to_body_ratio"]) else None
    latest_volume = float(latest["volume"]) if pd.notna(latest["volume"]) else None

    # Trend and Momentum
    bullish_trend = bool(ema_20 is not None and ema_50 is not None and ema_20 > ema_50)
    rsi_oversold = bool(rsi is not None and rsi < 30)
    rsi_overbought = bool(rsi is not None and rsi > 70)
    
    macd_bullish_crossover = bool(macd is not None and macd_signal is not None and macd > macd_signal and macd_hist is not None and macd_hist > 0)
    macd_bearish_crossover = bool(macd is not None and macd_signal is not None and macd < macd_signal and macd_hist is not None and macd_hist < 0)

    stoch_oversold = bool(stoch_k is not None and stoch_d is not None and stoch_k < 20 and stoch_d < 20 and stoch_k > stoch_d) # Added K > D for potential rebound
    stoch_overbought = bool(stoch_k is not None and stoch_d is not None and stoch_k > 80 and stoch_d > 80 and stoch_k < stoch_d) # Added K < D for potential reversal

    # Volatility and Price Action
    massive_move_recent = False
    if len(df) >= 4 and latest_close is not None:
        past_close = df['close'].iloc[-4] if len(df) >= 4 else df['close'].iloc[0]
        if pd.notna(past_close) and past_close != 0:
            percent_change = abs((latest_close - past_close) / past_close) * 100
            if percent_change > 2.0: # Define "massive" as > 2% move in 4 hours
                massive_move_recent = True

    candle_volatility_high = False
    candle_bodies_stable = False
    if latest_wick_to_body_ratio is not None and latest_body_size is not None and latest_candle_range is not None and latest_close is not None:
        if latest_wick_to_body_ratio > 1.5 or (latest_candle_range / latest_close) * 100 > 1.0:
            candle_volatility_high = True
        if latest_wick_to_body_ratio < 0.5 and (latest_body_size / latest_close) * 100 > 0.1:
            candle_bodies_stable = True
    if candle_volatility_high: # If high volatility, bodies are NOT stable.
        candle_bodies_stable = False

    # Bollinger Band interpretations
    price_near_bb_lower = bool(latest_close is not None and bb_lower is not None and latest_close <= bb_lower * 1.005) # Within 0.5% of lower band
    price_near_bb_upper = bool(latest_close is not None and bb_upper is not None and latest_close >= bb_upper * 0.995) # Within 0.5% of upper band
    
    bb_contracting = False
    bb_expanding = False
    if len(df) >= 20 and bb_width is not None:
        rolling_bb_width_avg = df['BB_Width'].iloc[-20:].mean() # Average of last 20 widths
        if rolling_bb_width_avg is not None and rolling_bb_width_avg > 0:
            if bb_width < rolling_bb_width_avg * 0.9: # 10% narrower than recent average
                bb_contracting = True
            elif bb_width > rolling_bb_width_avg * 1.1: # 10% wider than recent average
                bb_expanding = True

    # Volume Analysis: Check for significant volume spikes
    volume_spike_recent = False
    if len(df) >= 10 and latest_volume is not None:
        avg_volume_past = df['volume'].iloc[-10:-1].mean() # Average of last 9 volumes
        if avg_volume_past is not None and avg_volume_past > 0:
            if latest_volume > avg_volume_past * 1.5: # Current volume is 50% higher than recent average
                volume_spike_recent = True

    return {
        "bullish_trend": bullish_trend,
        "oversold": rsi_oversold,
        "overbought": rsi_overbought,
        "rsi": rsi,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_mid": latest["BB_Mid"] if pd.notna(latest["BB_Mid"]) else None,
        "bb_width": bb_width,
        "bb_percent": bb_percent,
        "massive_move_recent": massive_move_recent,
        "candle_volatility_high": candle_volatility_high,
        "candle_bodies_stable": candle_bodies_stable,
        "macd_bullish_crossover": macd_bullish_crossover,
        "macd_bearish_crossover": macd_bearish_crossover,
        "stoch_oversold": stoch_oversold,
        "stoch_overbought": stoch_overbought,
        "price_near_bb_lower": price_near_bb_lower,
        "price_near_bb_upper": price_near_bb_upper,
        "bb_contracting": bb_contracting,
        "bb_expanding": bb_expanding,
        "volume_spike_recent": volume_spike_recent
    }

# --- Predictor Functions (consolidated and enhanced) ---

def predict_future_price(df: pd.DataFrame, current_price: float, hours_ahead: float = 1):
    """
    Predicts the future price using Facebook Prophet.
    """
    if df.empty or 'close' not in df.columns or len(df) < 50: # Increased minimum data points for Prophet
        print("Warning: Not enough data in DataFrame for accurate Prophet prediction. Returning current price as projected.")
        return current_price, 0.0, current_price

    prophet_df = df.reset_index()[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df.dropna(subset=['y'], inplace=True)

    if prophet_df.empty or len(prophet_df) < 50:
        print("Warning: 'close' column has insufficient numeric data for Prophet prediction after cleaning. Returning current price as projected.")
        return current_price, 0.0, current_price

    # Initialize and fit Prophet model with adjusted parameters
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False, # Crypto often doesn't have strong weekly seasonality on 1hr
        yearly_seasonality=False,
        changepoint_prior_scale=0.1 # Increased flexibility for trend changes
    )
    try:
        model.fit(prophet_df)
    except Exception as e:
        print(f"Error fitting Prophet model: {e}. Returning current price as projected.")
        return current_price, 0.0, current_price

    future = model.make_future_dataframe(periods=int(hours_ahead), freq='H')
    forecast = model.predict(future)
    projected_price = forecast['yhat'].iloc[-1]

    last_known_price = prophet_df['y'].iloc[-1]
    total_predicted_change = projected_price - last_known_price
    avg_delta = total_predicted_change / hours_ahead if hours_ahead > 0 else 0.0

    projected_price = max(0.0, projected_price)

    return round(projected_price, 2), round(avg_delta, 2), current_price


def recommend_probo_vote_for_target(df: pd.DataFrame, current_price: float, sentiment_score: float, target_price: float, target_time_str: str):
    """
    Recommends a 'YES' or 'NO' vote for a Probo outcome based on projected price and sentiment.
    """
    now_utc = datetime.utcnow()
    target_time_only = datetime.strptime(target_time_str, "%H:%M").time()
    target_datetime_utc = now_utc.replace(hour=target_time_only.hour, minute=target_time_only.minute, second=0, microsecond=0)
    
    if target_datetime_utc < now_utc:
        target_datetime_utc += timedelta(days=1)

    hours_remaining = (target_datetime_utc - now_utc).total_seconds() / 3600
    hours_remaining = max(0.25, round(hours_remaining, 2))

    sentiment = sentiment_score
    projected, delta, current = predict_future_price(df, current_price, hours_remaining)

    if projected >= target_price and sentiment >= -0.1: # Sentiment not strongly negative
        vote = "YES"
    else:
        vote = "NO"

    result = {
        "current_price": current,
        "avg_delta_per_hour": delta,
        "hours_remaining": hours_remaining,
        "projected_price": projected,
        "sentiment": sentiment,
        "target_price": target_price,
        "target_time": target_time_str,
        "vote": vote
    }
    return result

# --- Flask App Endpoints ---

def load_initial_data():
    """Fetches and processes initial market data and sentiment for both BTC and ETH."""
    global btc_market_data_df, btc_current_price, btc_sentiment_score, btc_market_conditions
    global eth_market_data_df, eth_current_price, eth_sentiment_score, eth_market_conditions

    # Helper function to create default market conditions
    def get_default_market_conditions():
        return {k: None for k in [
            "bullish_trend", "oversold", "overbought", "rsi", "ema_20", "ema_50",
            "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
            "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_percent",
            "massive_move_recent", "candle_volatility_high", "candle_bodies_stable",
            "macd_bullish_crossover", "macd_bearish_crossover",
            "stoch_oversold", "stoch_overbought",
            "price_near_bb_lower", "price_near_bb_upper",
            "bb_contracting", "bb_expanding", "volume_spike_recent"
        ]}

    # --- Load BTC Data ---
    try:
        print("Fetching BTC OHLCV data...")
        df_btc = fetch_ohlcv(symbol="BTCUSDT", limit=200) # Increased limit for better indicator calculation
        if df_btc.empty:
            print("Failed to fetch BTC OHLCV data. Setting defaults.")
            btc_market_data_df = pd.DataFrame()
            btc_current_price = 0.0
            btc_sentiment_score = 0.0
            btc_market_conditions = get_default_market_conditions()
        else:
            print("Adding technical indicators for BTC...")
            df_btc = add_technical_indicators(df_btc)
            btc_market_data_df = df_btc

            print("Getting current BTC price...")
            btc_current_price = get_current_price(symbol="BTCUSDT")

            print("Getting Bitcoin sentiment...")
            btc_sentiment_score = get_crypto_sentiment(crypto_name="bitcoin")

            print("Interpreting BTC market conditions...")
            btc_market_conditions = interpret_market_conditions(btc_market_data_df)
            print("BTC Data loaded successfully.")
    except Exception as e:
        print(f"Error loading initial BTC data: {e}")
        btc_market_data_df = pd.DataFrame()
        btc_current_price = 0.0
        btc_sentiment_score = 0.0
        btc_market_conditions = get_default_market_conditions()

    # --- Load ETH Data ---
    try:
        print("Fetching ETH OHLCV data...")
        df_eth = fetch_ohlcv(symbol="ETHUSDT", limit=200) # Increased limit
        if df_eth.empty:
            print("Failed to fetch ETH OHLCV data. Setting defaults.")
            eth_market_data_df = pd.DataFrame()
            eth_current_price = 0.0
            eth_sentiment_score = 0.0
            eth_market_conditions = get_default_market_conditions()
        else:
            print("Adding technical indicators for ETH...")
            df_eth = add_technical_indicators(df_eth)
            eth_market_data_df = df_eth

            print("Getting current ETH price...")
            eth_current_price = get_current_price(symbol="ETHUSDT")

            print("Getting Ethereum sentiment...")
            eth_sentiment_score = get_crypto_sentiment(crypto_name="ethereum")

            print("Interpreting ETH market conditions...")
            eth_market_conditions = interpret_market_conditions(df_eth)
            print("ETH Data loaded successfully.")
    except Exception as e:
        print(f"Error loading initial ETH data: {e}")
        eth_market_data_df = pd.DataFrame()
        eth_current_price = 0.0
        eth_sentiment_score = 0.0
        eth_market_conditions = get_default_market_conditions()

# Load data when the Flask app starts
with app.app_context():
    load_initial_data()

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    """Returns current market data for both BTC and ETH."""
    load_initial_data() # Always refresh data for real-time feel

    response_data = {
        "BTC": {
            "current_price": btc_current_price,
            "sentiment_score": btc_sentiment_score,
            "market_conditions": btc_market_conditions,
            "chart_data": []
        },
        "ETH": {
            "current_price": eth_current_price,
            "sentiment_score": eth_sentiment_score,
            "market_conditions": eth_market_conditions,
            "chart_data": []
        }
    }

    # Convert BTC DataFrame to JSON serializable format
    if btc_market_data_df is not None and not btc_market_data_df.empty:
        chart_data_btc = btc_market_data_df[[
            'open', 'high', 'low', 'close', 'volume', 'EMA_20', 'EMA_50',
            'MACD', 'MACD_Signal', 'MACD_Hist', 'STOCH_K', 'STOCH_D',
            'BB_Upper', 'BB_Lower', 'BB_Mid', 'BB_Width', 'BB_Percent'
        ]].reset_index()
        chart_data_btc = chart_data_btc.replace({np.nan: None})
        response_data["BTC"]["chart_data"] = chart_data_btc.to_dict(orient='records')
        for item in response_data["BTC"]["chart_data"]:
            item['timestamp'] = item['timestamp'].isoformat() if pd.notna(item['timestamp']) else None
    else:
        print("BTC market data not available for chart.")

    # Convert ETH DataFrame to JSON serializable format
    if eth_market_data_df is not None and not eth_market_data_df.empty:
        chart_data_eth = eth_market_data_df[[
            'open', 'high', 'low', 'close', 'volume', 'EMA_20', 'EMA_50',
            'MACD', 'MACD_Signal', 'MACD_Hist', 'STOCH_K', 'STOCH_D',
            'BB_Upper', 'BB_Lower', 'BB_Mid', 'BB_Width', 'BB_Percent'
        ]].reset_index()
        chart_data_eth = chart_data_eth.replace({np.nan: None})
        response_data["ETH"]["chart_data"] = chart_data_eth.to_dict(orient='records')
        for item in response_data["ETH"]["chart_data"]:
            item['timestamp'] = item['timestamp'].isoformat() if pd.notna(item['timestamp']) else None
    else:
        print("ETH market data not available for chart.")

    return jsonify(response_data)

@app.route('/api/predict', methods=['POST'])
def predict_outcome():
    """Performs prediction for the selected cryptocurrency."""
    data = request.get_json()
    target_price = data.get('target_price')
    target_time_str = data.get('target_time') # HH:MM in IST
    currency = data.get('currency', 'BTC').upper() # Default to BTC if not specified

    if not target_price or not target_time_str or currency not in ['BTC', 'ETH']:
        return jsonify({"error": "Missing target_price, target_time, or invalid currency"}), 400

    load_initial_data() # Ensure market data is loaded and fresh

    df_to_use = None
    current_price_to_use = 0.0
    sentiment_to_use = 0.0 # This variable holds the sentiment score from the global store
    market_conditions_to_use = {}
    crypto_name_full = ""

    if currency == 'BTC':
        df_to_use = btc_market_data_df
        current_price_to_use = btc_current_price
        sentiment_to_use = btc_sentiment_score
        market_conditions_to_use = btc_market_conditions
        crypto_name_full = "Bitcoin"
    elif currency == 'ETH':
        df_to_use = eth_market_data_df
        current_price_to_use = eth_current_price
        sentiment_to_use = eth_sentiment_score
        market_conditions_to_use = eth_market_conditions
        crypto_name_full = "Ethereum"

    if df_to_use is None or df_to_use.empty:
        return jsonify({"error": f"Market data not available for {currency} prediction"}), 500

    try:
        ist_timezone = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist_timezone)
        target_time_only = datetime.strptime(target_time_str, "%H:%M").time()
        target_datetime_ist = now_ist.replace(hour=target_time_only.hour, minute=target_time_only.minute, second=0, microsecond=0)

        if target_datetime_ist < now_ist:
            target_datetime_ist += timedelta(days=1)

        target_datetime_utc = target_datetime_ist.astimezone(pytz.utc)
        hours_remaining = (target_datetime_utc - datetime.utcnow().replace(tzinfo=pytz.utc)).total_seconds() / 3600
        hours_remaining = max(0.25, round(hours_remaining, 2))

        result = recommend_probo_vote_for_target(
            df=df_to_use,
            current_price=current_price_to_use,
            sentiment_score=sentiment_to_use, # Pass this sentiment to the prediction function
            target_price=target_price,
            target_time_str=target_datetime_utc.strftime("%H:%M") 
        )
        result['hours_remaining'] = hours_remaining
        result['target_time_ist'] = target_time_str # Keep original IST string for frontend display
        result['currency'] = currency # Add currency to result

        # Define sentiment_for_advisor here, *after* result is populated
        # This sentiment_for_advisor will be the (potentially rounded) sentiment from the prediction logic's result
        sentiment_for_advisor = result['sentiment']

        # Retrieve stoch_k and stoch_d from market_conditions_to_use for confidence advisor calculations
        stoch_k = market_conditions_to_use.get('stoch_k')
        stoch_d = market_conditions_to_use.get('stoch_d')

        # --- Evaluate Trust Conditions (12 signals) ---
        trust_conditions_eval = {
            'time_expiry_lt_2hr': hours_remaining >= 0.25 and hours_remaining < 2, # Must be active and <2hr
            'trending_cleanly': market_conditions_to_use['bullish_trend'] is True and market_conditions_to_use['macd_bearish_crossover'] is False,
            'sentiment_strong': abs(sentiment_for_advisor) > 0.2, # Use the correctly defined sentiment_for_advisor
            'rsi_neutral': market_conditions_to_use['rsi'] is not None and 30 <= market_conditions_to_use['rsi'] <= 70,
            'no_major_news_expected': abs(sentiment_for_advisor) > 0.05, # Proxy for market conviction, not flat sentiment
            'candle_bodies_stable_not_volatile': market_conditions_to_use['candle_bodies_stable'] is True and market_conditions_to_use['candle_volatility_high'] is False,
            'macd_bullish_crossover_confirmed': market_conditions_to_use['macd_bullish_crossover'] is True, # Already implies hist > 0 from definition
            'stoch_oversold_bullish_cross': market_conditions_to_use['stoch_oversold'] is True and (stoch_k is not None and stoch_d is not None and stoch_k > stoch_d),
            'price_near_bb_lower': market_conditions_to_use['price_near_bb_lower'] is True,
            'no_massive_move_recent': market_conditions_to_use['massive_move_recent'] is False,
            'bb_stable': market_conditions_to_use['bb_contracting'] is False and market_conditions_to_use['bb_expanding'] is False,
            'no_volume_spike_recent': market_conditions_to_use['volume_spike_recent'] is False
        }
        trust_signals = sum(1 for k, v in trust_conditions_eval.items() if v)

        # --- Evaluate Caution Conditions (11 flags as per UI structure) ---
        caution_conditions_eval = {
            'target_time_gt_3hr': hours_remaining > 3,
            'btc_massive_move': market_conditions_to_use['massive_move_recent'] is True,
            'rsi_extreme': market_conditions_to_use['overbought'] is True or market_conditions_to_use['oversold'] is True,
            'sentiment_conflicting': abs(sentiment_for_advisor) < 0.05, # Use the correctly defined sentiment_for_advisor
            'big_news_coming_or_volume_spike': abs(sentiment_for_advisor) < 0.05 or market_conditions_to_use['volume_spike_recent'] is True,
            'candle_volatility_high': market_conditions_to_use['candle_volatility_high'] is True,
            'macd_bearish_crossover_confirmed': market_conditions_to_use['macd_bearish_crossover'] is True, # Already implies hist < 0 from definition
            'stoch_overbought_bearish_cross': market_conditions_to_use['stoch_overbought'] is True and (stoch_k is not None and stoch_d is not None and stoch_k < stoch_d),
            'price_near_bb_upper_or_bb_volatile': market_conditions_to_use['price_near_bb_upper'] is True or market_conditions_to_use['bb_contracting'] is True or market_conditions_to_use['bb_expanding'] is True,
            'bb_volatile': market_conditions_to_use['bb_contracting'] is True or market_conditions_to_use['bb_expanding'] is True, # Distinct general volatility flag
            'volume_spike_recent': market_conditions_to_use['volume_spike_recent'] is True # Distinct volume spike flag
        }
        caution_flags = sum(1 for k, v in caution_conditions_eval.items() if v)

        # Determine overall advice
        if trust_signals >= 7 and caution_flags < 4:
            advice_message = "🔐 Confidence: *GO with the vote!* Strong signals align."
        elif caution_flags >= 4:
            advice_message = "🔐 Confidence: *SKIP the trade or WAIT!* High uncertainty detected."
        else:
            advice_message = "🔐 Confidence: *Proceed with caution or wait for clearer signals.* Mixed indicators."

        result['confidence_advisor'] = {
            'trust_signals_count': trust_signals,
            'caution_flags_count': caution_flags,
            'advice_message': advice_message,
            'trust_conditions': trust_conditions_eval,
            'caution_conditions': caution_conditions_eval
        }
        
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": f"Invalid input or time format: {ve}"}), 400
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An internal server error occurred during prediction: {e}"}), 500

# --- NEW: Gemini AI Prediction Endpoint ---
@app.route('/api/gemini_ai_vote', methods=['POST'])
def gemini_ai_vote():
    if not gemini_prediction_model:
        return jsonify({"error": "Gemini prediction model not initialized."}), 500

    data = request.get_json()
    target_price = data.get('target_price')
    target_time_str = data.get('target_time') # HH:MM in IST
    currency = data.get('currency', 'BTC').upper()

    if not target_price or not target_time_str or currency not in ['BTC', 'ETH']:
        return jsonify({"error": "Missing target_price, target_time, or invalid currency"}), 400

    # Ensure market data is loaded and fresh
    load_initial_data() 

    df_to_use = None
    current_price_to_use = 0.0
    sentiment_to_use = 0.0
    market_conditions_to_use = {}
    prophet_projected_price = 0.0
    prophet_avg_delta = 0.0

    if currency == 'BTC':
        df_to_use = btc_market_data_df
        current_price_to_use = btc_current_price
        sentiment_to_use = btc_sentiment_score
        market_conditions_to_use = btc_market_conditions
    elif currency == 'ETH':
        df_to_use = eth_market_data_df
        current_price_to_use = eth_current_price
        sentiment_to_use = eth_sentiment_score
        market_conditions_to_use = eth_market_conditions
    
    if df_to_use is None or df_to_use.empty:
        return jsonify({"error": f"Market data not available for {currency} AI prediction"}), 500

    try:
        ist_timezone = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist_timezone)
        target_time_only = datetime.strptime(target_time_str, "%H:%M").time()
        target_datetime_ist = now_ist.replace(hour=target_time_only.hour, minute=target_time_only.minute, second=0, microsecond=0)

        if target_datetime_ist < now_ist:
            target_datetime_ist += timedelta(days=1)

        target_datetime_utc = target_datetime_ist.astimezone(pytz.utc)
        hours_remaining = (target_datetime_utc - datetime.utcnow().replace(tzinfo=pytz.utc)).total_seconds() / 3600
        hours_remaining = max(0.25, round(hours_remaining, 2))

        # Get Prophet prediction as part of context for Gemini
        prophet_projected_price, prophet_avg_delta, _ = predict_future_price(df_to_use, current_price_to_use, hours_remaining)

        # Prepare prompt for Gemini
        prompt = f"""
        You are an expert cryptocurrency market analyst and a Probo event predictor. Your task is to analyze the provided market data, technical indicators, and sentiment, then make a highly confident recommendation (YES/NO) for a Probo event based on whether the {currency} price will be AT or ABOVE a target price by a specific target time.

        Here is the current market context for {currency}:

        --- Market Data ---
        Current Price: ${current_price_to_use:.2f}
        Time Remaining until Target: {hours_remaining:.2f} hours
        Target Price for Prediction: ${target_price:.2f}
        Target Time (IST): {target_time_str}
        Sentiment Score (from news analysis, -1.0 to 1.0): {sentiment_to_use:.3f}

        --- Technical Indicators (Latest Candle) ---
        RSI: {market_conditions_to_use.get('rsi', 'N/A'):.2f} (30=oversold, 70=overbought)
        EMA 20: {market_conditions_to_use.get('ema_20', 'N/A'):.2f}
        EMA 50: {market_conditions_to_use.get('ema_50', 'N/A'):.2f}
        MACD: {market_conditions_to_use.get('macd', 'N/A'):.2f}
        MACD Signal: {market_conditions_to_use.get('macd_signal', 'N/A'):.2f}
        MACD Histogram: {market_conditions_to_use.get('macd_hist', 'N/A'):.2f} (Positive = bullish momentum, Negative = bearish momentum)
        Stochastic %K: {market_conditions_to_use.get('stoch_k', 'N/A'):.2f} (20=oversold, 80=overbought)
        Stochastic %D: {market_conditions_to_use.get('stoch_d', 'N/A'):.2f}
        Bollinger Bands (Upper/Mid/Lower): {market_conditions_to_use.get('bb_upper', 'N/A'):.2f} / {market_conditions_to_use.get('bb_mid', 'N/A'):.2f} / {market_conditions_to_use.get('bb_lower', 'N/A'):.2f}
        Bollinger Band Width: {market_conditions_to_use.get('bb_width', 'N/A'):.2f}
        Bollinger Band %B: {market_conditions_to_use.get('bb_percent', 'N/A'):.2f} (0=lower band, 0.5=mid band, 1=upper band)

        --- Market Conditions / Interpretations ---
        Bullish Trend (EMA20 > EMA50): {market_conditions_to_use.get('bullish_trend', 'N/A')}
        RSI Oversold (<30): {market_conditions_to_use.get('oversold', 'N/A')}
        RSI Overbought (>70): {market_conditions_to_use.get('overbought', 'N/A')}
        MACD Bullish Crossover (MACD > Signal, Hist > 0): {market_conditions_to_use.get('macd_bullish_crossover', 'N/A')}
        MACD Bearish Crossover (MACD < Signal, Hist < 0): {market_conditions_to_use.get('macd_bearish_crossover', 'N/A')}
        Stochastic Oversold (%K/%D < 20 and K > D): {market_conditions_to_use.get('stoch_oversold', 'N/A')}
        Stochastic Overbought (%K/%D > 80 and K < D): {market_conditions_to_use.get('stoch_overbought', 'N/A')}
        Price Near BB Lower Band: {market_conditions_to_use.get('price_near_bb_lower', 'N/A')}
        Price Near BB Upper Band: {market_conditions_to_use.get('price_near_bb_upper', 'N/A')}
        Bollinger Bands Contracting (getting narrower): {market_conditions_to_use.get('bb_contracting', 'N/A')}
        Bollinger Bands Expanding (getting wider): {market_conditions_to_use.get('bb_expanding', 'N/A')}
        Massive Price Move Recently (past 4 hours): {market_conditions_to_use.get('massive_move_recent', 'N/A')}
        High Candle Volatility (large wicks): {market_conditions_to_use.get('candle_volatility_high', 'N/A')}
        Stable Candle Bodies (small wicks, decent body): {market_conditions_to_use.get('candle_bodies_stable', 'N/A')}
        Recent Volume Spike: {market_conditions_to_use.get('volume_spike_recent', 'N/A')}

        --- Prophet Model Projection ---
        Prophet Projected Price by Target Time: ${prophet_projected_price:.2f}
        Prophet Average Price Change per Hour: ${prophet_avg_delta:.2f}

        Based on all the above information, including the current price, target price, time remaining, sentiment, all technical indicators, and the interpreted market conditions, should one vote YES or NO on the Probo event?

        Your answer MUST be structured as follows:
        VOTE: [YES/NO]
        REASONING: [A concise, yet comprehensive explanation of why you recommend this vote, referencing specific data points and indicators from the provided context. Consider the interplay of trend, momentum, volatility, and sentiment. Also, comment on the Prophet projection and whether your analysis supports or contradicts it.]
        """
        # print("Gemini Prompt:\n", prompt) # For debugging

        response = gemini_prediction_model.generate_content(prompt)
        ai_response_text = response.text.strip()
        # print("Gemini Raw Response:\n", ai_response_text) # For debugging

        ai_vote = "N/A"
        ai_reasoning = "Could not parse AI response."

        if ai_response_text.startswith("VOTE:"):
            lines = ai_response_text.split('\n')
            for line in lines:
                if line.startswith("VOTE:"):
                    ai_vote = line.replace("VOTE:", "").strip()
                elif line.startswith("REASONING:"):
                    ai_reasoning = line.replace("REASONING:", "").strip()
                    # Capture the rest of the lines as well for multi-line reasoning
                    reasoning_lines = lines[lines.index(line):]
                    ai_reasoning = "\n".join([l.replace("REASONING:", "").strip() if l.startswith("REASONING:") else l.strip() for l in reasoning_lines])
                    break # Stop after finding reasoning

        return jsonify({
            "ai_vote": ai_vote,
            "ai_reasoning": ai_reasoning,
            "status": "success"
        })

    except Exception as e:
        print(f"Error during Gemini AI prediction: {e}")
        return jsonify({"error": f"An internal server error occurred during AI prediction: {e}"}), 500

# --- NEW: Gemini AI Chatbot Endpoint ---
@app.route('/api/ai_chat', methods=['POST'])
def ai_chat():
    if not gemini_chat_model:
        return jsonify({"error": "Gemini chat model not initialized."}), 500

    user_message = request.form.get('message')
    files = request.files.getlist('files')

    if not user_message and not files:
        return jsonify({"error": "No message or files provided."}), 400

    content = []
    if user_message:
        content.append(user_message)

    for file in files:
        try:
            filename = file.filename
            file_bytes = file.read()
            mime_type = file.mimetype

            if mime_type.startswith('image/'):
                img = Image.open(io.BytesIO(file_bytes))
                content.append(img)
            elif mime_type == 'text/plain' or 'csv' in mime_type: # Basic text/CSV
                content.append(f"File '{filename}' content:\n{file_bytes.decode('utf-8')}")
            else:
                # Placeholder for other document types
                content.append(f"Received file '{filename}' ({mime_type}). I can only process text and images directly. For complex formats like PDF, DOCX, XLSX, dedicated libraries would be needed to extract meaningful content.")
                print(f"Unsupported file type for direct processing: {mime_type} ({filename})")
                
        except Exception as e:
            print(f"Error processing uploaded file {file.filename}: {e}")
            content.append(f"Error processing file '{file.filename}': {e}")
    
    if not content:
        return jsonify({"error": "No content to send to AI (message or files failed to process)."}), 400

    try:
        response = gemini_chat_model.generate_content(content)
        return jsonify({"ai_response": response.text.strip()})
    except Exception as e:
        print(f"Error during Gemini AI chat: {e}")
        return jsonify({"error": f"An internal server error occurred during AI chat: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)