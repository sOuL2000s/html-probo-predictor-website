# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime, timedelta # Correct import: datetime now refers to the datetime class
import pandas as pd
import pytz
import numpy as np
import requests
import feedparser
from textblob import TextBlob
import urllib.parse
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

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
# Telegram Bot Token and User ID (from environment variables)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_FALLBACK_TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.environ.get("TELEGRAM_USER_ID", 5368095453) # Replace with your actual numeric user ID

# --- Crypto Data Functions (formerly btc_data.py) ---

def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=100):
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
        
        # Convert relevant columns to numeric, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with any NaN values that resulted from coercion in core OHLCV
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
    Adds RSI, EMA 20, EMA 50, and candle volatility metrics to the DataFrame.
    """
    if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
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

# --- Sentiment Functions (formerly sentiment.py) ---

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
    return fetch_news_sentiment(f"{crypto_name} OR {crypto_name[:3].upper()}") # e.g., "bitcoin OR BTC"

# --- Probo Strategy Functions (formerly probo_strategy.py) ---

def interpret_market_conditions(df: pd.DataFrame):
    """
    Interprets market conditions based on technical indicators and volatility metrics.
    Automates checks for "massive move recent", "candle volatility high", and "candle bodies stable".
    """
    required_cols = ['RSI', 'EMA_20', 'EMA_50', 'body_size', 'candle_range', 'wick_to_body_ratio', 'close']
    if df.empty or not all(col in df.columns for col in required_cols):
        print("Warning: DataFrame is empty or missing required columns for market interpretation. Returning default conditions.")
        return {
            "bullish_trend": False, "oversold": False, "overbought": False,
            "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
            "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False
        }

    latest = df.iloc[-1]

    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    ema_20 = float(latest["EMA_20"]) if pd.notna(latest["EMA_20"]) else None
    ema_50 = float(latest["EMA_50"]) if pd.notna(latest["EMA_50"]) else None
    latest_close = float(latest["close"]) if pd.notna(latest["close"]) else None
    latest_body_size = float(latest["body_size"]) if pd.notna(latest["body_size"]) else None
    latest_candle_range = float(latest["candle_range"]) if pd.notna(latest["candle_range"]) else None
    latest_wick_to_body_ratio = float(latest["wick_to_body_ratio"]) if pd.notna(latest["wick_to_body_ratio"]) else None

    bullish_trend = bool(ema_20 is not None and ema_50 is not None and ema_20 > ema_50)
    oversold = bool(rsi is not None and rsi < 30)
    overbought = bool(rsi is not None and rsi > 70)

    # Automated "massive move recent"
    massive_move_recent = False
    if len(df) >= 4 and latest_close is not None:
        past_close = df['close'].iloc[-4] if len(df) >= 4 else df['close'].iloc[0]
        if pd.notna(past_close) and past_close != 0:
            percent_change = abs((latest_close - past_close) / past_close) * 100
            if percent_change > 2.0: # Example: >2% move in last 4 hours
                massive_move_recent = True

    # Automated "candle volatility high" and "candle bodies stable"
    candle_volatility_high = False
    candle_bodies_stable = False

    if latest_wick_to_body_ratio is not None and latest_body_size is not None and latest_candle_range is not None and latest_close is not None:
        # High volatility: high wick-to-body ratio OR large total candle range relative to price
        if latest_wick_to_body_ratio > 1.5 or (latest_candle_range / latest_close) * 100 > 1.0:
            candle_volatility_high = True
        
        # Stable bodies: small wick-to-body ratio AND decent body size (not a doji)
        if latest_wick_to_body_ratio < 0.5 and (latest_body_size / latest_close) * 100 > 0.1:
            candle_bodies_stable = True
    
    if candle_volatility_high: # If high volatility, bodies are not stable
        candle_bodies_stable = False

    return {
        "bullish_trend": bullish_trend,
        "oversold": oversold,
        "overbought": overbought,
        "rsi": rsi,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "massive_move_recent": massive_move_recent,
        "candle_volatility_high": candle_volatility_high,
        "candle_bodies_stable": candle_bodies_stable
    }

# --- Predictor Functions (formerly predictor.py) ---

def predict_future_price(df: pd.DataFrame, current_price: float, hours_ahead: float = 1):
    """
    Predicts the future price based on historical price changes.
    """
    if df.empty or 'close' not in df.columns or len(df) < 2:
        print("Warning: Not enough data in DataFrame for accurate price prediction. Returning current price as projected.")
        return current_price, 0.0, current_price

    df_numeric_close = pd.to_numeric(df['close'], errors='coerce').dropna()
    if df_numeric_close.empty or len(df_numeric_close) < 2:
        print("Warning: 'close' column has insufficient numeric data for prediction. Returning current price as projected.")
        return current_price, 0.0, current_price

    price_changes = df_numeric_close.diff().dropna()
    avg_delta = price_changes.mean() if not price_changes.empty else 0.0

    projected_price = current_price + (avg_delta * hours_ahead)

    return round(projected_price, 2), round(avg_delta, 2), current_price

def recommend_probo_vote_for_target(df: pd.DataFrame, current_price: float, sentiment_score: float, target_price: float, target_time_str: str):
    """
    Recommends a 'YES' or 'NO' vote for a Probo outcome based on projected price and sentiment.
    """
    now_utc = datetime.utcnow() # Corrected: Use datetime.utcnow()
    target_time_only = datetime.strptime(target_time_str, "%H:%M").time() # Corrected: Use datetime.strptime()
    target_datetime_utc = now_utc.replace(hour=target_time_only.hour, minute=target_time_only.minute, second=0, microsecond=0)
    
    if target_datetime_utc < now_utc:
        target_datetime_utc += timedelta(days=1)

    hours_remaining = (target_datetime_utc - now_utc).total_seconds() / 3600
    hours_remaining = max(0.25, round(hours_remaining, 2))

    sentiment = sentiment_score
    projected, delta, current = predict_future_price(df, current_price, hours_remaining)

    if projected >= target_price and sentiment >= -0.1:
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

# --- Telegram Bot Functions (formerly telegram_bot.py) ---

def send_telegram_alert(message):
    """
    Sends a Markdown-formatted message as an alert to a specified Telegram user.
    """
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_FALLBACK_TELEGRAM_BOT_TOKEN":
        print("❌ Telegram BOT_TOKEN not configured. Alert not sent. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    if not TELEGRAM_USER_ID:
        print("❌ Telegram USER_ID not configured. Alert not sent. Please set the TELEGRAM_USER_ID environment variable.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_USER_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            print("✅ Telegram alert sent successfully.")
        else:
            print(f"❌ Failed to send alert (status code: {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram alert error (network/request issue): {str(e)}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while sending Telegram alert: {str(e)}")

# --- Flask App Endpoints ---

def load_initial_data():
    """Fetches and processes initial market data and sentiment for both BTC and ETH."""
    global btc_market_data_df, btc_current_price, btc_sentiment_score, btc_market_conditions
    global eth_market_data_df, eth_current_price, eth_sentiment_score, eth_market_conditions

    # --- Load BTC Data ---
    try:
        print("Fetching BTC OHLCV data...")
        df_btc = fetch_ohlcv(symbol="BTCUSDT")
        if df_btc.empty:
            print("Failed to fetch BTC OHLCV data. Setting defaults.")
            btc_market_data_df = pd.DataFrame()
            btc_current_price = 0.0
            btc_sentiment_score = 0.0
            btc_market_conditions = {
                "bullish_trend": False, "oversold": False, "overbought": False,
                "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
                "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False
            }
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
        btc_market_conditions = {
            "bullish_trend": False, "oversold": False, "overbought": False,
            "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
            "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False
        }

    # --- Load ETH Data ---
    try:
        print("Fetching ETH OHLCV data...")
        df_eth = fetch_ohlcv(symbol="ETHUSDT")
        if df_eth.empty:
            print("Failed to fetch ETH OHLCV data. Setting defaults.")
            eth_market_data_df = pd.DataFrame()
            eth_current_price = 0.0
            eth_sentiment_score = 0.0
            eth_market_conditions = {
                "bullish_trend": False, "oversold": False, "overbought": False,
                "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
                "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False
            }
        else:
            print("Adding technical indicators for ETH...")
            df_eth = add_technical_indicators(df_eth)
            eth_market_data_df = df_eth

            print("Getting current ETH price...")
            eth_current_price = get_current_price(symbol="ETHUSDT")

            print("Getting Ethereum sentiment...")
            eth_sentiment_score = get_crypto_sentiment(crypto_name="ethereum")

            print("Interpreting ETH market conditions...")
            eth_market_conditions = interpret_market_conditions(eth_market_data_df)
            print("ETH Data loaded successfully.")
    except Exception as e:
        print(f"Error loading initial ETH data: {e}")
        eth_market_data_df = pd.DataFrame()
        eth_current_price = 0.0
        eth_sentiment_score = 0.0
        eth_market_conditions = {
            "bullish_trend": False, "oversold": False, "overbought": False,
            "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
            "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False
        }

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
        chart_data_btc = btc_market_data_df[['open', 'high', 'low', 'close', 'EMA_20', 'EMA_50']].reset_index()
        chart_data_btc = chart_data_btc.replace({np.nan: None})
        response_data["BTC"]["chart_data"] = chart_data_btc.to_dict(orient='records')
        for item in response_data["BTC"]["chart_data"]:
            item['timestamp'] = item['timestamp'].isoformat() if pd.notna(item['timestamp']) else None
    else:
        print("BTC market data not available for chart.")

    # Convert ETH DataFrame to JSON serializable format
    if eth_market_data_df is not None and not eth_market_data_df.empty:
        chart_data_eth = eth_market_data_df[['open', 'high', 'low', 'close', 'EMA_20', 'EMA_50']].reset_index()
        chart_data_eth = chart_data_eth.replace({np.nan: None})
        response_data["ETH"]["chart_data"] = chart_data_eth.to_dict(orient='records')
        for item in response_data["ETH"]["chart_data"]:
            item['timestamp'] = item['timestamp'].isoformat() if pd.notna(item['timestamp']) else None
    else:
        print("ETH market data not available for chart.")

    return jsonify(response_data)

@app.route('/api/predict', methods=['POST'])
def predict_outcome():
    """Performs prediction for the selected cryptocurrency and sends a Telegram alert."""
    data = request.get_json()
    target_price = data.get('target_price')
    target_time_str = data.get('target_time') # HH:MM in IST
    currency = data.get('currency', 'BTC').upper() # Default to BTC if not specified

    if not target_price or not target_time_str or currency not in ['BTC', 'ETH']:
        return jsonify({"error": "Missing target_price, target_time, or invalid currency"}), 400

    load_initial_data() # Ensure market data is loaded and fresh

    df_to_use = None
    current_price_to_use = 0.0
    sentiment_to_use = 0.0
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
        now_ist = datetime.now(ist_timezone) # Corrected: Use datetime.now()
        target_time_only = datetime.strptime(target_time_str, "%H:%M").time() # Corrected: Use datetime.strptime()
        target_datetime_ist = now_ist.replace(hour=target_time_only.hour, minute=target_time_only.minute, second=0, microsecond=0)

        if target_datetime_ist < now_ist:
            target_datetime_ist += timedelta(days=1)

        target_datetime_utc = target_datetime_ist.astimezone(pytz.utc)
        hours_remaining = (target_datetime_utc - datetime.utcnow().replace(tzinfo=pytz.utc)).total_seconds() / 3600 # Corrected: Use datetime.utcnow()
        hours_remaining = max(0.25, round(hours_remaining, 2))

        result = recommend_probo_vote_for_target(
            df=df_to_use,
            current_price=current_price_to_use,
            sentiment_score=sentiment_to_use,
            target_price=target_price,
            target_time_str=target_datetime_utc.strftime("%H:%M")
        )
        result['hours_remaining'] = hours_remaining
        result['target_time'] = target_time_str # Keep original IST string for frontend display
        result['currency'] = currency # Add currency to result

        trust_signals = 0
        caution_flags = 0

        hours_remaining_float = result['hours_remaining']
        sentiment_score = result['sentiment']

        # Evaluate Trust Conditions
        if hours_remaining_float < 2: trust_signals += 1
        if market_conditions_to_use['bullish_trend'] or (market_conditions_to_use['ema_20'] is not None and market_conditions_to_use['ema_50'] is not None and market_conditions_to_use['ema_20'] < market_conditions_to_use['ema_50'] and current_price_to_use < market_conditions_to_use['ema_20']): trust_signals += 1
        if abs(sentiment_score) > 0.2: trust_signals += 1
        if market_conditions_to_use['rsi'] is not None and 30 <= market_conditions_to_use['rsi'] <= 70: trust_signals += 1
        # Automated "No major news expected" (simplified: assume no news if sentiment is not conflicting)
        if abs(sentiment_score) > 0.05: trust_signals += 1
        # Automated "Candle bodies are stable"
        if market_conditions_to_use['candle_bodies_stable']: trust_signals += 1


        # Evaluate Caution Conditions
        if hours_remaining_float > 3: caution_flags += 1
        if market_conditions_to_use['overbought'] or market_conditions_to_use['oversold']: caution_flags += 1
        if abs(sentiment_score) < 0.05: caution_flags += 1
        # Automated "BTC just made a massive move"
        if market_conditions_to_use['massive_move_recent']: caution_flags += 1
        # Automated "Big news coming" (simplified: if sentiment is conflicting, assume big news might be coming)
        if abs(sentiment_score) < 0.05: caution_flags += 1
        # Automated "Candle volatility is high"
        if market_conditions_to_use['candle_volatility_high']: caution_flags += 1


        # Update total counts for advice message (now 6 for both trust and caution)
        if trust_signals >= 3 and caution_flags < 2:
            advice_message = "🔐 Confidence: *GO with the vote!* (Trust: {}/6, Caution: {}/6)".format(trust_signals, caution_flags)
        elif caution_flags >= 2:
            advice_message = "🔐 Confidence: *SKIP the trade or WAIT!* (Trust: {}/6, Caution: {}/6)".format(trust_signals, caution_flags)
        else:
            advice_message = "🔐 Confidence: *Proceed with caution or wait for clearer signals.* (Trust: {}/6, Caution: {}/6)".format(trust_signals, caution_flags)

        result['confidence_advisor'] = {
            'trust_signals_count': trust_signals,
            'caution_flags_count': caution_flags,
            'advice_message': advice_message,
            'trust_conditions': {
                'time_expiry_lt_2hr': hours_remaining_float < 2,
                'trending_cleanly': bool(market_conditions_to_use['bullish_trend'] or (market_conditions_to_use['ema_20'] is not None and market_conditions_to_use['ema_50'] is not None and market_conditions_to_use['ema_20'] < market_conditions_to_use['ema_50'] and current_price_to_use < market_conditions_to_use['ema_20'])),
                'sentiment_strong': abs(sentiment_score) > 0.2,
                'rsi_neutral': bool(market_conditions_to_use['rsi'] is not None and 30 <= market_conditions_to_use['rsi'] <= 70),
                'no_major_news_expected': abs(sentiment_score) > 0.05, # Automated based on sentiment
                'candle_bodies_stable': market_conditions_to_use['candle_bodies_stable'] # Automated
            },
            'caution_conditions': {
                'target_time_gt_3hr': hours_remaining_float > 3,
                'rsi_extreme': bool(market_conditions_to_use['overbought'] or market_conditions_to_use['oversold']),
                'sentiment_conflicting': abs(sentiment_score) < 0.05,
                'btc_massive_move': market_conditions_to_use['massive_move_recent'], # Automated
                'big_news_coming': abs(sentiment_score) < 0.05, # Automated based on sentiment
                'candle_volatility_high': market_conditions_to_use['candle_volatility_high'] # Automated
            }
        }

        alert_message = (
            f"📣 *{currency} Probo Vote Recommendation*\n"
            f"🕒 Target Time (IST): *{target_time_str}*\n"
            f"🎯 Target Price: *${result['target_price']}*\n"
            f"💰 Current: *${result['current_price']:.2f}*\n"
            f"📈 Projected: *${result['projected_price']:.2f}*\n"
            f"💬 Sentiment: *{result['sentiment']:.2f}*\n"
            f"✅ Vote: *{result['vote']}*\n"
            f"\n--- Confidence Advisor ---\n"
            f"{advice_message}"
        )
        send_telegram_alert(alert_message)

        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": f"Invalid input or time format: {ve}"}), 400
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An internal server error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    # For local development, set the host and port
    app.run(host='0.0.0.0', port=5000, debug=True)
