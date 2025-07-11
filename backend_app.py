# backend_app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import pandas as pd
import pytz
import numpy as np # Import numpy to handle NaN values

# Import your existing modules
from btc_data import fetch_ohlcv, add_technical_indicators, get_current_price
from sentiment import get_bitcoin_sentiment
from probo_strategy import interpret_market_conditions
from predictor import recommend_probo_vote_for_target
from telegram_bot import send_telegram_alert

app = Flask(__name__)
# Enable CORS for your frontend to access the API
CORS(app)

# Global data storage (for simplicity, in a real app consider a database or caching)
market_data_df = None
current_btc_price = 0.0
bitcoin_sentiment_score = 0.0
market_conditions = {}

def load_initial_data():
    """Fetches and processes initial market data and sentiment."""
    global market_data_df, current_btc_price, bitcoin_sentiment_score, market_conditions
    try:
        print("Fetching OHLCV data...")
        df = fetch_ohlcv()
        if df.empty:
            print("Failed to fetch OHLCV data or data is empty. Setting defaults.")
            market_data_df = pd.DataFrame()
            current_btc_price = 0.0
            bitcoin_sentiment_score = 0.0
            market_conditions = {
                "bullish_trend": False, "oversold": False, "overbought": False,
                "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
                "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False # New defaults
            }
            return
        
        print("Adding technical indicators...")
        df = add_technical_indicators(df)
        market_data_df = df

        print("Getting current price...")
        current_btc_price = get_current_price()

        print("Getting Bitcoin sentiment...")
        bitcoin_sentiment_score = get_bitcoin_sentiment()

        print("Interpreting market conditions...")
        market_conditions = interpret_market_conditions(market_data_df)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading initial data: {e}")
        market_data_df = pd.DataFrame()
        current_btc_price = 0.0
        bitcoin_sentiment_score = 0.0
        market_conditions = {
            "bullish_trend": False, "oversold": False, "overbought": False,
            "rsi": 50.0, "ema_20": 0.0, "ema_50": 0.0,
            "massive_move_recent": False, "candle_volatility_high": False, "candle_bodies_stable": False # New defaults
        }

# Load data when the Flask app starts
with app.app_context():
    load_initial_data()

# API Endpoint to get current market data
@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    global market_data_df, current_btc_price, bitcoin_sentiment_score, market_conditions
    load_initial_data() # Always refresh data for real-time feel

    if market_data_df is None or market_data_df.empty:
        return jsonify({"error": "Market data not available or failed to load"}), 500

    # Convert DataFrame to JSON serializable format
    # Fill NaN values with None before converting to dictionary
    chart_data_df = market_data_df[['open', 'high', 'low', 'close', 'EMA_20', 'EMA_50']].reset_index()
    chart_data_df = chart_data_df.replace({np.nan: None}) # Replace NaN with None for JSON serialization
    chart_data = chart_data_df.to_dict(orient='records')

    for item in chart_data:
        # Ensure timestamp is converted to string, handling potential NaT (Not a Time)
        item['timestamp'] = item['timestamp'].isoformat() if pd.notna(item['timestamp']) else None

    response_data = {
        "current_price": current_btc_price,
        "sentiment_score": bitcoin_sentiment_score,
        "market_conditions": market_conditions,
        "chart_data": chart_data
    }
    return jsonify(response_data)

# API Endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def predict_outcome():
    data = request.get_json()
    target_price = data.get('target_price')
    target_time_str = data.get('target_time') # HH:MM in IST

    if not target_price or not target_time_str:
        return jsonify({"error": "Missing target_price or target_time"}), 400

    load_initial_data() # Ensure market data is loaded and fresh

    if market_data_df is None or market_data_df.empty:
        return jsonify({"error": "Market data not available for prediction"}), 500

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
            df=market_data_df,
            current_price=current_btc_price,
            sentiment_score=bitcoin_sentiment_score,
            target_price=target_price,
            target_time_str=target_datetime_utc.strftime("%H:%M")
        )
        result['hours_remaining'] = hours_remaining
        result['target_time'] = target_time_str # Keep original IST string for frontend display

        trust_signals = 0
        caution_flags = 0

        hours_remaining_float = result['hours_remaining']
        sentiment_score = result['sentiment']

        # Evaluate Trust Conditions (now including automated manual checks)
        if hours_remaining_float < 2: trust_signals += 1
        if market_conditions['bullish_trend'] or (market_conditions['ema_20'] is not None and market_conditions['ema_50'] is not None and market_conditions['ema_20'] < market_conditions['ema_50'] and current_btc_price < market_conditions['ema_20']): trust_signals += 1
        if abs(sentiment_score) > 0.2: trust_signals += 1
        if market_conditions['rsi'] is not None and 30 <= market_conditions['rsi'] <= 70: trust_signals += 1
        # Automated "No major news expected" (simplified: assume no news if sentiment is not conflicting)
        if abs(sentiment_score) > 0.05: trust_signals += 1 # If sentiment is not near zero, assume no conflicting major news
        # Automated "Candle bodies are stable"
        if market_conditions['candle_bodies_stable']: trust_signals += 1


        # Evaluate Caution Conditions (now including automated manual checks)
        if hours_remaining_float > 3: caution_flags += 1
        if market_conditions['overbought'] or market_conditions['oversold']: caution_flags += 1
        if abs(sentiment_score) < 0.05: caution_flags += 1
        # Automated "BTC just made a massive move"
        if market_conditions['massive_move_recent']: caution_flags += 1
        # Automated "Big news coming" (simplified: if sentiment is conflicting, assume big news might be coming)
        if abs(sentiment_score) < 0.05: caution_flags += 1 # If sentiment is near zero, implies indecision/potential news
        # Automated "Candle volatility is high"
        if market_conditions['candle_volatility_high']: caution_flags += 1


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
                'trending_cleanly': bool(market_conditions['bullish_trend'] or (market_conditions['ema_20'] is not None and market_conditions['ema_50'] is not None and market_conditions['ema_20'] < market_conditions['ema_50'] and current_btc_price < market_conditions['ema_20'])),
                'sentiment_strong': abs(sentiment_score) > 0.2,
                'rsi_neutral': bool(market_conditions['rsi'] is not None and 30 <= market_conditions['rsi'] <= 70),
                'no_major_news_expected': abs(sentiment_score) > 0.05, # Automated
                'candle_bodies_stable': market_conditions['candle_bodies_stable'] # Automated
            },
            'caution_conditions': {
                'target_time_gt_3hr': hours_remaining_float > 3,
                'rsi_extreme': bool(market_conditions['overbought'] or market_conditions['oversold']),
                'sentiment_conflicting': abs(sentiment_score) < 0.05,
                'btc_massive_move': market_conditions['massive_move_recent'], # Automated
                'big_news_coming': abs(sentiment_score) < 0.05, # Automated
                'candle_volatility_high': market_conditions['candle_volatility_high'] # Automated
            }
        }

        alert_message = (
            f"📣 *BTC Probo Vote Recommendation*\n"
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
    app.run(host='0.0.0.0', port=5000, debug=True)
