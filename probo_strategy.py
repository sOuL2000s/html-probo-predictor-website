# probo_strategy.py

import pandas as pd
import numpy as np # Import numpy to check for NaN

def interpret_market_conditions(df: pd.DataFrame):
    """
    Interprets market conditions based on technical indicators (RSI, EMA)
    and new volatility/price movement metrics.

    Args:
        df (pandas.DataFrame): DataFrame with 'RSI', 'EMA_20', 'EMA_50',
                               'body_size', 'candle_range', 'wick_to_body_ratio' columns.

    Returns:
        dict: A dictionary containing interpreted market conditions.
    """
    required_cols = ['RSI', 'EMA_20', 'EMA_50', 'body_size', 'candle_range', 'wick_to_body_ratio', 'close']
    if df.empty or not all(col in df.columns for col in required_cols):
        print("Warning: DataFrame is empty or missing required columns for market interpretation. Returning default conditions.")
        return {
            "bullish_trend": False,
            "oversold": False,
            "overbought": False,
            "rsi": 50.0,
            "ema_20": 0.0,
            "ema_50": 0.0,
            "massive_move_recent": False, # New default
            "candle_volatility_high": False, # New default
            "candle_bodies_stable": False # New default
        }

    # Get the latest values
    latest = df.iloc[-1]

    # Convert numpy types to standard Python types and handle NaN
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    ema_20 = float(latest["EMA_20"]) if pd.notna(latest["EMA_20"]) else None
    ema_50 = float(latest["EMA_50"]) if pd.notna(latest["EMA_50"]) else None
    
    # New metrics
    latest_close = float(latest["close"]) if pd.notna(latest["close"]) else None
    latest_body_size = float(latest["body_size"]) if pd.notna(latest["body_size"]) else None
    latest_candle_range = float(latest["candle_range"]) if pd.notna(latest["candle_range"]) else None
    latest_wick_to_body_ratio = float(latest["wick_to_body_ratio"]) if pd.notna(latest["wick_to_body_ratio"]) else None

    # Trend signal: EMA20 > EMA50 generally indicates an uptrend
    bullish_trend = bool(ema_20 is not None and ema_50 is not None and ema_20 > ema_50)
    
    # RSI zones
    oversold = bool(rsi is not None and rsi < 30)
    overbought = bool(rsi is not None and rsi > 70)

    # --- Automated Analysis for previously manual checks ---

    # 1. "BTC just made a massive move"
    # Check for a significant percentage change over the last few candles (e.g., last 4 hours)
    massive_move_recent = False
    if len(df) >= 4 and latest_close is not None:
        # Calculate percentage change over the last 4 hours (or fewer if less data)
        past_close = df['close'].iloc[-4] if len(df) >= 4 else df['close'].iloc[0]
        if pd.notna(past_close) and past_close != 0:
            percent_change = abs((latest_close - past_close) / past_close) * 100
            # Define "massive" as, for example, > 2% move in 4 hours
            if percent_change > 2.0: # Threshold for a "massive move"
                massive_move_recent = True

    # 2. "Candle volatility is high (huge wicks)" / "Candle bodies are stable (not huge wicks)"
    candle_volatility_high = False
    candle_bodies_stable = False

    if latest_wick_to_body_ratio is not None and latest_body_size is not None and latest_candle_range is not None:
        # High volatility: high wick-to-body ratio OR large total candle range relative to price
        # Thresholds are examples and might need tuning
        if latest_wick_to_body_ratio > 1.5 or (latest_candle_range / latest_close) * 100 > 1.0: # Example: wicks 1.5x body, or 1% range
            candle_volatility_high = True
        
        # Stable bodies: small wick-to-body ratio AND decent body size (not a doji)
        # Assuming 'stable' means not too small (doji) and not dominated by wicks
        if latest_wick_to_body_ratio < 0.5 and (latest_body_size / latest_close) * 100 > 0.1: # Example: wicks < 0.5x body, body > 0.1% of price
            candle_bodies_stable = True
    
    # If volatility is high, then bodies are NOT stable. These are often mutually exclusive.
    # Prioritize volatility if detected.
    if candle_volatility_high:
        candle_bodies_stable = False # If high volatility, bodies are not stable

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

# The recommend_probo_vote function below is likely for standalone testing
# or was part of a previous Streamlit app. It is not directly called by the Flask backend
# in the current design, as the prediction logic is handled by backend_app.py calling predictor.py.
def recommend_probo_vote():
    """
    (Deprecated for Flask backend usage)
    Recommends a Probo vote based on market conditions and sentiment.
    This function is more suitable for a standalone script or a Streamlit app.
    For the Flask backend, individual components (fetch_ohlcv, add_technical_indicators,
    get_current_price, get_bitcoin_sentiment, interpret_market_conditions) are called
    and their results are used to build the prediction.
    """
    from btc_data import fetch_ohlcv, add_technical_indicators, get_current_price
    from sentiment import get_bitcoin_sentiment

    print("[+] Fetching market data...")
    df = fetch_ohlcv()
    if df.empty:
        print("Failed to fetch OHLCV data. Cannot recommend vote.")
        return "N/A"

    df = add_technical_indicators(df)
    market = interpret_market_conditions(df)
    price = get_current_price()

    print("[+] Analyzing sentiment...")
    sentiment_score = get_bitcoin_sentiment()

    print("\n📊 BTC Market Snapshot")
    print(f"Price: ${price:.2f}")
    # Handle None values for display
    rsi_display = f"{market['rsi']:.2f}" if market['rsi'] is not None else "N/A"
    ema20_display = f"{market['ema_20']:.2f}" if market['ema_20'] is not None else "N/A"
    ema50_display = f"{market['ema_50']:.2f}" if market['ema_50'] is not None else "N/A"
    print(f"RSI: {rsi_display} | EMA20: {ema20_display} | EMA50: {ema50_display}")
    sentiment_status = 'Bullish' if sentiment_score > 0 else 'Bearish' if sentiment_score < 0 else 'Neutral'
    print(f"Sentiment Score: {sentiment_score:.3f} ({sentiment_status})")

    # Decision logic (simplified for this example)
    vote = "NO"
    if market["bullish_trend"] and sentiment_score > 0.1:
        vote = "YES"
    elif market["oversold"] and sentiment_score > -0.05:
        vote = "YES"
    elif market["overbought"] and sentiment_score < -0.1:
        vote = "NO"

    print(f"\n🧠 Probo Recommendation: ✅ Vote {vote}")
    return vote

if __name__ == "__main__":
    print("--- Running standalone Probo Strategy Recommendation ---")
    recommend_probo_vote()
