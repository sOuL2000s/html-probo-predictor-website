# predictor.py

import datetime
import pandas as pd # Ensure pandas is imported

# Removed redundant imports (fetch_ohlcv, add_technical_indicators, get_current_price, get_bitcoin_sentiment)
# as these functions will now receive the necessary data (df, current_price, sentiment_score)
# directly from the calling backend_app.py

def predict_future_price(df: pd.DataFrame, current_price: float, hours_ahead: float = 1):
    """
    Predicts the future price based on historical price changes.

    Args:
        df (pd.DataFrame): DataFrame containing historical OHLCV data, specifically 'close' prices.
        current_price (float): The current BTC price.
        hours_ahead (float): The number of hours into the future to predict.

    Returns:
        tuple: (projected_price, average_delta_per_hour, current_price)
    """
    if df.empty or 'close' not in df.columns or len(df) < 2:
        # Not enough data to calculate price changes, return current price as projected
        print("Warning: Not enough data in DataFrame for accurate price prediction. Returning current price as projected.")
        return current_price, 0.0, current_price

    # Ensure 'close' column is numeric
    df_numeric_close = pd.to_numeric(df['close'], errors='coerce').dropna()
    if df_numeric_close.empty or len(df_numeric_close) < 2:
        print("Warning: 'close' column has insufficient numeric data for prediction. Returning current price as projected.")
        return current_price, 0.0, current_price

    # Calculate average price movement per hour from the historical data
    price_changes = df_numeric_close.diff().dropna()
    
    avg_delta = price_changes.mean() if not price_changes.empty else 0.0

    projected_price = current_price + (avg_delta * hours_ahead)

    return round(projected_price, 2), round(avg_delta, 2), current_price

def recommend_probo_vote_for_target(df: pd.DataFrame, current_price: float, sentiment_score: float, target_price: float, target_time_str: str):
    """
    Recommends a 'YES' or 'NO' vote for a Probo outcome based on projected price and sentiment.

    Args:
        df (pd.DataFrame): DataFrame containing historical OHLCV data.
        current_price (float): The current BTC price.
        sentiment_score (float): The current Bitcoin sentiment score.
        target_price (float): The target price for the Probo outcome.
        target_time_str (str): The target time in "HH:MM" format (UTC).

    Returns:
        dict: A dictionary containing prediction details and the recommended vote.
    """
    # 1. Parse time and calculate hours remaining
    now_utc = datetime.datetime.utcnow()
    
    # Parse target_time_str (which is expected to be HH:MM in UTC from backend)
    target_time_only = datetime.datetime.strptime(target_time_str, "%H:%M").time()
    
    # Combine today's UTC date with target time
    target_datetime_utc = now_utc.replace(hour=target_time_only.hour, minute=target_time_only.minute, second=0, microsecond=0)
    
    # If target time has already passed today (UTC), assume it's for tomorrow
    if target_datetime_utc < now_utc:
        target_datetime_utc += datetime.timedelta(days=1)

    hours_remaining = (target_datetime_utc - now_utc).total_seconds() / 3600
    hours_remaining = max(0.25, round(hours_remaining, 2))  # Minimum 15 min window (0.25 hours)

    # 2. Use passed sentiment
    sentiment = sentiment_score

    # 3. Predict price using the provided df and current_price
    projected, delta, current = predict_future_price(df, current_price, hours_remaining)

    # 4. Decision logic
    # Vote 'YES' if projected price meets or exceeds target AND sentiment is not strongly negative
    if projected >= target_price and sentiment >= -0.1:
        vote = "YES"
    else:
        vote = "NO"

    # 5. Return analysis
    result = {
        "current_price": current,
        "avg_delta_per_hour": delta,
        "hours_remaining": hours_remaining,
        "projected_price": projected,
        "sentiment": sentiment,
        "target_price": target_price,
        "target_time": target_time_str, # Keep the original HH:MM string for display
        "vote": vote
    }

    return result

if __name__ == "__main__":
    # This block is for local testing of predictor.py's functions
    # In a real application, these functions are called by backend_app.py
    print("--- Testing predictor.py (requires mock data) ---")
    
    # Example of how to test locally (uncomment and run if needed)
    # from btc_data import fetch_ohlcv, add_technical_indicators, get_current_price
    # from sentiment import get_bitcoin_sentiment

    # print("Fetching mock data for local predictor test...")
    # df_test = fetch_ohlcv(limit=50) # Fetch some data
    # df_test = add_technical_indicators(df_test) # Add indicators
    # current_price_test = get_current_price() # Get current price
    # sentiment_test = get_bitcoin_sentiment() # Get sentiment

    # if not df_test.empty and current_price_test > 0:
    #     target_price_test = current_price_test + 500 # Example target
    #     target_time_test = (datetime.datetime.utcnow() + datetime.timedelta(hours=2)).strftime("%H:%M") # 2 hours from now UTC

    #     question = recommend_probo_vote_for_target(
    #         df=df_test,
    #         current_price=current_price_test,
    #         sentiment_score=sentiment_test,
    #         target_price=target_price_test,
    #         target_time_str=target_time_test
    #     )
        
    #     print("\n🧠 Prediction Summary (Local Test):")
    #     for k, v in question.items():
    #         print(f"{k.replace('_', ' ').title()}: {v}")
    # else:
    #     print("Could not fetch sufficient data for local predictor test.")

