# telegram_bot.py
import requests
import os

# Your bot token and user ID
# Get BOT_TOKEN from environment variable
# IMPORTANT: Replace "YOUR_FALLBACK_TOKEN_IF_NOT_SET" with a dummy value
# but ensure you set the actual environment variable in your deployment environment.
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_FALLBACK_TOKEN_IF_NOT_SET")
USER_ID = os.environ.get("TELEGRAM_USER_ID", 5368095453) # Or keep your numeric ID if it's constant

def send_telegram_alert(message):
    """
    Sends a Markdown-formatted message as an alert to a specified Telegram user.

    Args:
        message (str): The message content to send.
    """
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_FALLBACK_TOKEN_IF_NOT_SET":
        print("❌ Telegram BOT_TOKEN not configured. Alert not sent. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    if not USER_ID:
        print("❌ Telegram USER_ID not configured. Alert not sent. Please set the TELEGRAM_USER_ID environment variable.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": USER_ID,
        "text": message,
        "parse_mode": "Markdown" # Allows bold, italics, etc.
    }
    try:
        response = requests.post(url, json=payload, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        if response.status_code == 200:
            print("✅ Telegram alert sent successfully.")
        else:
            print(f"❌ Failed to send alert (status code: {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram alert error (network/request issue): {str(e)}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while sending Telegram alert: {str(e)}")

# Test message
if __name__ == "__main__":
    print("--- Testing telegram_bot.py ---")
    # For local testing, you MUST set the environment variables in your terminal
    # before running this script. Example:
    # export TELEGRAM_BOT_TOKEN="YOUR_ACTUAL_BOT_TOKEN_HERE"
    # export TELEGRAM_USER_ID="YOUR_ACTUAL_USER_ID_HERE"
    
    test_message = "🚨 *Test Alert* from BTC Probo Predictor! If you see this, the bot is working."
    send_telegram_alert(test_message)

