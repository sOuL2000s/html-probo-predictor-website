# sentiment.py

import feedparser
from textblob import TextBlob
import urllib.parse
# Removed: import streamlit as st

def fetch_news_sentiment(query="bitcoin", max_items=20):
    """
    Fetches news headlines for a given query from Google News RSS and calculates
    an average sentiment polarity using TextBlob.

    Args:
        query (str): The search query (e.g., "bitcoin").
        max_items (int): Maximum number of news items to process.

    Returns:
        float: Average sentiment polarity (between -1.0 and 1.0).
               Returns 0.0 if no headlines are found or an error occurs.
    """
    encoded_query = urllib.parse.quote(query)  # URL encode the query
    url = f"https://news.google.com/rss/search?q={encoded_query}"
    
    try:
        feed = feedparser.parse(url)
        headlines = [entry.title for entry in feed.entries[:max_items]]

        if not headlines:
            print(f"No news headlines found for query: '{query}'")
            return 0.0  # Neutral if no news

        sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
        return round(sum(sentiments) / len(sentiments), 3)
    except Exception as e:
        print(f"Error fetching or analyzing news sentiment for '{query}': {e}")
        return 0.0 # Return neutral sentiment on error

# Removed: @st.cache_data(ttl=600) decorator as it's Streamlit-specific
def get_bitcoin_sentiment():
    """
    Gets the sentiment score specifically for Bitcoin.
    """
    return fetch_news_sentiment("bitcoin OR btc")

if __name__ == "__main__":
    print("--- Testing sentiment.py ---")
    sentiment = get_bitcoin_sentiment()
    print(f"Bitcoin Sentiment Score: {sentiment}")
    if sentiment > 0:
        print("Market sentiment is positive.")
    elif sentiment < 0:
        print("Market sentiment is negative.")
    else:
        print("Market sentiment is neutral or could not be determined.")

