import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
# News API Key
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    raise ValueError("API key not found! Make sure it's set in the .env file.")

# Load the tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Function to fetch news articles
def fetch_news(company_name, api_key):
    url = 'https://newsapi.org/v2/everything'
    to_date = datetime.now()
    from_date = to_date - timedelta(days=10)

    params = {
        'q': company_name,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'language': 'en',
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    if data['status'] == 'ok':
        return data['articles']
    else:
        st.error("Error fetching articles")
        return []

# Function to save articles to a DataFrame
def save_to_dataframe(articles):
    data = {
        'date': [article['publishedAt'] for article in articles],
        'title': [article['title'] for article in articles],
        'description': [article['description'] for article in articles],
        'content': [article['content'] for article in articles],
        'url': [article['url'] for article in articles]
    }
    return pd.DataFrame(data)

# Function to analyze sentiment
def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())

    sentiments = ["negative", "neutral", "positive"]
    sentiment = sentiments[scores.argmax()]

    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    return sentiment, sentiment_map[sentiment]

# Streamlit App
def main():
    st.title("Sentiment Analysis of Company News")

    # Input field for company name
    company_name = st.text_input("Enter the company name:", "")

    if st.button("Analyze News"):
        if not company_name:
            st.warning("Please enter a company name.")
            return

        st.info(f"Fetching news articles about '{company_name}'...")

        # Fetch news articles
        articles = fetch_news(company_name, API_KEY)

        if not articles:
            st.error("No articles found.")
            return

        # Convert articles to DataFrame
        df = save_to_dataframe(articles)

        # Perform sentiment analysis
        st.info("Performing sentiment analysis...")
        df[['sentiment', 'sentiment_numeric']] = df['content'].apply(
            lambda x: pd.Series(analyze_sentiment(x) if pd.notna(x) else ("neutral", 0))
        )

        # Show the DataFrame
        st.write("Fetched News Articles:", df)

        # Generate Pie Chart
        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=["#ff9999", "#66b3ff", "#99ff99"]
        )
        ax.set_title("Sentiment Analysis of News Contents")
        st.pyplot(fig)

if __name__ == "__main__":
    main()



