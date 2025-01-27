from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime as dt

app = Flask(__name__)

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def calculate_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_social_sentiment_data():
    # Simulated example data (use Twitter API for real data)
    sentiment_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'text': ["Positive news about the stock"] * 365  # Sample repeated text
    })
    sentiment_data['sentiment'] = sentiment_data['text'].apply(calculate_sentiment)
    return sentiment_data

def prepare_data(symbol, start_date, end_date):
    stock_data = get_stock_data(symbol, start_date, end_date)
    sentiment_data = get_social_sentiment_data()

    # Merge by date
    stock_data['Date'] = stock_data.index
    data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date')
    data['sentiment'] = data['sentiment'].fillna(0)  # Fill NaN sentiments

    # Choose features and labels
    data['Close_shifted'] = data['Close'].shift(-1)  # Next day's price as label
    data = data.dropna()  # Drop NaN values
    return data[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment']], data['Close_shifted']

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)
    return model

@app.route('/predict', methods=['POST'])
def predict_stock_price():
    data = request.get_json()
    symbol = data['symbol']
    start_date = data['startDate']
    end_date = data['endDate']

    features, labels = prepare_data(symbol, start_date, end_date)
    model = train_model(features, labels)
    prediction = model.predict([features.iloc[-1]])  # Predict using the latest data

    return jsonify({
        'prediction': round(prediction[0], 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
