# AI-social-sentiment-stock-predictor
# Stock Price Prediction Using Sentiment Analysis and Machine Learning

## Description
This project predicts stock prices using historical data and sentiment analysis. The financial data is fetched from Yahoo Finance, and sentiment analysis is performed on simulated social media data (you can extend this with real Twitter data). The prediction model is built using Linear Regression.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Sources](#data-sources)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
Follow these steps to get your development environment set up:

1. Clone the repository:
   ```bash
   git clone https://github.com/Monisha125/AI-Social-Sentiment-Stock-Predictor/tree/main
   ```

2. Navigate to the project directory:
   ```bash
   cd AI-Social-Sentiment-Stock-Predictor
   ```

3. Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask server by running the following command:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`.

3. Use the web form to enter the stock symbol, start date, and end date. Click "Predict" to receive a predicted stock price for the next day.

## Features
- Fetches historical stock data using the Yahoo Finance API (`yfinance`).
- Performs sentiment analysis using TextBlob (simulated social media data for now).
- Predicts future stock prices using a simple Linear Regression model.

## Data Sources
- **Stock Data**: Yahoo Finance (via `yfinance` library).
- **Sentiment Data**: Simulated positive sentiment (you can extend this with real social media data like Twitter).

## Model
- **Algorithm**: Linear Regression.
- **Libraries**: 
  - `Flask` for the web application.
  - `pandas` for data handling.
  - `yfinance` for fetching stock data.
  - `TextBlob` for sentiment analysis.
  - `scikit-learn` for building and evaluating the regression model.

## Contributing
We welcome contributions to improve this project! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add a feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.
