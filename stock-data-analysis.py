import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time

def download_stock_data(symbol, period, interval):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    data['Symbol'] = symbol
    data['Timeframe'] = interval
    return data

def save_to_database(data, db_name, table_name):
    conn = sqlite3.connect(db_name)
    data.to_sql(table_name, conn, if_exists='append', index=True)
    conn.close()

def prepare_features(data):
    # Price-related features
    data['Returns'] = data['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Volume-related features
    data['Volume_Change'] = data['Volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    data['Relative_Volume'] = (data['Volume'] / data['Volume'].rolling(window=20).mean()).replace([np.inf, -np.inf], np.nan)
    
    return data

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan)

def create_target(data, forward_days=5):
    data['Target'] = data['Close'].pct_change(periods=forward_days).shift(-forward_days).replace([np.inf, -np.inf], np.nan)
    return data

def train_model(features, target):
    print("Preparing data for model training...")
    valid_indices = ~(features.isna().any(axis=1) | target.isna() | np.isinf(features).any(axis=1) | np.isinf(target))
    X = features[valid_indices]
    y = target[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse:.4f}")
    return model

def rank_tickers(predictions_df):
    ranked_tickers = predictions_df.groupby('Symbol')['Prediction'].mean().sort_values(ascending=False)
    return ranked_tickers

def main():
    start_time = time.time()
    
    print("Reading stock symbols...")
    df_stocks = pd.read_csv('stocks.csv')
    symbols = df_stocks['Symbol'].tolist()

    print(f"Downloading data for {len(symbols)} stocks...")
    for symbol in tqdm(symbols, desc="Downloading stock data"):
        # Download 15-minute data for the past month
        data_15min = download_stock_data(symbol, 30, '15m')
        save_to_database(data_15min, 'summarytickers.db', 'stock_data_15min')

        # Download daily data for the past month
        data_daily = download_stock_data(symbol, 30, '1d')
        save_to_database(data_daily, 'summarytickers.db', 'stock_data_daily')

    print("Preparing data for machine learning...")
    conn = sqlite3.connect('summarytickers.db')
    data_15min = pd.read_sql('SELECT * FROM stock_data_15min', conn)
    data_daily = pd.read_sql('SELECT * FROM stock_data_daily', conn)
    conn.close()

    print("Preparing features and target for 15-minute data...")
    data_15min = prepare_features(data_15min)
    data_15min = create_target(data_15min)

    features = ['Returns', 'MA5', 'MA20', 'RSI', 'Volume_Change', 'Relative_Volume']
    X = data_15min[features]
    y = data_15min['Target']
    model = train_model(X, y)

    print("Preparing daily data for predictions...")
    data_daily = prepare_features(data_daily)
    X_daily = data_daily[features]
    
    print("Making predictions on daily data...")
    X_daily_clean = X_daily.replace([np.inf, -np.inf], np.nan).dropna()
    predictions = model.predict(X_daily_clean)
    
    data_daily.loc[X_daily_clean.index, 'Prediction'] = predictions
    
    print("Saving predictions to database...")
    save_to_database(data_daily[['Symbol', 'Timeframe', 'Prediction']], 'summarytickers.db', 'daily_predictions')

    print("Ranking tickers...")
    ranked_tickers = rank_tickers(data_daily.dropna(subset=['Prediction']))
    
    print("\nTickers ranked by predicted performance (from highest to lowest):")
    print(ranked_tickers)

    print("\nSaving ranked tickers to CSV file...")
    ranked_tickers.to_csv('ranked_tickers.csv', header=True)

    print("\nFeature Importances:")
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
