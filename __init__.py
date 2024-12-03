import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

TICKER_SYMBOL = os.getenv('TICKER_SYMBOL', 'AAPL')
START_DATE = os.getenv('START_DATE', '2020-01-01')
PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', 60))
DAYS_TO_PREDICT = int(os.getenv('DAYS_TO_PREDICT', 7))

def create_stock_prediction_model(ticker, prediction_days=60):
    print(f"\nDownloading data for {ticker}...")
    df = yf.download(ticker, start=START_DATE, progress=False)
    
    data = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    x_train = []
    y_train = []
    
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print("\nTraining model...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(units=50),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)
    
    return model, scaler, df

def predict_next_days(model, scaler, data, days_to_predict=7):
    predictions = []
    current_batch = data[-PREDICTION_DAYS:].reshape(-1, 1)
    current_batch = scaler.transform(current_batch)
    
    print(f"\nPredicting next {days_to_predict} days...")
    for _ in range(days_to_predict):
        current_batch_reshaped = current_batch[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 1)
        pred = model.predict(current_batch_reshaped, verbose=0)
        
        predictions.append(scaler.inverse_transform(pred)[0, 0])
        
        current_batch = np.append(current_batch, pred)
        current_batch = current_batch[1:]
    
    return predictions

def main():
    print(f"Starting stock prediction for {TICKER_SYMBOL}")
    
    model, scaler, stock_data = create_stock_prediction_model(
        TICKER_SYMBOL, 
        PREDICTION_DAYS
    )
    
    predictions = predict_next_days(
        model, 
        scaler, 
        stock_data['Close'].values,
        days_to_predict=DAYS_TO_PREDICT
    )
    
    print(f"\nLast closing price: ${stock_data['Close'].iloc[-1]:.2f}")
    print(f"\nPredicted prices for the next {DAYS_TO_PREDICT} days:")
    for i, price in enumerate(predictions, 1):
        print(f"Day {i}: ${price:.2f}")

if __name__ == "__main__":
    main()