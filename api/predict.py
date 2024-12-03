from http.server import BaseHTTPRequestHandler
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

def create_stock_prediction_model(ticker, start_date, prediction_days):
    df = yf.download(ticker, start=start_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
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
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(units=50),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    return model, scaler, df

def predict_next_days(model, scaler, data, prediction_days, days_to_predict):
    predictions = []
    current_batch = data[-prediction_days:].reshape(-1, 1)
    current_batch = scaler.transform(current_batch)
    
    for _ in range(days_to_predict):
        current_batch_reshaped = current_batch[-prediction_days:].reshape(1, prediction_days, 1)
        pred = model.predict(current_batch_reshaped, verbose=0)
        predictions.append(scaler.inverse_transform(pred)[0, 0])
        current_batch = np.append(current_batch, pred)
        current_batch = current_batch[1:]
    
    return predictions

def handle_request(request):
    try:
        body = json.loads(request.get('body', '{}'))
        ticker = body['ticker'].upper()
        start_date = body['startDate']
        prediction_days = int(body['predictionDays'])
        days_to_predict = int(body['daysToPredict'])
        
        if prediction_days < 1 or days_to_predict < 1:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Prediction days must be positive numbers'})
            }
            
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid date format'})
            }
        
        model, scaler, stock_data = create_stock_prediction_model(
            ticker, 
            start_date, 
            prediction_days
        )
        
        predictions = predict_next_days(
            model, 
            scaler, 
            stock_data['Close'].values,
            prediction_days,
            days_to_predict
        )
        
        last_date = stock_data.index[-1]
        future_dates = []
        for i in range(days_to_predict):
            future_dates.append((last_date + timedelta(days=i+1)).strftime('%Y-%m-%d'))
        
        response = {
            'currentPrice': float(stock_data['Close'].iloc[-1]),
            'predictions': [float(price) for price in predictions],
            'dates': future_dates
        }
        
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handler(request):
    if request.get('method') == 'POST':
        return handle_request(request)
    else:
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method not allowed'})
        }