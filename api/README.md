# Stock Price Prediction

Simple stock price prediction model using LSTM neural networks.

## Setup Instructions (/api)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create .env file:
   ```bash
   cp .env.demo .env
   ```

5. Run the prediction:
   ```bash
   python __init__.py
   ```

## Configuration

Edit the `.env` file to modify:
- TICKER_SYMBOL: Stock symbol to predict
- START_DATE: Historical data start date
- PREDICTION_DAYS: Number of days used for prediction
- DAYS_TO_PREDICT: Number of future days to predict