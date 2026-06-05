import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import date, timedelta
from src.data_utils import download_stock_data, preprocess_data, scale_data, create_sequences, fetch_news_sentiment

# Graceful degradation for tensorflow import
try:
    from src.model_utils import load_trained_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
import os

st.set_page_config(page_title="Advanced Stock Predictor", layout="wide")

st.title("📈 Advanced Stock Market Predictor")

st.markdown("""
This application uses Multivariate Deep Learning to predict the closing price of a stock.
It combines **Technical Indicators**, **Historical Price Data**, and **Financial News Sentiment (NLP)** for improved accuracy.
""")

# Sidebar settings
st.sidebar.header("Settings")
popular_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC-USD", "Custom..."]
selected_option = st.sidebar.selectbox("Select Stock", popular_tickers)

if selected_option == "Custom...":
    ticker = st.sidebar.text_input("Enter Custom Ticker", "NVDA")
else:
    ticker = selected_option

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

model_choice = st.sidebar.selectbox("Select Prediction Model", ["LSTM", "GRU", "LinearRegression"])
forecast_days = st.sidebar.slider("Future Forecast Days", min_value=1, max_value=30, value=7)

@st.cache_data
def load_data(ticker, start, end):
    try:
        df = download_stock_data(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        return preprocess_data(df)
    except Exception as e:
        return None

data_load_state = st.text('Loading data...')
df = load_data(ticker, start_date, end_date)

if df is None or df.empty:
    st.error(f"Could not load data for ticker '{ticker}'. Please ensure it is a valid Yahoo Finance ticker.")
    st.stop()
    
data_load_state.text('Data successfully loaded!')

# --- NLP NEWS SECTION ---
st.subheader("📰 Latest Financial News & Sentiment")
news_load_state = st.text(f"Fetching real-time news for {ticker}...")
real_sentiment_score, headlines = fetch_news_sentiment(ticker)
news_load_state.empty()

col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="Real-time VADER Sentiment Score", value=f"{real_sentiment_score:.3f}")
    if real_sentiment_score > 0.05:
        st.success("Overall Sentiment: POSITIVE")
    elif real_sentiment_score < -0.05:
        st.error("Overall Sentiment: NEGATIVE")
    else:
        st.info("Overall Sentiment: NEUTRAL")

with col2:
    if headlines:
        for article in headlines[:5]:
            st.markdown(f"- {article['title']} *(Score: {article['score']:.2f})*")
    else:
        st.write("No recent news found.")

st.markdown("---")

# --- EDA WITH CANDLESTICK & TECHNICAL INDICATORS ---
st.subheader("📊 Exploratory Data Analysis & Technical Indicators")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, subplot_titles=('Candlestick & Moving Averages', 'MACD & RSI'),
                    row_width=[0.3, 0.7])

# Candlestick chart
fig.add_trace(go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

# Moving Averages
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'), row=1, col=1)

# MACD
fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], line=dict(color='green', width=1), name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], line=dict(color='red', width=1), name='Signal'), row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, height=600)
st.plotly_chart(fig, use_container_width=True)


# --- PREDICTION SECTION ---
st.subheader(f"🤖 Model Predictions ({model_choice})")
model_path = f"models/{ticker}_{model_choice.lower()}_model.keras"

if not TF_AVAILABLE:
    st.error("⚠️ TensorFlow is not available in the current environment. Please run the app in an environment with Python 3.8 - 3.11 to view the predictions.")
elif not os.path.exists(model_path):
    st.warning(f"No pre-trained {model_choice} model found for {ticker}. Please run `python train_pipeline.py --ticker {ticker}` to train the models first.")
else:
    st.info(f"Loaded pre-trained {model_choice} model for {ticker}.")
    model = load_trained_model(model_path)
    
    # Process data for prediction
    scaled_data, scaler = scale_data(df, feature_cols=['Close', 'Sentiment_Score'])
    seq_length = 60
    
    if len(scaled_data) <= seq_length:
        st.error("Not enough data to create sequences for prediction.")
    else:
        # 1. Evaluate Historical Accuracy
        X, y = create_sequences(scaled_data, seq_length=seq_length)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        predictions = model.predict(X, verbose=0)
        
        pred_padded = np.zeros((len(predictions), 2))
        pred_padded[:, 0] = predictions.flatten()
        predictions_inv = scaler.inverse_transform(pred_padded)[:, 0]
        
        y_padded = np.zeros((len(y), 2))
        y_padded[:, 0] = y.flatten()
        y_inv = scaler.inverse_transform(y_padded)[:, 0]
        
        dates = df['Date'].values[seq_length:]
        
        # 2. Future Forecast (N days)
        # We start with the last available sequence
        last_sequence = scaled_data[-seq_length:].copy()
        
        # Scale the real_sentiment_score for injection
        dummy_for_scaling = np.array([[0, real_sentiment_score]])
        scaled_real_sentiment = scaler.transform(dummy_for_scaling)[0, 1]
        
        future_predictions = []
        current_seq = last_sequence.copy()
        
        for _ in range(forecast_days):
            # Inject live sentiment into the current day we are predicting from
            current_seq[-1, 1] = scaled_real_sentiment
            
            # Reshape and predict
            seq_reshaped = np.reshape(current_seq, (1, seq_length, 2))
            next_pred_scaled = model.predict(seq_reshaped, verbose=0)[0][0]
            
            # Inverse transform to get real price
            next_pred_padded = np.zeros((1, 2))
            next_pred_padded[0, 0] = next_pred_scaled
            next_pred_real = scaler.inverse_transform(next_pred_padded)[0, 0]
            
            future_predictions.append(next_pred_real)
            
            # Slide window: drop oldest, append newest (with real sentiment)
            new_row = np.array([next_pred_scaled, scaled_real_sentiment])
            current_seq = np.vstack((current_seq[1:], new_row))

        # Generate future dates (skipping weekends roughly)
        last_date = pd.to_datetime(dates[-1])
        future_dates = []
        days_added = 0
        while days_added < forecast_days:
            last_date += timedelta(days=1)
            if last_date.weekday() < 5: # Monday to Friday
                future_dates.append(last_date)
                days_added += 1

        # Plot Historical + Future
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=dates, y=y_inv, mode='lines', name='Actual Price', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=dates, y=predictions_inv, mode='lines', name=f'{model_choice} (Historical)', line=dict(color='orange')))
        
        # Connect the last historical point to the future forecast
        future_plot_dates = [dates[-1]] + future_dates
        future_plot_prices = [y_inv[-1]] + future_predictions
        
        fig_pred.add_trace(go.Scatter(x=future_plot_dates, y=future_plot_prices, mode='lines+markers', name=f'{forecast_days}-Day Forecast', line=dict(color='red', dash='dash')))
        
        fig_pred.layout.update(title_text=f'Historical Performance & {forecast_days}-Day Forecast ({model_choice})', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Show metrics
        st.success(f"Successfully forecasted the next {forecast_days} trading days using {model_choice} & Real-time NLP Sentiment!")
