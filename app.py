import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date
from src.data_utils import download_stock_data, preprocess_data, scale_data, create_sequences, fetch_news_sentiment
from src.model_utils import load_trained_model
import os

st.set_page_config(page_title="Stock Predictor with NLP", layout="wide")

st.title("📈 Stock Market Predictor + NLP Sentiment")

st.markdown("""
This application uses a Multivariate Deep Learning (LSTM) model to predict the closing price of a stock.
It combines **Historical Price Data** and **Financial News Sentiment (NLP)** for improved accuracy.
""")

# Sidebar settings
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

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

st.subheader("Exploratory Data Analysis")
fig_close = go.Figure()
fig_close.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price"))
fig_close.layout.update(title_text='Time Series Data for Close Price', xaxis_rangeslider_visible=True)
st.plotly_chart(fig_close, use_container_width=True)

# Prediction Section
st.subheader("Model Predictions")
model_path = f"models/{ticker}_lstm_model.keras"

if not os.path.exists(model_path):
    st.warning(f"No pre-trained model found for {ticker}. Please run `python train_pipeline.py --ticker {ticker}` to train the model first.")
else:
    st.info(f"Loaded pre-trained multivariate model for {ticker}.")
    model = load_trained_model(model_path)
    
    # Process data for prediction
    scaled_data, scaler = scale_data(df, feature_cols=['Close', 'Sentiment_Score'])
    seq_length = 60
    
    if len(scaled_data) <= seq_length:
        st.error("Not enough data to create sequences for prediction.")
    else:
        # Create sequences for the entire dataset to visualize
        X, y = create_sequences(scaled_data, seq_length=seq_length)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        predictions = model.predict(X)
        
        # Inverse transform padded array
        pred_padded = np.zeros((len(predictions), 2))
        pred_padded[:, 0] = predictions.flatten()
        predictions_inv = scaler.inverse_transform(pred_padded)[:, 0]
        
        y_padded = np.zeros((len(y), 2))
        y_padded[:, 0] = y.flatten()
        y_inv = scaler.inverse_transform(y_padded)[:, 0]
        
        # Plotly chart for actual vs predicted
        dates = df['Date'].values[seq_length:]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=dates, y=y_inv, mode='lines', name='Actual Price'))
        fig_pred.add_trace(go.Scatter(x=dates, y=predictions_inv, mode='lines', name='Predicted Price'))
        fig_pred.layout.update(title_text='Actual vs Predicted Close Price', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Predict Next Day
        last_sequence = scaled_data[-seq_length:].copy()
        
        # Inject the real-time sentiment score into the very last day of the sequence
        # index 1 is 'Sentiment_Score'
        # Scale the real_sentiment_score using the scaler
        dummy_for_scaling = np.array([[0, real_sentiment_score]])
        scaled_real_sentiment = scaler.transform(dummy_for_scaling)[0, 1]
        last_sequence[-1, 1] = scaled_real_sentiment
        
        last_sequence = np.reshape(last_sequence, (1, seq_length, 2))
        next_pred = model.predict(last_sequence)
        
        next_pred_padded = np.zeros((1, 2))
        next_pred_padded[0, 0] = next_pred[0][0]
        next_pred_inv = scaler.inverse_transform(next_pred_padded)[0, 0]
        
        st.metric(label=f"Predicted Next Day Close Price for {ticker} (using live sentiment)", value=f"${next_pred_inv:.2f}")
