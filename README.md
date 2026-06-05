# Stock Market Price Prediction using Deep Learning (LSTM)

This project is a complete end-to-end Stock Market Price Prediction system built using Python, TensorFlow/Keras, and Streamlit. It uses a Long Short-Term Memory (LSTM) neural network to predict future stock closing prices based on historical market data.

## Project Objectives and Features ("Stand Out" Upgrade)

- **Automated Data Collection:** Fetches historical stock data (Open, High, Low, Close, Volume) using the `yfinance` API (Yahoo Finance).
- **Multiple Stock Ticker Support:** Analyze any global stock by typing its ticker, with quick-select options for AAPL, TSLA, MSFT, and BTC-USD.
- **Data Preprocessing & Technical Indicators:** 
  - Handles missing values via forward/backward filling.
  - Automatically calculates **SMA_20**, **EMA_20**, **RSI_14**, and **MACD**.
  - Normalizes features using `MinMaxScaler` and creates sliding-window sequences.
- **Model Comparison (LSTM vs GRU vs Linear Regression):** Train, evaluate, and dynamically compare the performance of three different architectural approaches in the Streamlit UI.
- **Performance Metrics:** Evaluates model training and testing performance using **RMSE**, **MAE**, **MSE**, and **R² Score**.
- **Interactive UI & Candlestick Visualizations:** Provides a Streamlit web application featuring a Plotly Candlestick chart overlaid with Moving Averages and MACD subplots.
- **NLP Sentiment Analysis:** Analyzes the latest financial news using VADER sentiment analysis to inject live news polarity into the model predictions.
- **7-30 Day Future Forecasting:** Uses an autoregressive loop to predict and project stock prices multiple days into the future.
- **Cloud Deployment Ready:** Fully compatible with Streamlit Community Cloud for public deployment.

## Application Screenshots

*(After running the app locally, save a screenshot of the Streamlit dashboard as `screenshot.png` in the project root and it will display here)*
![App Screenshot](screenshot.png)

## Project Architecture

1.  **Data Extraction:** `yfinance`
2.  **Data Processing:** `pandas`, `numpy`, `scikit-learn`
3.  **Model Training:** `tensorflow`, `keras`
4.  **Web Application:** `streamlit`, `plotly`

## Installation and Setup

1.  **Clone the repository or navigate to the project directory.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the Model:**
    Before running the app, train the LSTM model for a specific ticker (e.g., AAPL). The model will be saved in the `models/` directory.
    ```bash
    python train_pipeline.py --ticker AAPL --epochs 20
    ```
4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## Project Structure

- `src/data_utils.py`: Functions for downloading data, handling missing values, and scaling.
- `src/model_utils.py`: Functions for defining the LSTM architecture, training, and evaluation.
- `train_pipeline.py`: The execution script for training and saving models.
- `app.py`: The interactive Streamlit user interface.
- `requirements.txt`: Python package dependencies.
- `project_report.md`: Detailed project report including methodology and analysis.
- `presentation.md`: Presentation slides content.
- `resume_description.md`: Resume bullet points.

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Streamlit
- Pandas, NumPy, Scikit-learn
- Plotly, Matplotlib
- yFinance
