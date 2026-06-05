# Stock Market Price Prediction using Deep Learning (LSTM)

This project is a complete end-to-end Stock Market Price Prediction system built using Python, TensorFlow/Keras, and Streamlit. It uses a Long Short-Term Memory (LSTM) neural network to predict future stock closing prices based on historical market data.

## Features

- **Automated Data Collection:** Fetches historical stock data (Open, High, Low, Close, Volume) using the `yfinance` API.
- **Data Preprocessing:** Handles missing values and scales features using `MinMaxScaler`.
- **Deep Learning Model:** Implements a multi-layer LSTM network with Dropout layers for regularization.
- **Evaluation Metrics:** Evaluates model performance using RMSE, MAE, and R² Score.
- **Interactive UI:** Provides a user-friendly Streamlit web application for EDA, visualizations, and dynamic predictions.

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
