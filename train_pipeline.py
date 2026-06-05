import os
import numpy as np
import pandas as pd
from src.data_utils import download_stock_data, preprocess_data, scale_data, create_sequences, split_data
from src.model_utils import create_lstm_model, create_gru_model, create_lr_model, train_model, evaluate_model, save_trained_model

def run_pipeline(ticker='AAPL', start_date='2015-01-01', epochs=20, batch_size=32):
    print(f"--- Starting Pipeline for {ticker} ---")
    
    # 1. Data Collection
    df = download_stock_data(ticker, start_date=start_date)
    print(f"Data downloaded. Shape: {df.shape}")
    
    # 2. Preprocessing & Sentiment Generation & Tech Indicators
    df = preprocess_data(df)
    print(f"Features ready: {df.columns.tolist()}")
    
    # 3. Scaling
    scaled_data, scaler = scale_data(df, feature_cols=['Close', 'Sentiment_Score'])
    
    # 4. Sequence Creation
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length=seq_length)
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2]) # (seq_length, 2)
    
    # 6. Model Architectures to Train
    models_to_train = {
        'LSTM': create_lstm_model(input_shape),
        'GRU': create_gru_model(input_shape),
        'LinearRegression': create_lr_model(input_shape)
    }
    
    for model_name, model in models_to_train.items():
        print(f"\n========== Training {model_name} Model ==========")
        
        # Train
        history = train_model(
            model, 
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_test, y_test)
        )
        
        # Evaluate
        predictions = model.predict(X_test)
        
        pred_padded = np.zeros((len(predictions), 2))
        pred_padded[:, 0] = predictions.flatten()
        predictions_inv = scaler.inverse_transform(pred_padded)[:, 0]
        
        y_test_padded = np.zeros((len(y_test), 2))
        y_test_padded[:, 0] = y_test.flatten()
        y_test_inv = scaler.inverse_transform(y_test_padded)[:, 0]
        
        metrics = evaluate_model(y_test_inv, predictions_inv)
        print(f"\n--- {model_name} Evaluation Metrics ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Save Model
        save_trained_model(model, filepath=f'models/{ticker}_{model_name.lower()}_model.keras')
        
    print("\n--- Pipeline Completed Successfully ---")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Stock Predictors")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock Ticker (e.g., AAPL, GOOGL)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    
    run_pipeline(ticker=args.ticker, epochs=args.epochs, batch_size=args.batch_size)
