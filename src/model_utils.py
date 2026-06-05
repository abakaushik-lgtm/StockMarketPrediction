import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
def create_lstm_model(input_shape):
    """
    Builds and compiles the LSTM model architecture.
    """
    model = Sequential()
    
    # First LSTM layer with Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Third LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_gru_model(input_shape):
    """
    Builds and compiles the GRU model architecture.
    """
    model = Sequential()
    
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_lr_model(input_shape):
    """
    Builds a Linear Regression equivalent model using Keras.
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
    """
    Trains the LSTM model.
    """
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=validation_data,
        verbose=1
    )
    return history

def evaluate_model(y_true, y_pred):
    """
    Calculates RMSE, MAE, MSE, and R2 Score.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2
    }

def save_trained_model(model, filepath='models/lstm_model.keras'):
    """
    Saves the trained model to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_trained_model(filepath='models/lstm_model.keras'):
    """
    Loads a trained model from disk.
    """
    if os.path.exists(filepath):
        model = load_model(filepath)
        print(f"Loaded model from {filepath}")
        return model
    else:
        raise FileNotFoundError(f"No model found at {filepath}")
