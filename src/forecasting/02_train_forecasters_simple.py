"""
Train Simple Time Series Forecasting Models (Moving Average, Linear Regression)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not available")

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'forecast_results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_sequences(data, sequence_length=24):
    """Create sequences for forecasting"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def moving_average_forecast(data, window=24):
    """Simple moving average forecast"""
    forecast = []
    for i in range(window, len(data)):
        forecast.append(np.mean(data[i-window:i]))
    return np.array(forecast)

def linear_regression_forecast(X, y):
    """Linear regression forecast"""
    if not HAS_SKLEARN:
        return None, None
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def main():
    """Main forecasting pipeline"""
    print("=" * 60)
    print("TIME SERIES FORECASTING (Simple Methods)")
    print("=" * 60)
    
    # Load hourly aggregated data
    print("\nLoading hourly aggregated data...")
    df = pd.read_csv(PROCESSED_DIR / 'hourly_vital_signs.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"Loaded {len(df)} hourly records")
    
    # Forecast Heart Rate
    target_col = 'Heart_Rate'
    data = df[target_col].values
    
    # Split train/test (80/20)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"\nTrain: {len(train_data)} records")
    print(f"Test: {len(test_data)} records")
    
    # 1. Moving Average Forecast
    print("\n1. Moving Average Forecast...")
    window = 24  # 24 hours
    ma_forecast = moving_average_forecast(data, window)
    ma_test = data[window:]
    
    # Get test portion (align lengths)
    test_start_idx = split_idx - window
    ma_test_actual = ma_test[test_start_idx:]
    ma_test_pred = ma_forecast[test_start_idx:]
    
    # Ensure same length
    min_len = min(len(ma_test_actual), len(ma_test_pred))
    ma_test_actual = ma_test_actual[:min_len]
    ma_test_pred = ma_test_pred[:min_len]
    
    # 2. Linear Regression Forecast (if sklearn available)
    if HAS_SKLEARN:
        print("\n2. Linear Regression Forecast...")
        sequence_length = 24
        X_train, y_train = create_sequences(train_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)
        
        lr_model, lr_pred = linear_regression_forecast(X_train, y_train)
        
        if lr_model:
            lr_test_pred = lr_model.predict(X_test)
            lr_test_actual = y_test
        else:
            lr_test_pred, lr_test_actual = None, None
    else:
        lr_model, lr_test_pred, lr_test_actual = None, None, None
    
    # Save predictions (ensure all arrays have same length)
    predictions_dict = {
        'ma_actual': ma_test_actual,
        'ma_pred': ma_test_pred
    }
    
    if lr_test_pred is not None and lr_test_actual is not None:
        # Align lengths
        min_lr_len = min(len(lr_test_actual), len(ma_test_actual))
        predictions_dict['lr_actual'] = lr_test_actual[:min_lr_len]
        predictions_dict['lr_pred'] = lr_test_pred[:min_lr_len]
        # Also trim MA to match
        predictions_dict['ma_actual'] = ma_test_actual[:min_lr_len]
        predictions_dict['ma_pred'] = ma_test_pred[:min_lr_len]
    
    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df.to_csv(OUTPUT_DIR / 'forecast_predictions.csv', index=False)
    print("✅ Predictions saved")
    
    # Save models
    if lr_model:
        with open(MODEL_DIR / 'linear_regression_forecaster.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        print("✅ Linear regression model saved")
    
    print("\n" + "=" * 60)
    print("✅ Forecasting models trained!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()

