"""
Train Time Series Forecasting Models (LSTM and Prophet)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not available, will skip LSTM training")

# Try to import sklearn (optional)
try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not available, will use manual scaling")

# Try to import Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠️  Prophet not available, will only use LSTM")

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'forecast_results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LSTM Model (if PyTorch available)
if HAS_TORCH:
    class LSTMForecaster(nn.Module):
        def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5):
            super(LSTMForecaster, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])  # Use last timestep
            return out

    class TimeSeriesDataset(Dataset):
        def __init__(self, data, sequence_length=24):
            self.data = data
            self.sequence_length = sequence_length
        
        def __len__(self):
            return len(self.data) - self.sequence_length
        
        def __getitem__(self, idx):
            x = self.data[idx:idx+self.sequence_length]
            y = self.data[idx+self.sequence_length]
            return torch.FloatTensor(x), torch.FloatTensor(y)
else:
    # Dummy classes if PyTorch not available
    class LSTMForecaster:
        pass
    class TimeSeriesDataset:
        pass

def create_sequences(data, sequence_length=24):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def train_lstm(df, target_col='Heart_Rate', sequence_length=24, epochs=50):
    """Train LSTM model"""
    if not HAS_TORCH:
        print("⚠️  PyTorch not available, skipping LSTM training")
        return None, None, None, None
    
    print(f"\nTraining LSTM for {target_col}...")
    
    # Prepare data
    data = df[['Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 
               'Temperature', 'Oxygen_Sat']].values
    
    # Scale data
    if HAS_SKLEARN:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        # Manual scaling
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_scaled = (data - data_min) / (data_max - data_min + 1e-8)
        scaler = type('Scaler', (), {
            'transform': lambda x: (x - data_min) / (data_max - data_min + 1e-8),
            'inverse_transform': lambda x: x * (data_max - data_min + 1e-8) + data_min
        })()
    
    # Create sequences
    X, y = create_sequences(data_scaled, sequence_length)
    
    # Split train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TimeSeriesDataset(data_scaled, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = LSTMForecaster(input_size=5, hidden_size=64, num_layers=2, output_size=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test.to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
        y_test_np = y_test.numpy()
        
        # Inverse transform
        y_pred_original = scaler.inverse_transform(y_pred)
        y_test_original = scaler.inverse_transform(y_test_np)
    
    return model, scaler, y_test_original, y_pred_original

def train_prophet(df, target_col='Heart_Rate'):
    """Train Prophet model"""
    if not HAS_PROPHET:
        return None, None, None, None
    
    print(f"\nTraining Prophet for {target_col}...")
    
    # Prepare data for Prophet
    prophet_data = df[['Timestamp', target_col]].copy()
    prophet_data.columns = ['ds', 'y']
    prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
    
    # Split train/test
    split_idx = int(len(prophet_data) * 0.8)
    train_data = prophet_data[:split_idx]
    test_data = prophet_data[split_idx:]
    
    # Train Prophet
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(train_data)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test_data))
    forecast = model.predict(future)
    
    # Get predictions for test period
    forecast_test = forecast.tail(len(test_data))
    y_pred = forecast_test['yhat'].values
    y_test = test_data['y'].values
    
    return model, forecast_test, y_test, y_pred

def main():
    """Main forecasting pipeline"""
    print("=" * 60)
    print("TIME SERIES FORECASTING")
    print("=" * 60)
    
    # Load hourly aggregated data
    print("\nLoading hourly aggregated data...")
    df = pd.read_csv(PROCESSED_DIR / 'hourly_vital_signs.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"Loaded {len(df)} hourly records")
    
    # Train LSTM (if available)
    if HAS_TORCH:
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)
        lstm_model, lstm_scaler, y_test_lstm, y_pred_lstm = train_lstm(df)
        
        if lstm_model is not None:
            # Save LSTM model
            torch.save(lstm_model.state_dict(), MODEL_DIR / 'lstm_forecaster.pt')
            with open(MODEL_DIR / 'lstm_scaler.pkl', 'wb') as f:
                pickle.dump(lstm_scaler, f)
            print("✅ LSTM model saved")
        else:
            y_test_lstm, y_pred_lstm = None, None
    else:
        print("\n⚠️  Skipping LSTM training (PyTorch not available)")
        y_test_lstm, y_pred_lstm = None, None
    
    # Train Prophet (if available)
    if HAS_PROPHET:
        print("\n" + "="*60)
        print("Training Prophet Model")
        print("="*60)
        prophet_model, forecast_df, y_test_prophet, y_pred_prophet = train_prophet(df, 'Heart_Rate')
        
        if prophet_model:
            # Save Prophet model
            with open(MODEL_DIR / 'prophet_model.pkl', 'wb') as f:
                pickle.dump(prophet_model, f)
            forecast_df.to_csv(OUTPUT_DIR / 'prophet_forecast.csv', index=False)
            print("✅ Prophet model saved")
    
    # Save predictions
    predictions_dict = {}
    if y_test_lstm is not None and y_pred_lstm is not None:
        predictions_dict['y_test_lstm'] = y_test_lstm[:, 0]  # Heart Rate
        predictions_dict['y_pred_lstm'] = y_pred_lstm[:, 0]
    
    if HAS_PROPHET and prophet_model:
        predictions_dict['y_test_prophet'] = y_test_prophet
        predictions_dict['y_pred_prophet'] = y_pred_prophet
    
    if predictions_dict:
        predictions_df = pd.DataFrame(predictions_dict)
        predictions_df.to_csv(OUTPUT_DIR / 'forecast_predictions.csv', index=False)
    
    print("\n" + "=" * 60)
    print("✅ Forecasting models trained!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()

