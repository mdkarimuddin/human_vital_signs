"""
Evaluate Time Series Forecasts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs' / 'forecast_results'

def calculate_metrics(y_true, y_pred):
    """Calculate forecast metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("TIME SERIES FORECAST EVALUATION")
    print("=" * 60)
    
    # Load predictions
    print("\nLoading predictions...")
    predictions_df = pd.read_csv(OUTPUT_DIR / 'forecast_predictions.csv')
    
    # Evaluate Moving Average
    print("\n=== Moving Average Forecast Evaluation ===")
    if 'ma_actual' in predictions_df.columns and 'ma_pred' in predictions_df.columns:
        y_test_ma = predictions_df['ma_actual'].values
        y_pred_ma = predictions_df['ma_pred'].values
        
        ma_metrics = calculate_metrics(y_test_ma, y_pred_ma)
        print(f"MAE:  {ma_metrics['MAE']:.4f}")
        print(f"RMSE: {ma_metrics['RMSE']:.4f}")
        print(f"MAPE: {ma_metrics['MAPE']:.2f}%")
    else:
        ma_metrics = None
        y_test_ma, y_pred_ma = None, None
    
    # Evaluate Linear Regression (if available)
    if 'lr_actual' in predictions_df.columns and 'lr_pred' in predictions_df.columns:
        print("\n=== Linear Regression Forecast Evaluation ===")
        y_test_lr = predictions_df['lr_actual'].values
        y_pred_lr = predictions_df['lr_pred'].values
        
        lr_metrics = calculate_metrics(y_test_lr, y_pred_lr)
        print(f"MAE:  {lr_metrics['MAE']:.4f}")
        print(f"RMSE: {lr_metrics['RMSE']:.4f}")
        print(f"MAPE: {lr_metrics['MAPE']:.2f}%")
    else:
        lr_metrics = None
        y_test_lr, y_pred_lr = None, None
    
    # 1. Forecast Plot - Moving Average
    print("\n1. Creating forecast plots...")
    if y_test_ma is not None and y_pred_ma is not None:
        plt.figure(figsize=(14, 6))
        plot_len = min(200, len(y_test_ma))
        plt.plot(y_test_ma[:plot_len], label='Actual', alpha=0.7)
        plt.plot(y_pred_ma[:plot_len], label='Moving Average Forecast', alpha=0.7)
        plt.title('Moving Average Forecast: Heart Rate')
        plt.xlabel('Time Step')
        plt.ylabel('Heart Rate (bpm)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'ma_forecast_plot.png', dpi=300)
        plt.close()
        print("✅ Saved: ma_forecast_plot.png")
    
    # 2. Forecast Plot - Linear Regression (if available)
    if y_test_lr is not None and y_pred_lr is not None:
        plt.figure(figsize=(14, 6))
        plot_len = min(200, len(y_test_lr))
        plt.plot(y_test_lr[:plot_len], label='Actual', alpha=0.7)
        plt.plot(y_pred_lr[:plot_len], label='Linear Regression Forecast', alpha=0.7)
        plt.title('Linear Regression Forecast: Heart Rate')
        plt.xlabel('Time Step')
        plt.ylabel('Heart Rate (bpm)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'lr_forecast_plot.png', dpi=300)
        plt.close()
        print("✅ Saved: lr_forecast_plot.png")
    
    # 3. Error Distribution
    print("\n2. Creating error distribution...")
    if y_test_ma is not None and y_pred_ma is not None:
        errors_ma = y_test_ma - y_pred_ma
    else:
        errors_ma = None
    
    if errors_ma is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(errors_ma, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Forecast Error')
        plt.ylabel('Frequency')
        plt.title(f'Moving Average Forecast Error Distribution\nMean Error: {errors_ma.mean():.4f}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'error_distribution.png', dpi=300)
        plt.close()
        print("✅ Saved: error_distribution.png")
    
    # 4. Scatter Plot
    print("\n3. Creating scatter plot...")
    if y_test_ma is not None and y_pred_ma is not None:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_ma, y_pred_ma, alpha=0.5)
        plt.plot([y_test_ma.min(), y_test_ma.max()], 
                 [y_test_ma.min(), y_test_ma.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Heart Rate')
        plt.ylabel('Predicted Heart Rate')
        plt.title('Moving Average Forecast: Actual vs Predicted')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'forecast_scatter.png', dpi=300)
        plt.close()
        print("✅ Saved: forecast_scatter.png")
    
    # Save metrics
    metrics_dict = {}
    if ma_metrics:
        metrics_dict['Moving_Average'] = ma_metrics
    if lr_metrics:
        metrics_dict['Linear_Regression'] = lr_metrics
    
    import json
    with open(OUTPUT_DIR / 'forecast_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print("✅ Saved: forecast_metrics.json")
    
    print("\n" + "=" * 60)
    print("✅ Forecast evaluation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()

