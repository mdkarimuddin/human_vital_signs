# Human Vital Signs Project - Summary

## 🎯 Project Overview

Comprehensive analysis combining **ML Classification** and **Time Series Forecasting** for human vital signs data.

## 📊 Dataset

- **Size**: 200,021 records
- **Features**: 17 columns
- **Time Series**: Yes (Timestamp column)
- **Target**: Risk Category (High Risk/Low Risk)

## 🔧 Components

### 1. ML Classification Pipeline

**Objective**: Predict Risk Category from vital signs

**Scripts**:
1. `src/ml/01_eda.py` - Exploratory data analysis
2. `src/ml/02_feature_engineering.py` - Feature creation
3. `src/ml/03_train_classifier.py` - Train models (RF, XGBoost)
4. `src/ml/04_evaluate_ml.py` - Evaluate & visualize (SHAP)

**Outputs**:
- Risk category predictions
- Feature importance
- SHAP analysis
- ROC curves
- Confusion matrices

### 2. Time Series Forecasting Pipeline

**Objective**: Forecast future vital signs (HR, BP, Temperature, etc.)

**Scripts**:
1. `src/forecasting/01_prepare_ts_data.py` - Prepare time series data
2. `src/forecasting/02_train_forecasters.py` - Train models (LSTM, Prophet)
3. `src/forecasting/03_evaluate_forecasts.py` - Evaluate forecasts

**Outputs**:
- Forecast plots
- Accuracy metrics (MAE, RMSE, MAPE)
- Error distributions

## 🚀 Quick Start

### Run ML Classification

```bash
python src/ml/01_eda.py
python src/ml/02_feature_engineering.py
python src/ml/03_train_classifier.py
python src/ml/04_evaluate_ml.py
```

### Run Time Series Forecasting

```bash
python src/forecasting/01_prepare_ts_data.py
python src/forecasting/02_train_forecasters.py
python src/forecasting/03_evaluate_forecasts.py
```

## 📈 Expected Results

### ML Classification
- Risk category prediction accuracy
- Feature importance rankings
- SHAP explanations
- Model comparison (RF vs XGBoost)

### Time Series Forecasting
- Multi-step ahead forecasts
- Forecast accuracy metrics
- Model comparison (LSTM vs Prophet)

## 💡 Relevance to Oura

- ✅ Multi-modal vital signs analysis
- ✅ Risk prediction (health status)
- ✅ Predictive health monitoring
- ✅ Explainable AI
- ✅ Production-ready code

## 📁 Project Structure

```
human_vital_signs/
├── src/
│   ├── ml/              # ML classification scripts
│   └── forecasting/     # Time series forecasting scripts
├── data/
│   ├── raw/             # Original dataset
│   └── processed/       # Processed data
├── models/              # Trained models
├── outputs/
│   ├── eda/            # EDA visualizations
│   ├── ml_results/     # ML results
│   └── forecast_results/ # Forecasting results
└── README.md
```

## 🛠️ Technologies

- Python 3.10+
- scikit-learn (ML)
- XGBoost (Classification)
- PyTorch (LSTM)
- Prophet (Forecasting)
- SHAP (Explainability)

---

**Status**: Ready to run! 🚀

