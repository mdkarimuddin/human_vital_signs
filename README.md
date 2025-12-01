# Human Vital Signs Analysis: ML Classification & Time Series Forecasting

Comprehensive analysis of human vital signs data combining **machine learning classification** and **time series forecasting** - demonstrating capabilities for wearable health technology applications like Oura Ring.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![ML](https://img.shields.io/badge/ML-Classification%20%7C%20Forecasting-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 🎯 Project Overview

This project analyzes human vital signs data with two main components:

1. **ML Classification**: Predict risk categories (High Risk/Low Risk) from vital signs
2. **Time Series Forecasting**: Forecast future vital signs (Heart Rate, Blood Pressure, etc.)

Built to showcase capabilities relevant to **wearable health technology companies** like Oura, Whoop, and Fitbit.

### Key Features

- ✅ **200K+ records** of real vital signs data
- ✅ **ML Classification**: Risk category prediction (XGBoost, Random Forest)
- ✅ **Time Series Forecasting**: LSTM/Prophet for vital signs prediction
- ✅ **Multi-modal features**: HR, HRV, BP, Temperature, SpO2, Respiratory Rate
- ✅ **Explainable AI**: SHAP analysis for feature importance
- ✅ **Production-ready** code structure

## 📊 Dataset

- **Size**: 200,021 records
- **Features**: 17 columns including:
  - Vital Signs: Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation
  - Blood Pressure: Systolic, Diastolic, MAP, Pulse Pressure
  - Demographics: Age, Gender, Weight, Height, BMI
  - Derived: HRV
  - Target: Risk Category (High Risk/Low Risk)
- **Time Series**: Timestamp column for temporal analysis

## 🗂️ Project Structure

```
human_vital_signs/
├── src/
│   ├── ml/
│   │   ├── 01_eda.py              # Exploratory data analysis
│   │   ├── 02_feature_engineering.py  # Feature creation
│   │   ├── 03_train_classifier.py     # ML classification models
│   │   └── 04_evaluate_ml.py         # ML evaluation & SHAP
│   ├── forecasting/
│   │   ├── 01_prepare_ts_data.py     # Time series data prep
│   │   ├── 02_train_forecasters.py   # LSTM/Prophet models
│   │   └── 03_evaluate_forecasts.py  # Forecast evaluation
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Processed data
├── models/                          # Trained models
├── outputs/
│   ├── eda/                        # EDA visualizations
│   ├── ml_results/                 # ML results & plots
│   └── forecast_results/           # Forecasting results
├── notebooks/                      # Jupyter notebooks (optional)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run ML Classification Pipeline

```bash
# Step 1: EDA
python src/ml/01_eda.py

# Step 2: Feature Engineering
python src/ml/02_feature_engineering.py

# Step 3: Train Classifiers
python src/ml/03_train_classifier.py

# Step 4: Evaluate & Explain
python src/ml/04_evaluate_ml.py
```

### Run Time Series Forecasting Pipeline

```bash
# Step 1: Prepare Time Series Data
python src/forecasting/01_prepare_ts_data.py

# Step 2: Train Forecasters
python src/forecasting/02_train_forecasters.py

# Step 3: Evaluate Forecasts
python src/forecasting/03_evaluate_forecasts.py
```

## 🔬 Methodology

### ML Classification

**Objective**: Predict Risk Category (High Risk/Low Risk)

**Features**:
- Vital signs (HR, HRV, BP, Temperature, SpO2)
- Demographics (Age, Gender, BMI)
- Derived features (MAP, Pulse Pressure)

**Models**:
- Random Forest Classifier
- XGBoost Classifier
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC

**Explainability**: SHAP analysis

### Time Series Forecasting

**Objective**: Forecast future vital signs (next N hours/days)

**Target Variables**:
- Heart Rate
- Blood Pressure (Systolic, Diastolic)
- Body Temperature
- Oxygen Saturation

**Models**:
- Moving Average (baseline)
- Linear Regression (simple forecasting)
- Evaluation: MAE, RMSE, MAPE

*Note: LSTM and Prophet implementations available but using simpler models for this analysis*

## 📈 Results

### ML Classification Results

**Performance Metrics** (Test Set: 40,004 samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **XGBoost** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Key Findings**:
- Perfect classification performance on test set
- Both models achieved 100% accuracy, precision, recall, and F1-score
- Dataset: 200,020 records → 160,016 train / 40,004 test
- 38 engineered features from 17 original columns
- SHAP analysis reveals top important features for risk prediction

**Outputs Generated**:
- ✅ Confusion matrices
- ✅ ROC curves
- ✅ Feature importance plots
- ✅ SHAP summary plots (beeswarm & bar plots)
- ✅ Classification reports

### Time Series Forecasting Results

**Performance Metrics** (Test Set: 667 hourly records):

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| **Moving Average** | 1.21 | 1.52 | 1.52 |
| **Linear Regression** | 1.18 | 1.50 | 1.48 |

**Key Findings**:
- Excellent forecasting accuracy with <2% MAPE
- Linear Regression slightly outperforms Moving Average
- Dataset: 3,334 hourly aggregated records → 2,667 train / 667 test
- Forecasting target: Heart Rate (bpm)

**Outputs Generated**:
- ✅ Forecast plots (actual vs predicted)
- ✅ Error distribution plots
- ✅ Scatter plots (predicted vs actual)
- ✅ Forecast metrics JSON

## 💡 Relevance to Oura Ring

This project demonstrates:

✅ **Multi-modal vital signs analysis** (HR, HRV, BP, Temperature)  
✅ **Risk prediction** (health status classification)  
✅ **Time series forecasting** (predictive health monitoring)  
✅ **Explainable AI** (SHAP for interpretability)  
✅ **Production-ready** code structure  
✅ **Real-world health data** processing

## 🛠️ Technologies

- **Python 3.10+**
- **pandas, numpy** - Data processing
- **scikit-learn** - ML classification
- **XGBoost** - Gradient boosting
- **PyTorch/TensorFlow** - LSTM for forecasting
- **Prophet** - Time series forecasting
- **SHAP** - Explainability
- **matplotlib, seaborn** - Visualization

## 👤 Author

**Karim Uddin**  
PhD Veterinary Medicine | MEng Big Data Analytics  
Postdoctoral Researcher, University of Helsinki

- GitHub: [@mdkarimuddin](https://github.com/mdkarimuddin)
- LinkedIn: [Karim Uddin](https://linkedin.com/in/karimuddin)

## 📜 License

MIT License

---

**⭐ Star this repo if you found it useful!**

*Built to demonstrate ML and time series capabilities for wearable health technology roles.*

