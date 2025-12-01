"""
Feature Engineering for ML Classification
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

def create_features(df):
    """
    Engineer features for risk classification
    """
    df = df.copy()
    
    print("Creating features...")
    
    # 1. Encode categorical variables
    df['Gender_encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # 2. Create risk indicators
    df['HR_risk'] = ((df['Heart Rate'] < 60) | (df['Heart Rate'] > 100)).astype(int)
    df['BP_risk'] = ((df['Systolic Blood Pressure'] > 140) | 
                     (df['Diastolic Blood Pressure'] > 90)).astype(int)
    df['Temp_risk'] = ((df['Body Temperature'] < 36.1) | 
                       (df['Body Temperature'] > 37.2)).astype(int)
    df['SpO2_risk'] = (df['Oxygen Saturation'] < 95).astype(int)
    
    # 3. Create composite risk score
    df['Composite_Risk_Score'] = (df['HR_risk'] + df['BP_risk'] + 
                                   df['Temp_risk'] + df['SpO2_risk'])
    
    # 4. Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], 
                             labels=['Young', 'Adult', 'Middle', 'Senior'])
    df['Age_Group_encoded'] = df['Age_Group'].cat.codes
    
    # 5. BMI categories
    df['BMI_Category'] = pd.cut(df['Derived_BMI'], 
                                bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['BMI_Category_encoded'] = df['BMI_Category'].cat.codes
    
    # 6. Blood pressure ratio
    df['BP_Ratio'] = df['Systolic Blood Pressure'] / (df['Diastolic Blood Pressure'] + 1)
    
    # 7. Vital signs stability (coefficient of variation for time series)
    # Group by patient and calculate CV (only for patients with multiple records)
    patient_stats = df.groupby('Patient ID').agg({
        'Heart Rate': ['mean', 'std'],
        'Systolic Blood Pressure': ['mean', 'std'],
        'Body Temperature': ['mean', 'std']
    }).reset_index()
    
    patient_stats.columns = ['Patient ID', 'HR_mean', 'HR_std', 
                            'SBP_mean', 'SBP_std', 'Temp_mean', 'Temp_std']
    
    # Fill NaN std values with 0 (for patients with single record)
    patient_stats = patient_stats.fillna(0)
    
    patient_stats['HR_CV'] = patient_stats['HR_std'] / (patient_stats['HR_mean'] + 1)
    patient_stats['SBP_CV'] = patient_stats['SBP_std'] / (patient_stats['SBP_mean'] + 1)
    patient_stats['Temp_CV'] = patient_stats['Temp_std'] / (patient_stats['Temp_mean'] + 1)
    
    # Fill any remaining NaN with 0
    patient_stats[['HR_CV', 'SBP_CV', 'Temp_CV']] = patient_stats[['HR_CV', 'SBP_CV', 'Temp_CV']].fillna(0)
    
    df = df.merge(patient_stats[['Patient ID', 'HR_CV', 'SBP_CV', 'Temp_CV']], 
                  on='Patient ID', how='left')
    
    # 8. Temporal features
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Cyclical encoding
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    print(f"Created {df.shape[1]} features")
    
    return df

def main():
    """Main feature engineering pipeline"""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load cleaned data
    print("\nLoading cleaned data...")
    df = pd.read_csv(PROCESSED_DIR / 'cleaned_vital_signs.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print(f"Loaded {len(df)} records")
    
    # Create features
    df_features = create_features(df)
    
    # Fill any remaining NaN values (handle categorical columns separately)
    initial_nan = df_features.isnull().sum().sum()
    if initial_nan > 0:
        # Fill numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
        
        # Fill categorical columns with mode or first category
        categorical_cols = df_features.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            if df_features[col].isnull().any():
                df_features[col] = df_features[col].fillna(df_features[col].mode()[0] if len(df_features[col].mode()) > 0 else df_features[col].cat.categories[0])
        
        # Fill object columns
        object_cols = df_features.select_dtypes(include=['object']).columns
        df_features[object_cols] = df_features[object_cols].fillna('Unknown')
        
        print(f"\nFilled {initial_nan} NaN values")
    
    print(f"\nFinal dataset shape: {df_features.shape}")
    print(f"Samples: {len(df_features)}")
    print(f"Features: {df_features.shape[1]}")
    
    # Save
    df_features.to_csv(PROCESSED_DIR / 'features_ml.csv', index=False)
    print(f"\n✅ Features saved to: {PROCESSED_DIR / 'features_ml.csv'}")

if __name__ == '__main__':
    main()

