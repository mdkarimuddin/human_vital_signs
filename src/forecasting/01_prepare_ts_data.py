"""
Prepare Time Series Data for Forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

def prepare_time_series_data(df):
    """
    Prepare data for time series forecasting
    """
    print("Preparing time series data...")
    
    # Sort by patient and timestamp
    df = df.sort_values(['Patient ID', 'Timestamp']).copy()
    
    # Select key vital signs for forecasting
    target_vars = ['Heart Rate', 'Systolic Blood Pressure', 
                   'Diastolic Blood Pressure', 'Body Temperature', 
                   'Oxygen Saturation']
    
    # Create time series datasets for each patient
    ts_data = []
    
    for patient_id in df['Patient ID'].unique()[:100]:  # Limit to 100 patients for manageability
        patient_data = df[df['Patient ID'] == patient_id].copy()
        
        if len(patient_data) < 10:  # Skip patients with too few records
            continue
        
        # Create time series entry
        ts_entry = {
            'Patient ID': patient_id,
            'timestamps': patient_data['Timestamp'].values,
            'heart_rate': patient_data['Heart Rate'].values,
            'systolic_bp': patient_data['Systolic Blood Pressure'].values,
            'diastolic_bp': patient_data['Diastolic Blood Pressure'].values,
            'temperature': patient_data['Body Temperature'].values,
            'oxygen_sat': patient_data['Oxygen Saturation'].values,
            'n_records': len(patient_data)
        }
        ts_data.append(ts_entry)
    
    print(f"Prepared time series for {len(ts_data)} patients")
    
    # Create aggregated time series (all patients combined, hourly)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.floor('H')
    
    hourly_agg = df.groupby('Hour').agg({
        'Heart Rate': 'mean',
        'Systolic Blood Pressure': 'mean',
        'Diastolic Blood Pressure': 'mean',
        'Body Temperature': 'mean',
        'Oxygen Saturation': 'mean'
    }).reset_index()
    hourly_agg.columns = ['Timestamp', 'Heart_Rate', 'Systolic_BP', 
                          'Diastolic_BP', 'Temperature', 'Oxygen_Sat']
    hourly_agg = hourly_agg.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"Aggregated hourly data: {len(hourly_agg)} records")
    
    return ts_data, hourly_agg

def main():
    """Main data preparation pipeline"""
    print("=" * 60)
    print("TIME SERIES DATA PREPARATION")
    print("=" * 60)
    
    # Load cleaned data
    print("\nLoading cleaned data...")
    df = pd.read_csv(PROCESSED_DIR / 'cleaned_vital_signs.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print(f"Loaded {len(df)} records")
    
    # Prepare time series data
    ts_data, hourly_agg = prepare_time_series_data(df)
    
    # Save hourly aggregated data (easier for forecasting)
    hourly_agg.to_csv(PROCESSED_DIR / 'hourly_vital_signs.csv', index=False)
    print(f"\n✅ Hourly aggregated data saved to: {PROCESSED_DIR / 'hourly_vital_signs.csv'}")
    
    # Save patient-level time series info
    ts_info = pd.DataFrame([{
        'Patient ID': entry['Patient ID'],
        'n_records': entry['n_records']
    } for entry in ts_data])
    ts_info.to_csv(PROCESSED_DIR / 'ts_patients_info.csv', index=False)
    print(f"✅ Time series info saved to: {PROCESSED_DIR / 'ts_patients_info.csv'}")
    
    print(f"\nTime series data ready for forecasting!")
    print(f"  - Hourly aggregated: {len(hourly_agg)} records")
    print(f"  - Patient-level series: {len(ts_data)} patients")

if __name__ == '__main__':
    main()

