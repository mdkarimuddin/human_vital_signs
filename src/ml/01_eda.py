"""
Exploratory Data Analysis for Human Vital Signs - ML Classification
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Puhti
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'eda'

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load vital signs dataset"""
    print("=== Loading Data ===")
    
    # Load dataset
    df = pd.read_csv(RAW_DIR / 'human_vital_signs_dataset_2024.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def clean_data(df):
    """Clean and prepare data"""
    print("\n=== Cleaning Data ===")
    
    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Check for missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe())
    
    # Risk category distribution
    print("\n=== Risk Category Distribution ===")
    print(df['Risk Category'].value_counts())
    print(f"\nRisk Category proportions:")
    print(df['Risk Category'].value_counts(normalize=True))
    
    return df

def plot_distributions(df):
    """Create distribution plots"""
    print("\n=== Creating Distribution Plots ===")
    
    # Select key vital signs
    vital_signs = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 
                   'Oxygen Saturation', 'Systolic Blood Pressure', 
                   'Diastolic Blood Pressure']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(vital_signs):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df[col].mean():.2f}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'vital_signs_distributions.png', dpi=300)
    plt.close()
    print("✅ Saved: vital_signs_distributions.png")

def plot_risk_comparison(df):
    """Compare vital signs by risk category"""
    print("\n=== Creating Risk Category Comparison ===")
    
    vital_signs = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 
                   'Oxygen Saturation', 'Systolic Blood Pressure', 
                   'Diastolic Blood Pressure']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(vital_signs):
        for risk in df['Risk Category'].unique():
            data = df[df['Risk Category'] == risk][col]
            axes[idx].hist(data, bins=30, alpha=0.6, label=risk, edgecolor='black')
        axes[idx].set_title(f'{col} by Risk Category')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_category_comparison.png', dpi=300)
    plt.close()
    print("✅ Saved: risk_category_comparison.png")

def plot_correlations(df):
    """Create correlation matrix"""
    print("\n=== Creating Correlation Matrix ===")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove Patient ID
    numeric_cols = [col for col in numeric_cols if col != 'Patient ID']
    
    corr_matrix = df[numeric_cols].corr()
    
    # Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Vital Signs', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300)
    plt.close()
    print("✅ Saved: correlation_matrix.png")
    
    return corr_matrix

def plot_time_series(df):
    """Create time series plots for sample patients"""
    print("\n=== Creating Time Series Plots ===")
    
    # Get sample patients
    sample_patients = df['Patient ID'].unique()[:5]
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 15))
    
    for idx, patient_id in enumerate(sample_patients):
        patient_data = df[df['Patient ID'] == patient_id].sort_values('Timestamp')
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Plot Heart Rate
        ax.plot(patient_data['Timestamp'], patient_data['Heart Rate'], 
                'o-', color='blue', label='Heart Rate', alpha=0.7)
        
        # Plot Blood Pressure
        ax2.plot(patient_data['Timestamp'], patient_data['Systolic Blood Pressure'], 
                 's-', color='red', label='Systolic BP', alpha=0.7)
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Heart Rate (bpm)', color='blue')
        ax2.set_ylabel('Systolic BP (mmHg)', color='red')
        ax.set_title(f'Patient {patient_id}: Vital Signs Over Time')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_series_vital_signs.png', dpi=300)
    plt.close()
    print("✅ Saved: time_series_vital_signs.png")

def main():
    """Main EDA pipeline"""
    print("=" * 60)
    print("HUMAN VITAL SIGNS - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Clean data
    df = clean_data(df)
    
    # Visualizations
    plot_distributions(df)
    plot_risk_comparison(df)
    corr_matrix = plot_correlations(df)
    plot_time_series(df)
    
    # Save cleaned data
    df.to_csv(PROCESSED_DIR / 'cleaned_vital_signs.csv', index=False)
    print(f"\n✅ Processed data saved to: {PROCESSED_DIR / 'cleaned_vital_signs.csv'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("=== KEY FINDINGS ===")
    print(f"1. Total records: {len(df):,}")
    print(f"2. Unique patients: {df['Patient ID'].nunique()}")
    print(f"3. Risk distribution:")
    for risk, count in df['Risk Category'].value_counts().items():
        print(f"   {risk}: {count:,} ({count/len(df)*100:.1f}%)")
    print(f"4. Time span: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"5. Average Heart Rate: {df['Heart Rate'].mean():.1f} bpm")
    print(f"6. Average Systolic BP: {df['Systolic Blood Pressure'].mean():.1f} mmHg")
    print("=" * 60)
    print("\n✅ EDA Complete!")
    print(f"Visualizations saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

