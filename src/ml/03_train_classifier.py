"""
Train ML Classification Models for Risk Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
import xgboost as xgb
import pickle
import json
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'ml_results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("ML CLASSIFICATION - RISK PREDICTION")
    print("=" * 60)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(PROCESSED_DIR / 'features_ml.csv')
    
    # Encode target variable
    le = LabelEncoder()
    df['Risk_Category_encoded'] = le.fit_transform(df['Risk Category'])
    
    # Define features (exclude targets and IDs)
    exclude_cols = ['Patient ID', 'Timestamp', 'Risk Category', 'Risk_Category_encoded',
                   'Gender', 'Age_Group', 'BMI_Category']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Risk_Category_encoded']
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution:")
    print(df['Risk Category'].value_counts())
    
    # Split data (stratified by risk category)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    best_name = None
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        
        # Cross-validation
        print("Running cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                     cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"CV ROC-AUC = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train on full training set
        print("Training on full training set...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        results[name] = {
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std()),
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1),
            'test_roc_auc': float(roc_auc)
        }
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_name = name
    
    print(f"\n{'='*60}")
    print(f"✅ Best Model: {best_name} (ROC-AUC = {best_score:.4f})")
    print(f"{'='*60}")
    
    # Save best model
    print("\nSaving models...")
    with open(MODEL_DIR / 'best_classifier.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(MODEL_DIR / 'scaler_ml.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    with open(MODEL_DIR / 'feature_names_ml.json', 'w') as f:
        json.dump(feature_cols, f)
    
    with open(OUTPUT_DIR / 'classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for evaluation
    y_test_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': best_model.predict(X_test_scaled),
        'y_pred_proba': best_model.predict_proba(X_test_scaled)[:, 1]
    })
    y_test_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
    
    print("\n✅ Models saved!")
    print(f"  - Best model: {MODEL_DIR / 'best_classifier.pkl'}")
    print(f"  - Scaler: {MODEL_DIR / 'scaler_ml.pkl'}")
    print(f"  - Label encoder: {MODEL_DIR / 'label_encoder.pkl'}")
    print(f"  - Feature names: {MODEL_DIR / 'feature_names_ml.json'}")
    print(f"  - Results: {OUTPUT_DIR / 'classification_results.json'}")

if __name__ == '__main__':
    main()

