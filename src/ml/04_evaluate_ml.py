"""
Evaluate ML Models and Create Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                            classification_report)

# Try to import shap
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not available, skipping SHAP visualizations")

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'ml_results'

def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("ML MODEL EVALUATION & VISUALIZATION")
    print("=" * 60)
    
    # Load everything
    print("\nLoading models and data...")
    with open(MODEL_DIR / 'best_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(MODEL_DIR / 'scaler_ml.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(MODEL_DIR / 'label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    with open(MODEL_DIR / 'feature_names_ml.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load test predictions
    predictions_df = pd.read_csv(OUTPUT_DIR / 'predictions.csv')
    y_test = predictions_df['y_true'].values
    y_pred = predictions_df['y_pred'].values
    y_pred_proba = predictions_df['y_pred_proba'].values
    
    # Load test data for SHAP
    df_test = pd.read_csv(PROCESSED_DIR / 'features_ml.csv')
    # Get same test split
    from sklearn.model_selection import train_test_split
    df_test['Risk_Category_encoded'] = le.transform(df_test['Risk Category'])
    exclude_cols = ['Patient ID', 'Timestamp', 'Risk Category', 'Risk_Category_encoded',
                   'Gender', 'Age_Group', 'BMI_Category']
    feature_cols = [col for col in df_test.columns if col not in exclude_cols]
    X_test = df_test[feature_cols]
    _, X_test_split, _, _ = train_test_split(
        X_test, df_test['Risk_Category_encoded'], test_size=0.2, 
        random_state=42, stratify=df_test['Risk_Category_encoded']
    )
    X_test_scaled = scaler.transform(X_test_split)
    
    # 1. Confusion Matrix
    print("\n1. Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300)
    plt.close()
    print("✅ Saved: confusion_matrix.png")
    
    # 2. ROC Curve
    print("\n2. Creating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=300)
    plt.close()
    print("✅ Saved: roc_curve.png")
    
    # 3. Feature Importance
    print("\n3. Creating feature importance plot...")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[-20:]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importance[indices])
        plt.yticks(range(20), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features for Risk Prediction')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300)
        plt.close()
        print("✅ Saved: feature_importance.png")
    
    # 4. SHAP Analysis
    if HAS_SHAP:
        print("\n4. Computing SHAP values...")
        try:
            explainer = shap.TreeExplainer(model)
            sample_size = min(500, len(X_test_scaled))  # Use up to 500 samples
            X_sample = X_test_scaled[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification (shap_values might be a list or 3D array)
            if isinstance(shap_values, list):
                # For binary classification, use the positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Ensure shap_values is numpy array
            shap_values = np.array(shap_values)
            
            # If 3D array (samples, features, classes), extract positive class
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Use positive class (High Risk = 1)
            
            # Calculate mean absolute SHAP values for feature ranking
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Get top 20 features
            top_n = min(20, len(feature_names))
            top_indices = np.argsort(mean_shap)[-top_n:][::-1].tolist()
            top_features = [feature_names[i] for i in top_indices]
            mean_shap_top = mean_shap[top_indices]
            
            # 4a. Improved Summary plot (beeswarm) - top 20 features only
            # Create larger figure with more space for x-axis
            fig = plt.figure(figsize=(18, 12))
            shap.summary_plot(shap_values[:, top_indices], 
                            X_sample[:, top_indices], 
                            feature_names=top_features,
                            max_display=top_n,
                            show=False)
            plt.title('SHAP Summary Plot: Top 20 Most Important Features', 
                     fontsize=18, pad=25, fontweight='bold')
            # Adjust margins to prevent x-axis clipping
            plt.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.12)
            # Get current axes and adjust x-axis label position
            ax = plt.gca()
            ax.xaxis.label.set_size(12)
            plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.3, edgecolor='none')
            plt.close()
            print("✅ Saved: shap_summary.png (improved - larger size, fixed x-axis)")
            
            # 4b. Bar plot (mean absolute SHAP values) - cleaner version
            fig, ax = plt.subplots(figsize=(12, 10))
            colors = plt.cm.viridis(np.linspace(0, 1, len(mean_shap_top)))
            bars = ax.barh(range(len(mean_shap_top)), mean_shap_top, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(mean_shap_top)))
            ax.set_yticklabels(top_features, fontsize=10)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance: Mean Absolute SHAP Values (Top 20)', 
                        fontsize=14, pad=15, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'shap_bar_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Saved: shap_bar_plot.png")
            
            # 4c. Custom feature importance comparison
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(14, 8))
                model_importance = model.feature_importances_[top_indices]
                x = np.arange(len(top_features))
                width = 0.35
                
                # Normalize for comparison
                mean_shap_norm = mean_shap_top / (mean_shap_top.max() + 1e-8)
                model_imp_norm = model_importance / (model_importance.max() + 1e-8)
                
                ax.barh(x - width/2, mean_shap_norm, width, label='SHAP Importance (normalized)', 
                       alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
                ax.barh(x + width/2, model_imp_norm, width, label='Model Importance (normalized)', 
                       alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
                ax.set_yticks(x)
                ax.set_yticklabels(top_features, fontsize=10)
                ax.set_xlabel('Normalized Importance Score', fontsize=12, fontweight='bold')
                ax.set_title('Feature Importance Comparison: SHAP vs Model (Top 20)', 
                           fontsize=14, pad=15, fontweight='bold')
                ax.legend(fontsize=11, loc='lower right')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'shap_vs_model_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print("✅ Saved: shap_vs_model_importance.png")
            
        except Exception as e:
            print(f"⚠️  SHAP visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n4. Skipping SHAP analysis (not available)")
    
    # 5. Classification Report
    print("\n5. Generating classification report...")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\nClassification Report:")
    print(report)
    
    with open(OUTPUT_DIR / 'classification_report.txt', 'w') as f:
        f.write(report)
    print("✅ Saved: classification_report.txt")
    
    print("\n" + "=" * 60)
    print("✅ All evaluations complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()

