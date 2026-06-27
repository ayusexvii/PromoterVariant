#!/usr/bin/env python3
"""
Model Evaluation with Diagnostic Plots for Promoter Variant Prediction
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(
    features_path: str = "processed/feature_matrix_chr22.csv.gz",
    target_path: str = "processed/target_chr22.csv.gz",
    model_path: str = "processed/baseline_model.pkl"
):
    """Load features, target, and trained model."""
    logger.info("📂 Loading data and model...")
    
    try:
        X = pd.read_csv(features_path, compression='gzip', low_memory=False)
        y = pd.read_csv(target_path, compression='gzip').iloc[:, 0].values
        model = joblib.load(model_path)
        
        logger.info(f"✅ Features: {X.shape[0]:,} × {X.shape[1]}")
        logger.info(f"✅ Target  : {len(y):,}")
        
        # Clean NaNs
        mask = X.notna().all(axis=1) & ~np.isnan(y)
        X_clean = X[mask].reset_index(drop=True)
        y_clean = y[mask]
        
        logger.info(f"✅ Clean samples: {len(X_clean):,}")
        return X_clean, y_clean, model
        
    except FileNotFoundError as e:
        logger.error(f"Missing file: {e}")
        raise
    except Exception as e:
        logger.error(f"Loading error: {e}")
        raise


def generate_evaluation_plots(X_test, y_test, y_pred, model, output_dir: str = "processed"):
    """Generate and save diagnostic plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    residuals = y_test - y_pred
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7, s=35, c='steelblue', edgecolors='white')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual eQTL Slope')
    axes[0, 0].set_ylabel('Predicted eQTL Slope')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7, s=35, c='coral', edgecolors='white')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted eQTL Slope')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.75, color='mediumseagreen')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Residual Distribution (std = {residuals.std():.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = X_test.columns
        idx = np.argsort(importance)[-12:]
        
        axes[1, 1].barh(range(len(idx)), importance[idx], color='steelblue', edgecolor='black')
        axes[1, 1].set_yticks(range(len(idx)))
        axes[1, 1].set_yticklabels(feature_names[idx])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top 12 Feature Importances')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Save importance table
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        imp_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        logger.info(f"💾 Feature importance saved")
        
        print("\n🔝 TOP 10 FEATURES:")
        for _, row in imp_df.head(10).iterrows():
            print(f"   {row['feature']:<30} : {row['importance']:.5f}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_plots.png", dpi=180, bbox_inches='tight')
    logger.info(f"💾 Plots saved to: {output_dir}/evaluation_plots.png")
    
    return r2, mae, residuals


def main(
    features_path: str = "processed/feature_matrix_chr22.csv.gz",
    target_path: str = "processed/target_chr22.csv.gz",
    model_path: str = "processed/baseline_model.pkl"
):
    logger.info("=" * 80)
    logger.info("📊 MODEL EVALUATION & DIAGNOSTICS")
    logger.info("=" * 80)
    
    X, y, model = load_data(features_path, target_path, model_path)
    
    # Train-test split for evaluation
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    r2, mae, residuals = generate_evaluation_plots(X_test, y_test, y_pred, model)
    
    logger.info("\n" + "="*70)
    logger.info("🔍 INTERPRETATION")
    logger.info("="*70)
    logger.info(f"Test R²          : {r2:.4f}")
    logger.info(f"Test MAE         : {mae:.4f}")
    
    corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
    if corr > 0.35:
        logger.warning("⚠️  Possible heteroscedasticity detected")
    else:
        logger.info("✅ Residuals appear well-behaved")
    
    logger.info("\n✅ EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
