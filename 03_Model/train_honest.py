#!/usr/bin/env python3
"""
Honest Model Training - No Data Leakage
Trains only on pre-GTEx features (distance + gene count)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_honest_data(
    clean_features_path: str = "processed/clean_features_chr22.csv.gz",
    full_matrix_path: str = "processed/feature_matrix_chr22.csv.gz"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load clean features + target."""
    logger.info("📂 Loading clean features and target...")
    
    X = pd.read_csv(clean_features_path, compression='gzip', low_memory=False)
    master = pd.read_csv(full_matrix_path, compression='gzip', low_memory=False)
    
    y = master['eqtl_slope'].values
    
    # Align data
    mask = ~np.isnan(y)
    X = X[mask].reset_index(drop=True)
    y = y[mask]
    
    logger.info(f"✅ Clean features: {X.shape[1]} features × {len(X):,} samples")
    logger.info(f"✅ Target aligned: {len(y):,} values")
    
    return X, y


def train_honest_model(X: pd.DataFrame, y: np.ndarray):
    """Train honest model."""
    logger.info("\n🔧 Training Honest HistGradientBoostingRegressor...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=4,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        n_iter_no_change=15
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    logger.info("Running 5-fold CV...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    logger.info("\n" + "=" * 75)
    logger.info("📊 HONEST MODEL PERFORMANCE")
    logger.info("=" * 75)
    logger.info(f"Test R²         : {r2:.4f}")
    logger.info(f"Test MAE        : {mae:.4f}")
    logger.info(f"CV R² (mean)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Target std      : {np.std(y):.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        imp_df.to_csv("processed/honest_feature_importance.csv", index=False)
        
        logger.info("\n🔝 TOP 8 FEATURES:")
        for _, row in imp_df.head(8).iterrows():
            logger.info(f"   {row['feature']:<28}: {row['importance']:.5f}")
    
    return model, r2, mae, cv_scores


def main():
    logger.info("=" * 85)
    logger.info("🧬 HONEST MODEL TRAINING (No GTEx Leakage)")
    logger.info("=" * 85)
    
    X, y = load_honest_data()
    
    model, r2, mae, cv_scores = train_honest_model(X, y)
    
    # Save model
    Path("processed").mkdir(exist_ok=True)
    joblib.dump(model, "processed/honest_model.pkl")
    
    # Save metrics
    metrics = {
        "r2_test": float(r2),
        "mae": float(mae),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "n_samples": len(X),
        "n_features": X.shape[1]
    }
    
    with open("processed/honest_model_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("🎯 DECISION")
    logger.info("="*70)
    
    if r2 > 0.15:
        logger.info("✅ Strong honest signal! Ready to scale to full genome.")
    elif r2 > 0.05:
        logger.info("📊 Moderate signal. Consider adding PhyloP conservation or sequence features.")
    else:
        logger.warning("⚠️  Weak or no signal from distance/gene features alone.")
    
    logger.info("\n✅ HONEST TRAINING COMPLETE!")
    logger.info("💾 honest_model.pkl + metrics saved")


if __name__ == "__main__":
    main()
