#!/usr/bin/env python3
"""
Full-Genome Honest Model Training
Trains on clean features from all chromosomes (no leakage).
"""

import pandas as pd
import numpy as np
import json
import logging
import gc
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


def load_full_data(
    matrix_path: str = "../02_Features/processed/training_matrix_allchr.csv.gz"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and prepare full-genome training data."""
    logger.info(f"📂 Loading full-genome training matrix: {matrix_path}")
    
    df = pd.read_csv(matrix_path, compression='gzip', low_memory=False)
    logger.info(f"✅ Loaded {len(df):,} rows with {df.shape[1]} columns")
    
    # Build clean (non-leaky) features
    df['abs_distance'] = df['distance_to_tss'].abs()
    
    # Gene-level feature
    gene_counts = df['promoter_gene'].value_counts()
    df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)
    
    # Select features
    feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count']
    X = df[feature_cols].copy()
    y = df['eqtl_slope'].values
    
    # Clean NaNs
    mask = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[mask].reset_index(drop=True)
    y = y[mask]
    
    logger.info(f"✅ Clean training data: {len(X):,} samples × {X.shape[1]} features")
    logger.info(f"   Target mean: {np.mean(y):.4f} ± {np.std(y):.4f}")
    
    return X, y


def train_full_model(X: pd.DataFrame, y: np.ndarray):
    """Train the honest full-genome model."""
    logger.info("\n🔧 Training HistGradientBoostingRegressor on full genome...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_depth=7,
        min_samples_leaf=8,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        n_iter_no_change=12
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation (optional - can be heavy)
    logger.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 FULL GENOME MODEL PERFORMANCE")
    logger.info("=" * 80)
    logger.info(f"Test R²           : {r2:.4f}")
    logger.info(f"Test MAE          : {mae:.4f}")
    logger.info(f"CV R² (mean)      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Target std        : {np.std(y):.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        imp.to_csv("processed/full_feature_importance.csv", index=False)
        logger.info("\n🔝 TOP FEATURES:")
        for _, row in imp.iterrows():
            logger.info(f"   {row['feature']:<25}: {row['importance']:.5f}")
    
    return model, r2, mae, cv_scores


def main():
    logger.info("=" * 90)
    logger.info("🧬 FULL GENOME HONEST MODEL TRAINING")
    logger.info("=" * 90)
    
    X, y = load_full_data()
    
    model, r2, mae, cv_scores = train_full_model(X, y)
    
    # Save artifacts
    Path("processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "processed/full_honest_model.pkl")
    
    metrics = {
        "r2_test": float(r2),
        "mae": float(mae),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "n_samples": len(X),
        "n_features": X.shape[1]
    }
    
    with open("processed/full_model_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("🎯 FINAL ASSESSMENT")
    logger.info("="*70)
    
    if r2 > 0.12:
        logger.info("✅ Strong honest signal across full genome!")
        logger.info("   → Model is biologically meaningful.")
    elif r2 > 0.05:
        logger.info("📊 Moderate signal. Good baseline.")
    else:
        logger.warning("⚠️  Weak signal. Consider adding conservation scores or sequence features.")
    
    logger.info("\n💾 Model saved: processed/full_honest_model.pkl")
    logger.info("✅ FULL GENOME TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
