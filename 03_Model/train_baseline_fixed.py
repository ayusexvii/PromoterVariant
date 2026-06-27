#!/usr/bin/env python3
"""
Baseline Model Training for Promoter Variant Expression Impact Prediction
Uses HistGradientBoostingRegressor (CPU-efficient).
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_feature_data(features_path: str = "processed/feature_matrix_chr22.csv.gz") -> Tuple[pd.DataFrame, np.ndarray]:
    """Load features and target from unified feature matrix."""
    logger.info("📂 Loading feature matrix...")
    
    try:
        df = pd.read_csv(features_path, compression='gzip', low_memory=False)
        logger.info(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Target is 'eqtl_slope' - remove it from features
        y = df['eqtl_slope'].values.ravel()
        X = df.drop(columns=['eqtl_slope'], errors='ignore')
        
        # Clean NaNs
        mask = X.notna().all(axis=1) & ~np.isnan(y)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        
        logger.info(f"✅ Clean data: {len(X):,} samples × {X.shape[1]} features")
        logger.info(f"   Target mean: {np.mean(y):.4f} ± {np.std(y):.4f}")
        return X, y
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray) -> Tuple[HistGradientBoostingRegressor, Dict[str, Any]]:
    """Train baseline model and evaluate."""
    logger.info("\n🔧 Training HistGradientBoostingRegressor...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        n_iter_no_change=10
    )
    
    model.fit(X_train, y_train)
    
    # Test evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    logger.info("\n" + "=" * 75)
    logger.info("📊 MODEL PERFORMANCE")
    logger.info("=" * 75)
    logger.info(f"Test R²          : {r2:.4f}")
    logger.info(f"Test MAE         : {mae:.4f}")
    logger.info(f"CV R² (mean)     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Target std       : {np.std(y):.4f}")
    
    if r2 < 0.05:
        logger.warning("⚠️  Very weak signal. Consider adding sequence context, conservation, or epigenetic features.")
    elif r2 > 0.25:
        logger.info("✅ Good signal detected for baseline!")
    else:
        logger.info("📊 Moderate signal. Room for improvement with better features.")
    
    # Return model AND metrics
    metrics = {
        "r2_test": float(r2),
        "mae": float(mae),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y))
    }
    
    return model, metrics


def save_artifacts(model, X: pd.DataFrame, metrics: dict, output_dir: str = "processed") -> None:
    """Save model, feature importance, and metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, f"{output_dir}/baseline_model.pkl")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        logger.info(f"\n📊 Top 10 features saved to {output_dir}/feature_importance.csv")
        
        # Print top features
        print("\n🔝 TOP 10 FEATURES:")
        for i, row in importance.head(10).iterrows():
            print(f"   {row['feature']:<25}: {row['importance']:.4f}")
    
    with open(f"{output_dir}/model_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n💾 Model saved: {output_dir}/baseline_model.pkl")
    logger.info(f"💾 Metrics saved: {output_dir}/model_metrics.json")


def main(features_path: str = "processed/feature_matrix_chr22.csv.gz", output_dir: str = "processed") -> None:
    logger.info("=" * 80)
    logger.info("🧬 BASELINE MODEL TRAINING - PROMOTER VARIANT IMPACT")
    logger.info("=" * 80)
    
    X, y = load_feature_data(features_path)
    
    model, metrics = train_and_evaluate(X, y)
    
    save_artifacts(model, X, metrics, output_dir)
    
    logger.info("\n✅ BASELINE TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
