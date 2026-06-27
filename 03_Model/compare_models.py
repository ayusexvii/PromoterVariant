#!/usr/bin/env python3
"""
Compare Leaky (with leakage) vs Honest (no leakage) Model Performance
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_comparison_data():
    """Load data for both models."""
    logger.info("📂 Loading data for comparison...")
    
    # Full matrix (for leaky model + target)
    full_df = pd.read_csv('processed/feature_matrix_chr22.csv.gz', compression='gzip')
    y = full_df['eqtl_slope'].values
    
    # Clean features (for honest model)
    X_clean = pd.read_csv('processed/clean_features_chr22.csv.gz', compression='gzip')
    
    # Leaky features = all except target
    X_leaky = full_df.drop(columns=['eqtl_slope'], errors='ignore')
    
    # Align everything
    mask = ~np.isnan(y)
    y = y[mask]
    X_clean = X_clean[mask].reset_index(drop=True)
    X_leaky = X_leaky[mask].reset_index(drop=True)
    
    logger.info(f"✅ Aligned dataset: {len(y):,} samples")
    return X_clean, X_leaky, y


def main():
    logger.info("=" * 85)
    logger.info("📊 COMPARING LEAKY vs HONEST MODELS")
    logger.info("=" * 85)
    
    X_clean, X_leaky, y = load_comparison_data()
    
    # Train-test split (same split for fair comparison)
    (X_clean_train, X_clean_test, 
     X_leaky_train, X_leaky_test, 
     y_train, y_test) = train_test_split(
        X_clean, X_leaky, y, test_size=0.2, random_state=42
    )
    
    # Load models
    try:
        leaky_model = joblib.load('processed/baseline_model.pkl')
        honest_model = joblib.load('processed/honest_model.pkl')
    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
        return
    
    # Predictions
    y_pred_leaky = leaky_model.predict(X_leaky_test)
    y_pred_honest = honest_model.predict(X_clean_test)
    
    # Metrics
    r2_leaky = r2_score(y_test, y_pred_leaky)
    r2_honest = r2_score(y_test, y_pred_honest)
    mae_leaky = mean_absolute_error(y_test, y_pred_leaky)
    mae_honest = mean_absolute_error(y_test, y_pred_honest)
    
    logger.info(f"\n📊 Leaky  Model → R²: {r2_leaky:.4f} | MAE: {mae_leaky:.4f}")
    logger.info(f"📊 Honest Model → R²: {r2_honest:.4f} | MAE: {mae_honest:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    # Leaky
    axes[0].scatter(y_test, y_pred_leaky, alpha=0.65, s=40, color='coral', edgecolors='white')
    minv, maxv = min(y_test.min(), y_pred_leaky.min()), max(y_test.max(), y_pred_leaky.max())
    axes[0].plot([minv, maxv], [minv, maxv], 'r--', lw=2.5, label='Ideal')
    axes[0].set_xlabel('Actual eQTL Slope')
    axes[0].set_ylabel('Predicted eQTL Slope')
    axes[0].set_title(f'Leaky Model (with leakage)\nR² = {r2_leaky:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Honest
    axes[1].scatter(y_test, y_pred_honest, alpha=0.65, s=40, color='steelblue', edgecolors='white')
    axes[1].plot([minv, maxv], [minv, maxv], 'r--', lw=2.5, label='Ideal')
    axes[1].set_xlabel('Actual eQTL Slope')
    axes[1].set_ylabel('Predicted eQTL Slope')
    axes[1].set_title(f'Honest Model (No Leakage)\nR² = {r2_honest:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path("processed").mkdir(exist_ok=True)
    plt.savefig('processed/honest_vs_leaky.png', dpi=180, bbox_inches='tight')
    logger.info("💾 Comparison plot saved: processed/honest_vs_leaky.png")
    
    # Summary
    drop_pct = ((r2_leaky - r2_honest) / r2_leaky * 100) if r2_leaky > 0 else 0
    
    logger.info("\n" + "="*75)
    logger.info("🔍 FINAL SUMMARY")
    logger.info("="*75)
    logger.info(f"Leaky R²   : {r2_leaky:.4f}")
    logger.info(f"Honest R²  : {r2_honest:.4f}")
    logger.info(f"Performance drop: {drop_pct:.1f}%")
    logger.info("\n✅ Honest model represents true biological signal from distance + gene context.")
    
    if r2_honest > 0.12:
        logger.info("🎯 Strong enough signal → Proceed to full genome scaling!")
    else:
        logger.info("📌 Moderate/weak signal → Consider adding PhyloP, sequence motifs, or epigenomics.")
    
    logger.info("\n✅ TASK 3 COMPLETE!")


if __name__ == "__main__":
    main()
