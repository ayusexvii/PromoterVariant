#!/usr/bin/env python3
"""
Train final model on verified multi-tissue exact-match matrix.
Optimized with parallelized permutation feature evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import json
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 FINAL UNIFIED EXACT-MATCH MODEL TRAINING ENGINE")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. FIXED: Added correct unified matrix path to lookup checklist
    candidate_paths = [
        script_dir / "../02_Features/processed/unified_exact_match_matrix.csv.gz",
        script_dir / "../02_Features/processed/exact_match_training_clean.csv.gz",
        script_dir / "../02_Features/processed/unified_matrix_all_genes.csv.gz"
    ]
    
    matrix_path = None
    for path in candidate_paths:
        if path.exists():
            matrix_path = path
            break
            
    if matrix_path is None:
        print("❌ CRITICAL ERROR: Could not locate a valid unified feature matrix on disk!", file=sys.stderr)
        print(f"   Checked paths: {[p.name for p in candidate_paths]}", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Ingesting training feature matrix: {matrix_path.name}")
    df = pd.read_csv(matrix_path, compression='gzip')
    print(f"✅ Ingested {len(df):,} records across {len(df.columns)} matrix data columns.")
    
    # 2. Extract Valid Feature Schema Space
    feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP']
    
    # Verify target variable name variations case-insensitively
    slope_candidates = [c for c in df.columns if c.lower() in ['gtex_slope', 'eqtl_slope', 'slope']]
    if not slope_candidates:
        print("❌ CRITICAL ERROR: Could not find target effect slope vector inside data frame.", file=sys.stderr)
        sys.exit(1)
    target_col = slope_candidates[0]
    
    X = df[feature_cols].copy()
    y = df[target_col].values
    
    print(f"   Using Target Feature Track    : '{target_col}'")
    print(f"   Target Signal Distribution Profile: Mean={np.mean(y):.4f} ± {np.std(y):.4f}")
    
    # 3. Partition Data Splits Allocations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Data Blocks Locked         -> Train Set: {len(X_train):,} | Test Holdout: {len(X_test):,}")
    
    # 4. Fit Gradient Boosting Architecture
    print("\n🚀 Initializing HistGradientBoostingRegressor fitting cycle...")
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate Predictive Variance Accuracy
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "=" * 75)
    print("📊 UNIFIED EXACT-MATCH MODEL DEEP EVALUATION METRICS")
    print("=" * 75)
    print(f"   Test R² Prediction Accuracy Score  : {r2:.4f}")
    print(f"   Test Mean Absolute Error (MAE)     : {mae:.4f}")
    
    print("   Running multi-threaded 5-fold cross-validation pass...")
    cv = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"   Cross-Validated R² Metric Profile  : {cv.mean():.4f} ± {cv.std():.4f}")
    
    # 6. FIXED: Parallelized Permutation Significance Calculations
    print("\n🔍 Extracting True Feature Importance Metrics via Permutation Shuffling...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 TRUE COORD REGULATORY FEATURE RANKINGS:")
    for row in imp_df.itertuples():
        print(f"   {row.feature:<25}: {row.importance:.4f} (± {row.std_dev:.4f})")
        
    # 7. Serialize States and Export Files
    model_file_path = output_dir / 'final_exact_model.pkl'
    metrics_file_path = output_dir / 'final_exact_model_metrics.json'
    
    joblib.dump(model, model_file_path)
    
    metrics_payload = {
        "r2_test": float(r2),
        "mae": float(mae),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "samples_count": len(X),
        "features_evaluated": len(feature_cols),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "importances_ranking": dict(zip(imp_df['feature'], imp_df['importance']))
    }
    
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"\n💾 Model state serialized safely to : {model_file_path}")
    print(f"💾 Metrics summary JSON flushed to  : {metrics_file_path}")
    print("=" * 75)
    print("✅ MASTER TRAINING OPERATION PIPELINE STABLY RUN AND CONCLUDED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
