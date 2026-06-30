#!/usr/bin/env python3
"""
Train model on exact-match training matrix.
Position-specific predictions utilizing high-speed vectorized permutation calculations.
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
    print("🧬 HARDWARE-ACCELERATED EXACT-MATCH TRAINING CORE ENGINE")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    matrix_path = script_dir / "../02_Features/processed/exact_match_training_matrix.csv.gz"
    output_dir = script_dir / "processed"
    
    # Guarantee write target workspace exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not matrix_path.exists():
        print(f"❌ ERROR: Scaled exact match training matrix missing at: {matrix_path}", file=sys.stderr)
        print("Please complete the 'build_exact_match_matrix.py' step beforehand.", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Loading exact-match feature training matrix: {matrix_path.name}")
    df = pd.read_csv(matrix_path, compression='gzip')
    print(f"✅ Ingested {len(df):,} structural variant tracks with {len(df.columns)} base columns.")
    
    # 1. Dynamic Feature Schema Isolation
    # Prioritize advanced biochemical features while ignoring text tracking labels
    all_possible_features = [
        'distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP',
        'max_shift', 'num_affected', 'num_novel_kmers', 'num_lost_kmers', 'disruption_intensity'
    ]
    
    # Filter only features physically present in the incoming matrix file
    feature_cols = [c for c in all_possible_features if c in df.columns]
    
    if 'eqtl_slope' not in df.columns:
        print("❌ CRITICAL ERROR: Target dependent variable 'eqtl_slope' missing from data rows.", file=sys.stderr)
        sys.exit(1)
        
    X = df[feature_cols].copy()
    y = df['eqtl_slope'].values
    
    # Drop any remaining unparsed nulls or extreme infinite records
    clean_mask = X.notna().all(axis=1) & np.isfinite(y)
    X, y = X[clean_mask], y[clean_mask]
    
    print(f"✅ Clean Matrix Assembly Locked: {len(X):,} variants across {X.shape[1]} active feature columns.")
    print(f"   🧬 Engineered Inputs Profile: {feature_cols}")
    print(f"   🎯 Target Signal Metrics    : Mean={np.mean(y):.4f} ± {np.std(y):.4f}")
    
    # 2. Split Arrays Allocation Holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Partitioning Data Cores    -> Training Pool: {len(X_train):,} | Test Validation: {len(X_test):,}")
    
    # 3. Model Parameters Construction
    print("\n🚀 Spawning Histogram-Based Gradient Boosting Regressor core...")
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Comparative Inference Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "=" * 75)
    print("📊 TRACE-SPECIFIC EXACT-MATCH MODEL PERFORMANCE EVALUATION")
    print("=" * 75)
    print(f"   Test R² Predictive Accuracy Variance Score: {r2:.4f}")
    print(f"   Test Mean Absolute Error (MAE)            : {mae:.4f}")
    
    # Execute Cross-Validation over all underlying system threads (-1)
    print("   Evaluating 5-Fold Cross-Validation splits across hardware threads...")
    cv = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"   Cross-Validated R² Metric Performance      : {cv.mean():.4f} ± {cv.std():.4f}")
    
    # 5. FIXED: Direct Multi-Threaded Permutation Significance Extraction
    print("\n🔍 Extracting Model Feature Importance via Permutation Calibration...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 TRUE GENOMIC FEATURE IMPORTANCES:")
    for row in imp_df.itertuples():
        print(f"   {row.feature:<25}: {row.importance:.4f} (± {row.std_dev:.4f})")
        
    # 6. Serialize State and Export Metrics
    model_out_path = output_dir / 'exact_match_model.pkl'
    metrics_out_path = output_dir / 'exact_match_model_metrics.json'
    
    joblib.dump(model, model_out_path)
    
    metrics_payload = {
        "r2_test": float(r2),
        "mae": float(mae),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "samples": len(X),
        "features_count": X.shape[1],
        "feature_names_list": list(feature_cols),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "importances": dict(zip(imp_df['feature'], imp_df['importance']))
    }
    
    with open(metrics_out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"\n💾 Model state serialized safely to : {model_out_path}")
    print(f"💾 Evaluation JSON metrics cataloged to: {metrics_out_path}")
    print("=" * 75)
    print("✅ EXACT POSITION PIPELINE STABLY RUN AND CONCLUDED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
