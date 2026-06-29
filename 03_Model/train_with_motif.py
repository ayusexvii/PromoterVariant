#!/usr/bin/env python3
"""
Train model with motif disruption features.
Optimized to handle structural strings safely and run multi-core permutation testing.
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

def clean_chrom_column(series):
    """Normalize chromosome strings uniformly to avoid merge failures."""
    return series.astype(str).str.replace('chr', '', case=False).str.split('.').str[0].str.strip()

def main():
    print("=" * 70)
    print("🧬 HARDWARE-ACCELERATED REGULATORY MOTIF MODEL TRAINING ENGINE")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    motif_path = script_dir / '../02_Features/processed/motif_features_full.csv.gz'
    train_matrix_path = script_dir / '../02_Features/processed/training_matrix_with_conservation.csv.gz'
    output_dir = script_dir / 'processed'
    
    # Secure destination directories natively
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Data Vectors Safely
    if not motif_path.exists() or not train_matrix_path.exists():
        print(f"❌ ERROR: Subsystem data files missing. Ensure features step is fully complete.", file=sys.stderr)
        sys.exit(1)
        
    print("📂 Loading motif disruption features array...")
    motif_df = pd.read_csv(motif_path, compression='gzip')
    print(f"✅ Loaded {len(motif_df):,} motif metrics.")
    
    print("📂 Loading baseline conservation training matrix...")
    train_df = pd.read_csv(train_matrix_path, compression='gzip')
    print(f"✅ Loaded {len(train_df):,} base variant coordinates.")
    
    # 2. Robust Key Standardization (Prevents inner merge dropping data)
    train_df['clean_chrom'] = clean_chrom_column(train_df['chrom'] if 'chrom' in train_df.columns else train_df['Chromosome'])
    motif_df['clean_chrom'] = clean_chrom_column(motif_df['chrom'] if 'chrom' in motif_df.columns else motif_df['Chromosome'])
    
    pos_col_train = 'pos' if 'pos' in train_df.columns else 'Start'
    pos_col_motif = 'pos' if 'pos' in motif_df.columns else 'Start'
    
    train_df['merge_key'] = train_df['clean_chrom'] + '_' + train_df[pos_col_train].astype(str)
    motif_df['merge_key'] = motif_df['clean_chrom'] + '_' + motif_df[pos_col_motif].astype(str)
    
    # 3. Dynamic Text-to-Numeric Feature Engineering
    # Converting raw text string entries into robust numerical features to prevent model crashes
    for df_tmp in [motif_df]:
        for col_name, target in [('novel_kmers', 'num_novel_kmers'), ('lost_kmers', 'num_lost_kmers')]:
            if col_name in df_tmp.columns:
    merged = train_df.merge(motif_df[motif_cols].drop_duplicates(subset=['merge_key']), on='merge_key', how='inner')
    print(f"✅ Matrix synchronized cleanly. Overlapping variants available: {len(merged):,}")
    
    if len(merged) == 0:
        print("❌ CRITICAL ERROR: Synchronized overlap returned zero rows. Check your source keys.", file=sys.stderr)
        sys.exit(1)
        
    # 4. Feature Matrix Assembly
    feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP',                 
                    'max_shift', 'num_affected', 'num_novel_kmers', 'num_lost_kmers']
    if 'disruption_intensity' in merged.columns:
        feature_cols.append('disruption_intensity')
        
    X = merged[feature_cols].copy()
    y = merged['eqtl_slope'].values
    
    # Drop any leftover missing values or infinite targets safely
    clean_mask = X.notna().all(axis=1) & np.isfinite(y)
    X, y = X[clean_mask], y[clean_mask]
    print(f"✅ Cleaned training matrix verified: {len(X):,} samples across {X.shape[1]} active features.")
    
    # 5. Model Compilation Pass
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n🚀 Initializing Histogram-Based Gradient Boosting Regressor core...")
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_depth=5, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Comprehensive Metrics Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("📊 ENHANCED REGULATORY MOTIF MODEL PERFORMANCE")
    print("=" * 70)
    print(f"   Test R² Prediction Accuracy : {r2:.4f}")
    print(f"   Test Mean Absolute Error   : {mae:.4f}")
    
    print("   Running 5-Fold Cross-Validation splits across multi-thread cores...")
    cv = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"   Cross-Validated R² Metric   : {cv.mean():.4f} ± {cv.std():.4f}")
    
    # 7. FIXED: Calibrate Permutation Importance Metrics Directly
    print("\n🔍 Extracting Model Feature Importance via Permutation Calibration...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 CALIBRATED GENOMIC FEATURE IMPORTANCES:")
    for row in imp_df.itertuples():
        print(f"   {row.feature:<25}: {row.importance:.4f} (± {row.std_dev:.4f})")
        
    # 8. Serialize and Write State to Local Storage
    model_out = output_dir / 'motif_model.pkl'
    metrics_out = output_dir / 'motif_model_metrics.json'
    
    joblib.dump(model, model_out)
    
    metrics = {
        "r2_test": float(r2), 
        "mae": float(mae), 
        "cv_r2_mean": float(cv.mean()), 
        "cv_r2_std": float(cv.std()), 
        "samples": len(X),
        "feature_importances": dict(zip(imp_df['feature'], imp_df['importance']))
    }
    
    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n💾 Model serialized seamlessly to : {model_out}")
    print(f"💾 Metrics summary saved directly to: {metrics_out}")
    print("=" * 70)
    print("✅ PROCESS MATRIX SEAMLESSLY UPGRADED! TASK COMPLETE.")
    print("=" * 70)

if __name__ == "__main__":
    main()
