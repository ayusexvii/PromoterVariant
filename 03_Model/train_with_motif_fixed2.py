#!/usr/bin/env python3
"""
Train model with motif disruption features.
Fixed: Add engineered features after merge.
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
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not motif_path.exists() or not train_matrix_path.exists():
        print("❌ ERROR: Data files missing.", file=sys.stderr)
        sys.exit(1)
        
    print("📂 Loading motif disruption features...")
    motif_df = pd.read_csv(motif_path, compression='gzip')
    print(f"✅ Loaded {len(motif_df):,} motif metrics.")
    
    print("📂 Loading training matrix...")
    train_df = pd.read_csv(train_matrix_path, compression='gzip')
    print(f"✅ Loaded {len(train_df):,} base variant coordinates.")
    
    # Standardize chromosome keys
    train_df['clean_chrom'] = clean_chrom_column(train_df['chrom'] if 'chrom' in train_df.columns else train_df['Chromosome'])
    motif_df['clean_chrom'] = clean_chrom_column(motif_df['chrom'] if 'chrom' in motif_df.columns else motif_df['Chromosome'])
    
    pos_col_train = 'pos' if 'pos' in train_df.columns else 'Start'
    pos_col_motif = 'pos' if 'pos' in motif_df.columns else 'Start'
    
    train_df['merge_key'] = train_df['clean_chrom'] + '_' + train_df[pos_col_train].astype(str)
    motif_df['merge_key'] = motif_df['clean_chrom'] + '_' + motif_df[pos_col_motif].astype(str)
    
    # Merge
    motif_cols = ['merge_key', 'max_shift', 'sum_shift', 'num_affected', 'novel_kmers', 'lost_kmers']
    if 'disruption_intensity' in motif_df.columns:
        motif_cols.append('disruption_intensity')
        
    merged = train_df.merge(motif_df[motif_cols].drop_duplicates(subset=['merge_key']), on='merge_key', how='inner')
    print(f"✅ Merged: {len(merged):,} overlapping variants")
    
    if len(merged) == 0:
        print("❌ ERROR: Zero rows after merge.", file=sys.stderr)
        sys.exit(1)
    
    # --- ADD ENGINEERED FEATURES ---
    print("🔧 Adding engineered features...")
    merged['abs_distance'] = merged['distance_to_tss'].abs()
    
    # Gene variant count
    gene_counts = merged['promoter_gene'].value_counts().to_dict()
    merged['gene_variant_count'] = merged['promoter_gene'].map(gene_counts)
    
    # Features
    feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP', 
                    'max_shift', 'num_affected', 'novel_kmers', 'lost_kmers']
    if 'disruption_intensity' in merged.columns:
        feature_cols.append('disruption_intensity')
    
    print(f"   Features: {feature_cols}")
        
    X = merged[feature_cols].copy()
    y = merged['eqtl_slope'].values
    
    clean_mask = X.notna().all(axis=1) & np.isfinite(y)
    X, y = X[clean_mask], y[clean_mask]
    print(f"✅ Clean data: {len(X):,} samples, {X.shape[1]} features")
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n🚀 Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_depth=5, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("📊 MOTIF MODEL PERFORMANCE")
    print("=" * 70)
    print(f"   Test R²: {r2:.4f}")
    print(f"   Test MAE: {mae:.4f}")
    
    print("   Running 5-fold CV...")
    cv = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"   CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
    
    # Permutation importance
    print("\n🔍 Feature importance via permutation...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 FEATURE IMPORTANCES:")
    for row in imp_df.itertuples():
        print(f"   {row.feature:<25}: {row.importance:.4f} (± {row.std_dev:.4f})")
        
    # Save
    joblib.dump(model, output_dir / 'motif_model.pkl')
    metrics = {
        "r2_test": float(r2), 
        "mae": float(mae), 
        "cv_r2_mean": float(cv.mean()), 
        "cv_r2_std": float(cv.std()), 
        "samples": len(X),
        "feature_importances": dict(zip(imp_df['feature'], imp_df['importance']))
    }
    with open(output_dir / 'motif_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n💾 Model saved: processed/motif_model.pkl")
    print("\n✅ COMPLETE!")

if __name__ == "__main__":
    main()
