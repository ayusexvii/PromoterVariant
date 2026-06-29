#!/usr/bin/env python3
"""
Train model WITHOUT gene_variant_count to see true biological signal.
Clean version - no text processing needed.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import json
from pathlib import Path

def clean_chrom_column(series):
    """Uniformly normalize chromosome strings."""
    return series.astype(str).str.replace('chr', '', case=False).str.split('.').str[0].str.strip()

def main():
    print("=" * 75)
    print("🧬 TRAINING UNBIASED MODEL (BIOLOGICAL SIGNAL PURIFICATION)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    motif_path = script_dir / '../02_Features/processed/motif_features_full.csv.gz'
    train_matrix_path = script_dir / '../02_Features/processed/training_matrix_with_conservation.csv.gz'
    output_dir = script_dir / 'processed'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not motif_path.exists() or not train_matrix_path.exists():
        print("❌ ERROR: Source matrices missing.")
        return
        
    print("📂 Sourcing motif dataset...")
    motif_df = pd.read_csv(motif_path, compression='gzip')
    print("📂 Sourcing core feature matrix...")
    train_df = pd.read_csv(train_matrix_path, compression='gzip')
    
    # Standardize keys
    train_df['clean_chrom'] = clean_chrom_column(train_df['chrom'] if 'chrom' in train_df.columns else train_df['Chromosome'])
    motif_df['clean_chrom'] = clean_chrom_column(motif_df['chrom'] if 'chrom' in motif_df.columns else motif_df['Chromosome'])
    
    pos_col_train = 'pos' if 'pos' in train_df.columns else 'Start'
    pos_col_motif = 'pos' if 'pos' in motif_df.columns else 'Start'
    
    train_df['key'] = train_df['clean_chrom'] + '_' + train_df[pos_col_train].astype(str)
    motif_df['key'] = motif_df['clean_chrom'] + '_' + motif_df[pos_col_motif].astype(str)
    
    # FIXED: Columns are already numeric, no text processing needed
    motif_subset = ['key', 'max_shift', 'num_affected', 'novel_kmers', 'lost_kmers']
    merged = train_df.merge(motif_df[motif_subset].drop_duplicates(subset=['key']), on='key', how='inner')
    merged['abs_distance'] = merged['distance_to_tss'].abs()
    
    print(f"✅ Clean overlapping baseline variants matched: {len(merged):,}")
    
    if len(merged) == 0:
        print("❌ ERROR: Zero overlapping rows.")
        return
        
    # Features WITHOUT gene_variant_count
    feature_cols = ['distance_to_tss', 'abs_distance', 'phyloP', 
                    'max_shift', 'num_affected', 'novel_kmers', 'lost_kmers']
    
    X = merged[feature_cols].copy()
    y = merged['eqtl_slope'].values
    
    clean_mask = X.notna().all(axis=1) & np.isfinite(y)
    X, y = X[clean_mask], y[clean_mask]
    print(f"🧬 Active Features: {X.shape[1]} | Samples: {len(X):,}")
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_depth=5, min_samples_leaf=10, random_state=42)
    print("🚀 Training model (No gene_variant_count)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "=" * 75)
    print("📊 UNBIASED MODEL METRICS")
    print("=" * 75)
    print(f"   Test R²: {r2:.4f}")
    print(f"   Test MAE: {mae:.4f}")
    
    print("   Calculating 5-Fold CV...")
    cv = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"   CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
    
    # Feature importance
    print("\n🔍 Extracting Permutation Importances...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 FEATURE IMPORTANCES:")
    for row in imp_df.itertuples():
        print(f"   {row.feature:<22}: {row.importance:.4f} (± {row.std_dev:.4f})")
        
    # Save
    joblib.dump(model, output_dir / 'model_no_gene_count.pkl')
    
    metrics = {
        "r2_test": float(r2),
        "mae": float(mae),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "samples": len(X),
        "features": feature_cols,
        "importances": dict(zip(imp_df['feature'], imp_df['importance']))
    }
    with open(output_dir / 'model_no_gene_count_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n💾 Model saved: processed/model_no_gene_count.pkl")
    print("=" * 75)
    print("✅ COMPLETE!")

if __name__ == "__main__":
    main()
