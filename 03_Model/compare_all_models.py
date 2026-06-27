#!/usr/bin/env python3
"""
Compare all models: Distance-only vs PhyloP
Guarantees mathematically rigorous row-synchronization across testing splits.
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    print("=" * 80)
    print("📊 VALIDATED REPRODUCIBLE MODEL COMPARISON ENGINE")
    print("=" * 80)
    
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / '../02_Features/processed/training_matrix_with_conservation.csv.gz'
    processed_dir = script_dir / 'processed'
    
    # 1. Load data vector safely
    if not data_path.exists():
        print(f"❌ ERROR: Training matrix not found at: {data_path}", file=sys.stderr)
        sys.exit(1)
        
    print("📂 Loading evaluation features matrix...")
    df = pd.read_csv(data_path, compression='gzip')
    
    # 2. Reconstruct explicit features matrix
    df['abs_distance'] = df['distance_to_tss'].abs()
    gene_counts = df['promoter_gene'].value_counts().to_dict()
    df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)
    
    # Define exact feature tracking subsets
    features_without = ['distance_to_tss', 'abs_distance', 'gene_variant_count']
    features_with = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP']
    
    # Filter matrix rows globally to remove missing blocks and infinite targets
    clean_mask = df[features_with].notna().all(axis=1) & np.isfinite(df['eqtl_slope'])
    df_clean = df[clean_mask].copy()
    
    print(f"✅ Cleaned evaluation subset preserved: {len(df_clean):,} variants.")
    
    # 3. CRITICAL IMPLEMENTATION FIX: Single split to guarantee perfect data synchronization
    # Split the clean dataframe itself rather than splitting individual arrays independently
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    
    y_test = test_df['eqtl_slope'].values
    X_test_without = test_df[features_without].copy()
    X_test_with = test_df[features_with].copy()
    
    print(f"🎯 Row validation aligned cleanly. Synchronized Test Pool Count: {len(y_test):,}")
    
    # 4. Defensive Model Loading Engine
    old_model = None
    possible_baselines = ['full_model.pkl', 'full_honest_model.pkl', 'baseline_model.pkl']
    
    for model_name in possible_baselines:
        target_path = processed_dir / model_name
        if target_path.exists():
            try:
                old_model = joblib.load(target_path)
                print(f"✅ Baseline model matched and loaded successfully: {model_name}")
                break
            except Exception as e:
                print(f"⚠️ Warning: Found {model_name} but failed to read it: {e}", file=sys.stderr)
                
    new_model_path = processed_dir / 'conservation_model.pkl'
    if not new_model_path.exists():
        print(f"❌ ERROR: Conservation model not found at: {new_model_path}", file=sys.stderr)
        print("Please run 'train_with_conservation.py' before executing this comparison.", file=sys.stderr)
        sys.exit(1)
        
    try:
        new_model = joblib.load(new_model_path)
        print("✅ Conservation model loaded successfully: conservation_model.pkl")
    except Exception as e:
        print(f"❌ ERROR: Failed to decode conservation model: {e}", file=sys.stderr)
        sys.exit(1)
        
    if old_model is None:
        print("\n❌ CRITICAL CRASH: No baseline model file could be found or loaded!", file=sys.stderr)
        print(f"Looked inside {processed_dir} for paths: {possible_baselines}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Execute Comparative Inference
    pred_old = old_model.predict(X_test_without)
    pred_new = new_model.predict(X_test_with)
    
    r2_old = r2_score(y_test, pred_old)
    r2_new = r2_score(y_test, pred_new)
    
    mae_old = mean_absolute_error(y_test, pred_old)
    mae_new = mean_absolute_error(y_test, pred_new)
    
    # 6. Print Production Evaluation Diagnostics
    print("\n" + "=" * 80)
    print("📊 PHYLOP CONSERVATION MODEL INFERENCE METRICS")
    print("=" * 80)
    print(f"   Baseline Model (Distance-Only) R² : {r2_old:.4f}")
    print(f"   Conservation Model (With PhyloP) R²: {r2_new:.4f}")
    print("-" * 80)
    
    r2_diff = r2_new - r2_old
    if r2_old != 0:
        pct_improvement = (r2_diff / abs(r2_old)) * 100
        print(f"   R² Absolute Improvement           : {r2_diff:+.4f} ({pct_improvement:+.1f}%)")
    else:
        print(f"   R² Absolute Improvement           : {r2_diff:+.4f}")
        
    print("-" * 80)
    print(f"   Baseline Model MAE                : {mae_old:.4f}")
    print(f"   Conservation Model MAE             : {mae_new:.4f}")
    print(f"   MAE Error Reduction Change        : {mae_new - mae_old:.4f}")
    print("=" * 80)
    print("✅ BENCHMARK RUN COMPLETE! TASK 4 FINISHED STABLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
