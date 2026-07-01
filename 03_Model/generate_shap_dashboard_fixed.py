#!/usr/bin/env python3
"""
Generate SHAP-like feature importance for dashboard display.
USES GENE-FALLBACK MATRIX (31,877 rows).
"""
import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.inspection import permutation_importance
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 DASHBOARD FEATURE IMPORTANCE ENGINE (GENE-FALLBACK)")
    print("=" * 75)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Model
    model_path = output_dir / "full_honest_model.pkl"
    if not model_path.exists():
        model_path = output_dir / "full_model.pkl"
        if not model_path.exists():
            print("❌ Model not found!", file=sys.stderr)
            sys.exit(1)
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path.name}")

    # 2. Force use gene-fallback matrix
    matrix_path = script_dir / "../02_Features/processed/training_matrix_allchr.csv.gz"
    if not matrix_path.exists():
        print("❌ Gene-fallback matrix not found!", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Loading gene-fallback matrix: {matrix_path.name}")
    df = pd.read_csv(matrix_path, compression='gzip')
    print(f"✅ Loaded {len(df):,} rows")

    # 3. On-the-Fly Feature Re-Engineering
    if 'abs_distance' not in df.columns and 'distance_to_tss' in df.columns:
        df['abs_distance'] = df['distance_to_tss'].abs()
        
    if 'gene_variant_count' not in df.columns:
        gene_col = 'promoter_gene' if 'promoter_gene' in df.columns else None
        if gene_col:
            df['gene_variant_count'] = df[gene_col].map(df[gene_col].value_counts())
        else:
            df['gene_variant_count'] = 1

    if 'phyloP' not in df.columns:
        df['phyloP'] = 0.0

    # 4. Target column
    target_col = 'eqtl_slope'
    if target_col not in df.columns:
        print("❌ Target column not found!", file=sys.stderr)
        sys.exit(1)

    # 5. Features for the honest model (3 features)
    feature_names = ['distance_to_tss', 'abs_distance', 'gene_variant_count']
    
    # 6. Clean and sample
    clean_df = df.dropna(subset=feature_names + [target_col])
    sample_size = min(5000, len(clean_df))
    sample_df = clean_df.sample(n=sample_size, random_state=42)
    X_sample = sample_df[feature_names].copy()
    y_sample = sample_df[target_col].values

    print(f"🚀 Extracted {len(X_sample):,} clean samples")

    # 7. Permutation importance
    perm_importance = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    # 8. Create dataframe
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    print("\n📊 FEATURE RANKINGS (Gene-Fallback Matrix):")
    print("-" * 60)
    for row in imp_df.itertuples():
        print(f"   {row.feature:<25}: {row.importance:.4f} (± {row.std_dev:.4f})")
    print("-" * 60)

    # 9. Save outputs
    out_path_raw = output_dir / "shap_feature_importance.csv"
    out_path_dash = output_dir / "shap_dashboard_data.csv"
    
    imp_df.to_csv(out_path_raw, index=False)
    shap_df = imp_df.rename(columns={'feature': 'Feature'})
    shap_df.to_csv(out_path_dash, index=False)
    
    print(f"💾 Saved to:\n   - {out_path_raw}\n   - {out_path_dash}")
    print("=" * 75)
    print("✅ COMPLETE!")

if __name__ == "__main__":
    main()
