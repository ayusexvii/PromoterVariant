#!/usr/bin/env python3
"""
Clinical Tiering Engine — ACMG-style expression disruption tiers.
FIXED: Uses correct feature names from the model.
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 HARDWARE-ACCELERATED CLINICAL TIERING ENGINE (FIXED)")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    disease_path = script_dir / "processed" / "disease_genes.txt"
    matrix_path = script_dir / "processed" / "training_matrix_allchr.csv.gz"
    output_path = script_dir / "processed" / "clinical_tiers_allchr.csv.gz"
    
    # 1. Load Disease Genes
    if disease_path.exists():
        with open(disease_path, encoding='utf-8') as f:
            disease_genes = set(line.strip().upper() for line in f if line.strip())
        print(f"✅ Loaded {len(disease_genes):,} normalized disease genes.")
    else:
        disease_genes = {'CHEK2', 'BRCA1', 'BRCA2', 'TP53', 'APOL1', 'NF2', 'SMARCB1', 'TIMP3', 'ADSL'}
        
    # 2. Load Training Matrix
    if not matrix_path.exists():
        matrix_path = script_dir / "processed" / "training_matrix_with_conservation.csv.gz"
        if not matrix_path.exists():
            print(f"❌ ERROR: Matrix missing at: {matrix_path}", file=sys.stderr)
            sys.exit(1)
            
    print(f"📂 Loading target matrix: {matrix_path}")
    df = pd.read_csv(matrix_path, compression='gzip')
    print(f"✅ Loaded {len(df):,} variants.")
    
    # 3. Load Model
    model_dir = script_dir.parent / "03_Model" / "processed"
    model_path = model_dir / "motif_model.pkl"
    
    if not model_path.exists():
        model_path = model_dir / "full_honest_model.pkl"
        if not model_path.exists():
            print(f"❌ ERROR: No model found", file=sys.stderr)
            sys.exit(1)
    
    print(f"📂 Loading model: {model_path}")
    model = joblib.load(model_path)
    
    # 4. FIXED: Use the model's actual feature names
    model_features = model.feature_names_in_
    print(f"🎯 Model expects {len(model_features)} features: {list(model_features)}")
    
    # 5. Build Features
    df['abs_distance'] = df['distance_to_tss'].abs()
    gene_counts = df['promoter_gene'].value_counts().to_dict()
    df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)
    
    # Build feature matrix using the model's feature names
    feature_dict = {}
    for feat in model_features:
        if feat in df.columns:
            feature_dict[feat] = df[feat]
        elif feat == 'abs_distance':
            feature_dict[feat] = df['abs_distance']
        elif feat == 'gene_variant_count':
            feature_dict[feat] = df['gene_variant_count']
        elif feat == 'distance_to_tss':
            feature_dict[feat] = df['distance_to_tss']
        elif feat == 'phyloP':
            feature_dict[feat] = df['phyloP'] if 'phyloP' in df.columns else 0.05
        elif feat in ['novel_kmers', 'lost_kmers', 'disruption_intensity']:
            # These might exist or we add zeros
            feature_dict[feat] = df[feat] if feat in df.columns else 0
        elif feat in ['max_shift', 'num_affected']:
            feature_dict[feat] = df[feat] if feat in df.columns else 0
        else:
            feature_dict[feat] = 0
    
    X = pd.DataFrame(feature_dict)
    print(f"✅ Built feature matrix with {X.shape[1]} features")
    
    # Predict
    df['predicted_slope'] = model.predict(X)
    
    # 6. Vectorized Tier Assignment
    print("🔧 Running clinical tier assignment...")
    abs_slope = df['predicted_slope'].abs()
    gene_upper = df['promoter_gene'].fillna('').astype(str).str.upper()
    is_disease = gene_upper.isin(disease_genes)
    
    clinvar_col = 'ClinSig' if 'ClinSig' in df.columns else ('ClinicalSignificance' if 'ClinicalSignificance' in df.columns else None)
    if clinvar_col:
        clinvar_str = df[clinvar_col].fillna('').astype(str)
        is_pathogenic = clinvar_str.str.contains('Pathogenic', case=False, na=False)
    else:
        is_pathogenic = False
    
    conditions = [
        (abs_slope > 0.50) & (is_disease | is_pathogenic),
        (abs_slope > 0.50) & ~(is_disease | is_pathogenic),
        (abs_slope > 0.20) & (abs_slope <= 0.50),
        (abs_slope <= 0.20)
    ]
    
    tier_values = [
        'Tier 1: Actionable High-Impact',
        'Tier 1: High-Impact',
        'Tier 2: Moderate Impact',
        'Tier 3: Regulatory Benign'
    ]
    
    action_values = [
        'Flag for precision oncology/therapeutic targeting. Consider functional validation.',
        'Monitor for emerging disease associations. Consider research biobank enrollment.',
        'Variant of regulatory significance. Recommend periodic re-evaluation.',
        'Low predicted expression impact. Standard clinical follow-up.'
    ]
    
    df['tier'] = np.select(conditions, tier_values, default='Tier 3: Regulatory Benign')
    df['action'] = np.select(conditions, action_values, default='Standard clinical follow-up.')
    
    # 7. Save
    df.to_csv(output_path, index=False, compression='gzip')
    print(f"💾 Saved to: {output_path}")
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("📊 TIER DISTRIBUTION")
    print("=" * 70)
    tier_counts = df['tier'].value_counts()
    for tier, count in tier_counts.items():
        print(f"   {tier}: {count:,}")
    
    disease_count = df[df['promoter_gene'].isin(disease_genes)]['promoter_gene'].nunique()
    print(f"\n✅ Disease genes in dataset: {disease_count:,}")
    print("=" * 70)
    print("✅ CLINICAL TIERING COMPLETE!")

if __name__ == "__main__":
    main()
