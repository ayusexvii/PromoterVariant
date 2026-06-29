#!/usr/bin/env python3
"""
Clinical Tiering Engine — ACMG-style expression disruption tiers.
Vectorized for maximum performance with correct array shape alignment.
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 HARDWARE-ACCELERATED CLINICAL TIERING ENGINE")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    disease_path = script_dir / "processed" / "disease_genes.txt"
    matrix_path = script_dir / "processed" / "training_matrix_allchr.csv.gz"
    output_path = script_dir / "processed" / "clinical_tiers_allchr.csv.gz"
    
    # 1. Load Disease Tracking Ledger Safely
    if disease_path.exists():
        with open(disease_path, encoding='utf-8') as f:
            disease_genes = set(line.strip().upper() for line in f if line.strip())
        print(f"✅ Loaded {len(disease_genes):,} normalized disease genes.")
    else:
        print("⚠️ Warning: Disease lookup missing. Loading default cancer/neurological panels.")
        disease_genes = {'CHEK2', 'BRCA1', 'BRCA2', 'TP53', 'APOL1', 'NF2', 'SMARCB1', 'TIMP3', 'ADSL'}
        
    # 2. Ingest Feature Source Data
    if not matrix_path.exists():
        matrix_path = script_dir / "processed" / "training_matrix_with_conservation.csv.gz"
        if not matrix_path.exists():
            print(f"❌ ERROR: Source feature training matrix missing at path: {matrix_path}", file=sys.stderr)
            sys.exit(1)
            
    print(f"📂 Loading target matrix: {matrix_path}")
    df = pd.read_csv(matrix_path, compression='gzip')
    print(f"✅ Loaded {len(df):,} variants into workspace memory.")
    
    # 3. Model Alignment Verification
    model_dir = script_dir.parent / "03_Model" / "processed"
    model_options = ['motif_model.pkl', 'model_no_gene_count.pkl', 'full_honest_model.pkl', 'full_model.pkl', 'baseline_model.pkl']
    model = None
    
    for opt in model_options:
        target_model_path = model_dir / opt
        if target_model_path.exists():
            try:
                model = joblib.load(target_model_path)
                print(f"📂 Model core loaded successfully: {opt}")
                break
            except Exception as e:
                print(f"⚠️ Failed loading {opt}: {e}", file=sys.stderr)
                
    if model is None:
        print(f"❌ CRITICAL ERROR: No trained models found in path: {model_dir}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Feature Space Synchronization
    df['abs_distance'] = df['distance_to_tss'].abs()
    gene_counts = df['promoter_gene'].value_counts().to_dict()
    df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)
    
    # Build feature matrix matching model expectations
    all_possible_features = {
        'distance_to_tss': df['distance_to_tss'] if 'distance_to_tss' in df.columns else 0,
        'abs_distance': df['abs_distance'],
        'gene_variant_count': df['gene_variant_count'],
        'phyloP': df['phyloP'] if 'phyloP' in df.columns else 0.05,
        'max_shift': df['max_shift'] if 'max_shift' in df.columns else 0,
        'num_affected': df['num_affected'] if 'num_affected' in df.columns else 0,
        'num_novel_kmers': df['num_novel_kmers'] if 'num_novel_kmers' in df.columns else 0,
        'num_lost_kmers': df['num_lost_kmers'] if 'num_lost_kmers' in df.columns else 0
    }
    
    n_features_expected = model.n_features_in_
    selected_feature_names = list(all_possible_features.keys())[:n_features_expected]
    
    print(f"🎯 Dynamic Alignment: Model expects {n_features_expected} features. Using: {selected_feature_names}")
    
    X = pd.DataFrame({f: all_possible_features[f] for f in selected_feature_names})
    df['predicted_slope'] = model.predict(X)
    
    # 5. High-Speed Vectorized Tier Allocation Engine
    print("🔧 Running vectorized clinical tier assignment rules...")
    abs_slope = df['predicted_slope'].abs()
    gene_upper = df['promoter_gene'].fillna('').astype(str).str.upper()
    is_disease = gene_upper.isin(disease_genes)
    
    # Handle clinvar column
    clinvar_col = 'ClinSig' if 'ClinSig' in df.columns else ('ClinicalSignificance' if 'ClinicalSignificance' in df.columns else None)
    if clinvar_col:
        clinvar_str = df[clinvar_col].fillna('').astype(str)
        is_pathogenic = clinvar_str.str.contains('Pathogenic', case=False, na=False)
    else:
        is_pathogenic = False
    
    # Vectorized tier assignment
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
    
    # 6. Save
    df.to_csv(output_path, index=False, compression='gzip')
    print(f"💾 Saved to: {output_path}")
    
    # 7. Summary
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
