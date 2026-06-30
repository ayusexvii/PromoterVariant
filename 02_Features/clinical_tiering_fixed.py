#!/usr/bin/env python3
"""
Clinical Tiering Engine - FIXED & OPTIMIZED
Uses vectorized numpy operations for ultra-fast processing speeds.
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 ACCELERATED CLINICAL TIERING ENGINE (HONEST 3-FEATURE MODEL)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    disease_path = script_dir / "processed" / "disease_genes_fixed.txt"
    input_matrix_path = script_dir / "processed" / "training_matrix_allchr.csv.gz"
    output_path = script_dir / "processed" / "clinical_tiers_fixed.csv.gz"
    
    # 1. Load Disease Tracking Ledger Safely
    if not disease_path.exists():
        print(f"❌ ERROR: Normalized disease lookup missing at: {disease_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(disease_path, encoding='utf-8') as f:
        disease_genes = set(line.strip().upper() for line in f if line.strip())
    print(f"✅ Loaded {len(disease_genes):,} normalized disease genes.")
    
    # 2. Ingest Honest Predictive Model Core
    model_path = script_dir / "../03_Model/processed/full_honest_model.pkl"
    if not model_path.exists():
        model_path = script_dir / "../03_Model/processed/full_model.pkl"
        if not model_path.exists():
            print("❌ ERROR: Predictive machine learning models absent from workspace path.", file=sys.stderr)
            sys.exit(1)
            
    print(f"📂 Connecting to model archive: {model_path.name}")
    model = joblib.load(model_path)
    expected_features_count = model.n_features_in_
    print(f"   Model verification -> Evaluates exactly {expected_features_count} inputs.")
    
    # 3. Stream and Validate Input Genomic Coordinates Matrix
    if not input_matrix_path.exists():
        input_matrix_path = script_dir / "processed" / "training_matrix_with_conservation.csv.gz"
        if not input_matrix_path.exists():
            print(f"❌ ERROR: Variant sample matrix missing at: {input_matrix_path}", file=sys.stderr)
            sys.exit(1)
            
    print(f"📂 Unpacking variant tracking matrix: {input_matrix_path.name}")
    df = pd.read_csv(input_matrix_path, compression='gzip')
    print(f"✅ Ingested {len(df):,} structural regulatory variants.")
    
    # 4. Feature Synchronization Engineering Pass
    df['abs_distance'] = df['distance_to_tss'].abs()
    gene_counts = df['promoter_gene'].value_counts().to_dict()
    df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)
    
    # Build a stable feature frame strictly aligned with the model's structural parameters
    all_features_pool = {
        'distance_to_tss': df['distance_to_tss'],
        'abs_distance': df['abs_distance'],
        'gene_variant_count': df['gene_variant_count']
    }
    
    selected_cols = list(all_features_pool.keys())[:expected_features_count]
    X_honest = pd.DataFrame({col: all_features_pool[col] for col in selected_cols})
    
    print(f"🚀 Feeding dimensional layout matrix ({X_honest.shape[1]} features) into the estimator...")
    df['predicted_slope'] = model.predict(X_honest)
    
    # 5. High-Velocity Vectorized Clinical Tier Allocator (Replaces apply loops)
    print("🔧 Running vectorized variant diagnostic rules classification...")
    abs_slope = df['predicted_slope'].abs()
    gene_normalized = df['promoter_gene'].fillna('').astype(str).str.upper().str.strip()
    is_disease = gene_normalized.isin(disease_genes)
    
    # Extract structural ClinVar pathogenicity attributes case-insensitively
    clinvar_col = 'ClinSig' if 'ClinSig' in df.columns else ('ClinicalSignificance' if 'ClinicalSignificance' in df.columns else None)
    if clinvar_col:
        clinvar_clean_str = df[clinvar_col].fillna('').astype(str).str.lower()
        is_pathogenic = clinvar_clean_str.str.contains('pathogenic')
    else:
        is_pathogenic = pd.Series(False, index=df.index)
        
    # Set priority tier constraints using unified arrays
    conditions = [
        (abs_slope > 0.50) & (is_disease | is_pathogenic),
        (abs_slope > 0.50),
        (abs_slope > 0.20),
        (abs_slope <= 0.20)
    ]
    
    tier_labels = [
        'Tier 1: Actionable High-Impact',
        'Tier 1: High-Impact',
        'Tier 2: Moderate Impact',
        'Tier 3: Regulatory Benign'
    ]
    
    action_labels = [
        'Flag for precision oncology/therapeutic targeting. Consider functional validation.',
        'Monitor for emerging disease associations. Consider research biobank enrollment.',
        'Variant of regulatory significance. Recommend periodic re-evaluation.',
        'Low predicted expression impact. Standard clinical follow-up.'
    ]
    
    df['tier'] = np.select(conditions, tier_labels, default='Tier 3: Regulatory Benign')
    df['action'] = np.select(conditions, action_labels, default='Low predicted expression impact. Standard clinical follow-up.')
    
    # 6. Compress and Export Results
    print(f"📝 Saving updated annotations to: {output_path.name}")
    df.to_csv(output_path, index=False, compression='gzip')
    
    # Output Quality Metric Summaries
    print("\n" + "=" * 75)
    print("📊 RE-CALIBRATED TIER DISTRIBUTION METRICS")
    print("=" * 75)
    counts = df['tier'].value_counts()
    for tier, count in counts.items():
        print(f"   {tier:<32}: {count:,}")
        
    print("\n" + "=" * 75)
    print("🧬 PORTFOLIO CHROMOSOME VARIANT TEST RUN (CHEK2 GENE VALIDATION)")
    print("=" * 75)
    chek2_mask = df['promoter_gene'].fillna('').astype(str).str.upper().str.strip() == 'CHEK2'
    chek2_df = df[chek2_mask]
    
    if not chek2_df.empty:
        print(f"   Total CHEK2 Mutation Rows Located  : {len(chek2_df):,}")
        print(f"   CHEK2 Pathogenic Gene Association  : {'YES (Verified Match)' if 'CHEK2' in disease_genes else 'NO'}")
        print("   CHEK2 Internal Tier Distribution Profile:")
        for tier, count in chek2_df['tier'].value_counts().items():
            print(f"      - {tier:<30}: {count:,}")
    else:
        print("   ⚠️ Check Warning: No explicit CHEK2 records found matching input parameters.")
    print("=" * 75)
    print("✅ PIPELINE EXECUTION SUCCESSFUL! TASKS COMPLETE.")

if __name__ == "__main__":
    main()
