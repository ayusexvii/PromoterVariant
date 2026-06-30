#!/usr/bin/env python3
"""
Clinical Tiering Engine v2.0.2 — Triple-Lock Tiering (FIXED)
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 CLINICAL TIERING ENGINE v2.0.2 (TRIPLE-LOCK CORE - FIXED)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    disease_path = script_dir / "processed" / "disease_genes_fixed.txt"
    input_path = script_dir / "processed" / "training_matrix_allchr.csv.gz"
    output_path = script_dir / "processed" / "clinical_tiers_v202.csv.gz"
    
    # 1. Load normalized disease genes list
    if not disease_path.exists():
        print(f"❌ ERROR: Disease genes tracking list missing at: {disease_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(disease_path, encoding='utf-8') as f:
        disease_genes = set(line.strip().upper() for line in f if line.strip())
    print(f"✅ Loaded {len(disease_genes):,} normalized disease genes.")
    
    # 2. Load honest model framework
    model_path = script_dir / "../03_Model/processed/full_honest_model.pkl"
    if not model_path.exists():
        model_path = script_dir / "../03_Model/processed/full_model.pkl"
        if not model_path.exists():
            print("❌ ERROR: Honest predictive model weights missing from workspace.", file=sys.stderr)
            sys.exit(1)
            
    print(f"📂 Loading model: {model_path.name}")
    model = joblib.load(model_path)
    expected_features_count = model.n_features_in_
    print(f"   Model verification -> Expects exactly {expected_features_count} inputs.")
    
    # 3. Load variant coordinate dataset
    if not input_path.exists():
        input_path = script_dir / "processed" / "training_matrix_with_conservation.csv.gz"
        if not input_path.exists():
            print("❌ ERROR: Training matrix not found!", file=sys.stderr)
            sys.exit(1)
            
    print(f"📂 Loading training matrix: {input_path.name}")
    df = pd.read_csv(input_path, compression='gzip')
    print(f"✅ Ingested {len(df):,} functional variants.")
    
    # 4. Feature Space Allocation
    df['abs_distance'] = df['distance_to_tss'].abs()
    gene_counts = df['promoter_gene'].value_counts().to_dict()
    df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)
    
    all_features_pool = {
        'distance_to_tss': df['distance_to_tss'],
        'abs_distance': df['abs_distance'],
        'gene_variant_count': df['gene_variant_count']
    }
    
    selected_cols = list(all_features_pool.keys())[:expected_features_count]
    X_honest = pd.DataFrame({col: all_features_pool[col] for col in selected_cols})
    
    df['predicted_slope'] = model.predict(X_honest)
    df['abs_slope'] = df['predicted_slope'].abs()
    print(f"✅ Predictions generated across validation matrices.")
    
    # 5. TRIPLE-LOCK CLASSIFICATION
    clinvar_col = 'ClinSig' if 'ClinSig' in df.columns else ('ClinicalSignificance' if 'ClinicalSignificance' in df.columns else None)
    if clinvar_col:
        df['is_pathogenic'] = df[clinvar_col].fillna('').astype(str).str.contains(
            'Pathogenic|Likely pathogenic|Likely_pathogenic', case=False, regex=True
        )
    else:
        df['is_pathogenic'] = False
        
    df['gene_upper'] = df['promoter_gene'].fillna('').astype(str).str.upper().str.strip()
    df['is_disease_gene'] = df['gene_upper'].isin(disease_genes)
    
    # FIXED: Clean conditional logic
    conditions = [
        # Tier 1 Actionable: ALL THREE locks are met
        (df['abs_slope'] > 1.0) & df['is_pathogenic'] & df['is_disease_gene'],
        
        # Tier 1 High-Impact: Extreme outlier slope (>1.0)
        (df['abs_slope'] > 1.0),
        
        # Tier 2 Moderate: |slope| > 0.75 OR (|slope| > 0.50 AND (pathogenic OR disease gene))
        (df['abs_slope'] > 0.75) | ((df['abs_slope'] > 0.50) & (df['is_pathogenic'] | df['is_disease_gene'])),
        
        # Tier 3 Benign: Flat minimal expression alterations
        (df['abs_slope'] <= 0.50)
    ]
    
    tier_labels = [
        'Tier 1: Actionable High-Impact',
        'Tier 1: High-Impact',
        'Tier 2: Moderate Impact',
        'Tier 3: Regulatory Benign'
    ]
    
    action_labels = [
        'Flag for precision oncology/therapeutic targeting. Functional validation required.',
        'Monitor for emerging disease associations. Consider research biobank enrollment.',
        'Variant of regulatory significance. Recommend periodic re-evaluation.',
        'Low predicted expression impact. Standard clinical follow-up.'
    ]
    
    df['tier'] = np.select(conditions, tier_labels, default='Tier 3: Regulatory Benign')
    df['action'] = np.select(conditions, action_labels, default='Low predicted expression impact. Standard clinical follow-up.')
    
    # 6. Compress and Export Results
    df.to_csv(output_path, index=False, compression='gzip')
    print(f"💾 Saved updated clinical annotation matrix to: {output_path}")
    
    # 7. Summary
    print("\n" + "=" * 75)
    print("📊 TIER DISTRIBUTION SUMMARY (v2.0.2 FIXED)")
    print("=" * 75)
    tier_counts = df['tier'].value_counts()
    total = len(df)
    
    for tier in ['Tier 1: Actionable High-Impact', 'Tier 1: High-Impact', 'Tier 2: Moderate Impact', 'Tier 3: Regulatory Benign']:
        count = tier_counts.get(tier, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"   {tier:<35}: {count:,} ({pct:.2f}%)")
        
    # 8. Portfolio Validation
    print("\n" + "=" * 75)
    print("🧬 GENOMIC PORTFOLIO VALIDATION (CHEK2 TARGET TRACK)")
    print("=" * 75)
    chek2_df = df[df['gene_upper'] == 'CHEK2']
    print(f"   Total CHEK2 Variants Located      : {len(chek2_df):,}")
    print(f"   CHEK2 Actionable Tier-1 Mutations : {len(chek2_df[chek2_df['tier'] == 'Tier 1: Actionable High-Impact']):,}")
    
    if not chek2_df.empty:
        print("   CHEK2 Internal Tier Distribution Profile:")
        for tier, count in chek2_df['tier'].value_counts().items():
            print(f"      - {tier:<30}: {count:,}")
    print("=" * 75)
    print("✅ PROCESS MATRIX SEAMLESSLY UPGRADED! TASK COMPLETE.")
    print("=" * 75)

if __name__ == "__main__":
    main()
