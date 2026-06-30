#!/usr/bin/env python3
"""
Build unified exact-match matrix - FINAL VERSION.
Preserves clinvar_gene through the merge process.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 UNIFIED MATRIX COMPILATION (FINAL - PRESERVES GENE LABELS)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Liver exact matches
    liver_path = output_dir / "exact_matches.csv.gz"
    if not liver_path.exists():
        print("❌ ERROR: Liver matches missing!", file=sys.stderr)
        sys.exit(1)
    
    print("📂 Loading Liver matches...")
    liver_df = pd.read_csv(liver_path, compression='gzip')
    print(f"   Liver: {len(liver_df)} rows")
    
    # 2. Load TERT from Brain Caudate
    tert_path = script_dir / "../01_Data/processed/tert_caudate_matches.csv"
    tert_df = pd.DataFrame()
    if tert_path.exists():
        print("📂 Loading TERT from Brain Caudate...")
        tert_df = pd.read_csv(tert_path)
        tert_df['clinvar_gene'] = 'TERT'
        tert_df['clinvar_sig'] = 'Validated'
        # Rename columns to match liver format
        tert_df['gtex_slope'] = tert_df['slope']
        tert_df['gtex_pval'] = tert_df['pval_nominal']
        tert_df['gtex_gene'] = tert_df['phenotype_id']
        print(f"   TERT: {len(tert_df)} rows")
    
    # 3. Load HBB from Skin
    hbb_path = script_dir / "../01_Data/processed/hbb_skin_matches.csv"
    hbb_df = pd.DataFrame()
    if hbb_path.exists():
        print("📂 Loading HBB from Skin...")
        hbb_df = pd.read_csv(hbb_path)
        hbb_df['clinvar_gene'] = 'HBB'
        hbb_df['clinvar_sig'] = 'Validated'
        hbb_df['gtex_slope'] = hbb_df['slope']
        hbb_df['gtex_pval'] = hbb_df['pval_nominal']
        hbb_df['gtex_gene'] = hbb_df['phenotype_id']
        print(f"   HBB: {len(hbb_df)} rows")
    
    # 4. Combine ALL with preserved gene labels
    print("\n🔗 Combining all matches...")
    
    # Ensure column alignment
    for df in [liver_df, tert_df, hbb_df]:
        if 'clinvar_gene' not in df.columns:
            df['clinvar_gene'] = 'UNKNOWN'
        if 'clinvar_sig' not in df.columns:
            df['clinvar_sig'] = 'NA'
        if 'gtex_slope' not in df.columns:
            df['gtex_slope'] = 0.0
        if 'gtex_pval' not in df.columns:
            df['gtex_pval'] = 1.0
        if 'gtex_gene' not in df.columns:
            df['gtex_gene'] = 'NA'
        if 'clinvar_ref' not in df.columns:
            df['clinvar_ref'] = 'NA'
        if 'clinvar_alt' not in df.columns:
            df['clinvar_alt'] = 'NA'
    
    combined = pd.concat([liver_df, tert_df, hbb_df], ignore_index=True)
    print(f"   Combined: {len(combined)} rows")
    
    # 5. Load promoter features
    print("\n📂 Loading promoter features...")
    prom_df = pd.read_csv(output_dir / "promoter_variants_allchr.txt.gz", sep='\t', compression='gzip')
    prom_df['chrom'] = prom_df['chrom'].astype(str).str.replace('chr', '', case=False)
    prom_df['pos'] = prom_df['pos'].astype(int)
    prom_lookup = prom_df[['chrom', 'pos', 'distance_to_tss', 'promoter_gene']].drop_duplicates(subset=['chrom', 'pos'])
    
    # 6. Merge features (PRESERVE clinvar_gene)
    combined['chrom'] = combined['clinvar_chrom'].astype(str).str.replace('chr', '', case=False)
    combined['pos'] = pd.to_numeric(combined['clinvar_pos'], errors='coerce').fillna(0).astype(int)
    
    merged = combined.merge(prom_lookup, on=['chrom', 'pos'], how='left')
    
    # 7. Add features
    merged['abs_distance'] = merged['distance_to_tss'].abs()
    gene_counts = merged['promoter_gene'].value_counts().to_dict()
    merged['gene_variant_count'] = merged['promoter_gene'].map(gene_counts)
    
    # 8. Add PhyloP
    cons_df = pd.read_csv(output_dir / "training_matrix_with_conservation.csv.gz", compression='gzip')
    cons_df['chrom'] = cons_df['chrom'].astype(str).str.replace('chr', '', case=False)
    cons_df['pos'] = cons_df['pos'].astype(int)
    cons_lookup = cons_df[['chrom', 'pos', 'phyloP']].drop_duplicates(subset=['chrom', 'pos'])
    merged = merged.merge(cons_lookup, on=['chrom', 'pos'], how='left')
    merged['phyloP'] = merged['phyloP'].fillna(0.0)
    
    # 9. Build final training matrix
    feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP']
    metadata_cols = ['chrom', 'pos', 'clinvar_gene', 'clinvar_sig', 'gtex_gene']
    
    train_df = merged[metadata_cols + ['gtex_slope'] + feature_cols].copy()
    train_df.rename(columns={'gtex_slope': 'eqtl_slope'}, inplace=True)
    
    # 10. Clean
    initial = len(train_df)
    train_df = train_df.dropna(subset=feature_cols + ['eqtl_slope'])
    print(f"🧹 Removed {initial - len(train_df)} incomplete rows")
    
    # 11. Target gene summary
    print("\n📊 Target Genes Summary:")
    for gene in ['CHEK2', 'NF2', 'TERT', 'HBB']:
        count = train_df[train_df['clinvar_gene'].fillna('').astype(str).str.contains(gene, case=False, regex=False)].shape[0]
        print(f"   {gene:<10}: {count} entries")
    
    # 12. Save
    output_path = output_dir / "unified_exact_match_matrix_final.csv.gz"
    train_df.to_csv(output_path, index=False, compression='gzip')
    print(f"\n💾 Saved: {output_path}")
    print(f"   Rows: {len(train_df)}, Columns: {len(train_df.columns)}")
    print("=" * 75)
    print("✅ COMPLETE!")
    print("=" * 75)

if __name__ == "__main__":
    main()
