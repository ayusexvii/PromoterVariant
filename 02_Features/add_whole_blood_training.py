"""
Add Whole Blood GTEx data to the training matrix.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("🧬 ADDING WHOLE BLOOD GTEX DATA")
print("=" * 70)

gtex_dir = Path.home() / "gtex_analysis" / "raw" / "GTEx_Analysis_v11_eQTL"

# Check if Whole Blood parquet exists
wb_path = gtex_dir / "Whole_Blood.v11.eQTLs.signif_pairs.parquet"

if wb_path.exists():
    print("📂 Loading Whole Blood parquet...")
    df = pd.read_parquet(wb_path)
    print(f"✅ Loaded {len(df):,} Whole Blood eQTLs")
    
    # Parse variant IDs
    df['chrom'] = df['variant_id'].str.split('_').str[0].str.replace('chr', '')
    df['pos'] = df['variant_id'].str.split('_').str[1].astype(int)
    df['ref'] = df['variant_id'].str.split('_').str[2]
    df['alt'] = df['variant_id'].str.split('_').str[3]
    
    # Prepare for merging
    wb_clean = df[['chrom', 'pos', 'ref', 'alt', 'phenotype_id', 'slope', 'pval_nominal']].copy()
    wb_clean.columns = ['chrom', 'pos', 'ref', 'alt', 'gene_id', 'slope', 'pval']
    
    # Show sample genes
    print(f"\n📊 Sample genes in Whole Blood:")
    print(wb_clean['gene_id'].value_counts().head(10))
    
    # Check for TERT and HBB
    tert_count = wb_clean[wb_clean['gene_id'].str.contains('ENSG00000164362')].shape[0]
    hbb_count = wb_clean[wb_clean['gene_id'].str.contains('ENSG00000244734')].shape[0]
    nf2_count = wb_clean[wb_clean['gene_id'].str.contains('ENSG00000186575')].shape[0]
    apol1_count = wb_clean[wb_clean['gene_id'].str.contains('ENSG00000100342')].shape[0]
    
    print(f"\n🎯 Target genes in Whole Blood:")
    print(f"   TERT: {tert_count} entries")
    print(f"   HBB: {hbb_count} entries")
    print(f"   NF2: {nf2_count} entries")
    print(f"   APOL1: {apol1_count} entries")
    
    # Save
    wb_clean.to_csv('processed/whole_blood_training.csv.gz', index=False, compression='gzip')
    print(f"\n💾 Saved to: processed/whole_blood_training.csv.gz")
else:
    print("⚠️ Whole Blood parquet not found. Check path.")
