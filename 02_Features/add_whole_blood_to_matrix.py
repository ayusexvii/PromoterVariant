"""
Add Whole Blood GTEx data to the training matrix.
"""
import pandas as pd
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
    
    # Prepare for merging with existing matrix
    wb_clean = df[['chrom', 'pos', 'ref', 'alt', 'phenotype_id', 'slope', 'pval_nominal']].copy()
    wb_clean.columns = ['chrom', 'pos', 'ref', 'alt', 'gene_id', 'slope', 'pval']
    
    print(f"✅ Parsed {len(wb_clean)} Whole Blood records")
    wb_clean.to_csv('processed/whole_blood_eqtl_clean.csv.gz', index=False, compression='gzip')
    print("💾 Saved to processed/whole_blood_eqtl_clean.csv.gz")
else:
    print("⚠️ Whole Blood parquet not found. Skip for now.")
