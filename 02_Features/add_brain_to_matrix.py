"""
Add Brain GTEx data to the training matrix.
"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧬 ADDING BRAIN GTEX DATA")
print("=" * 70)

gtex_dir = Path.home() / "gtex_analysis" / "raw" / "GTEx_Analysis_v11_eQTL"

# Check if Brain parquet exists
brain_paths = list(gtex_dir.glob("Brain_*.v11.eQTLs.signif_pairs.parquet"))

if brain_paths:
    for path in brain_paths[:1]:  # Take first brain tissue
        print(f"📂 Loading: {path.name}")
        df = pd.read_parquet(path)
        print(f"   Loaded {len(df):,} records")
        
        df['chrom'] = df['variant_id'].str.split('_').str[0].str.replace('chr', '')
        df['pos'] = df['variant_id'].str.split('_').str[1].astype(int)
        df['ref'] = df['variant_id'].str.split('_').str[2]
        df['alt'] = df['variant_id'].str.split('_').str[3]
        
        brain_clean = df[['chrom', 'pos', 'ref', 'alt', 'phenotype_id', 'slope', 'pval_nominal']].copy()
        brain_clean.columns = ['chrom', 'pos', 'ref', 'alt', 'gene_id', 'slope', 'pval']
        
        tissue_name = path.name.split('.')[0]
        brain_clean.to_csv(f'processed/{tissue_name}_clean.csv.gz', index=False, compression='gzip')
        print(f"💾 Saved to processed/{tissue_name}_clean.csv.gz")
else:
    print("⚠️ No Brain parquet files found.")
