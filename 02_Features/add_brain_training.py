"""
Add Brain GTEx data to the training matrix.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("🧬 ADDING BRAIN GTEX DATA")
print("=" * 70)

gtex_dir = Path.home() / "gtex_analysis" / "raw" / "GTEx_Analysis_v11_eQTL"

# Find Brain tissues - use multiple for diversity
brain_paths = list(gtex_dir.glob("Brain_*.v11.eQTLs.signif_pairs.parquet"))

if brain_paths:
    print(f"Found {len(brain_paths)} Brain tissues")
    
    all_brain_dfs = []
    for path in brain_paths[:3]:  # Use first 3 brain tissues
        print(f"\n📂 Loading: {path.name}")
        df = pd.read_parquet(path)
        print(f"   Loaded {len(df):,} records")
        
        # Parse variant IDs
        df['chrom'] = df['variant_id'].str.split('_').str[0].str.replace('chr', '')
        df['pos'] = df['variant_id'].str.split('_').str[1].astype(int)
        df['ref'] = df['variant_id'].str.split('_').str[2]
        df['alt'] = df['variant_id'].str.split('_').str[3]
        
        # Clean
        brain_clean = df[['chrom', 'pos', 'ref', 'alt', 'phenotype_id', 'slope', 'pval_nominal']].copy()
        brain_clean.columns = ['chrom', 'pos', 'ref', 'alt', 'gene_id', 'slope', 'pval']
        
        # Check for target genes
        tert_count = brain_clean[brain_clean['gene_id'].str.contains('ENSG00000164362')].shape[0]
        hbb_count = brain_clean[brain_clean['gene_id'].str.contains('ENSG00000244734')].shape[0]
        nf2_count = brain_clean[brain_clean['gene_id'].str.contains('ENSG00000186575')].shape[0]
        apol1_count = brain_clean[brain_clean['gene_id'].str.contains('ENSG00000100342')].shape[0]
        
        print(f"\n🎯 Target genes in {path.name}:")
        print(f"   TERT: {tert_count} entries")
        print(f"   HBB: {hbb_count} entries")
        print(f"   NF2: {nf2_count} entries")
        print(f"   APOL1: {apol1_count} entries")
        
        # Save individually
        tissue_name = path.name.split('.')[0]
        brain_clean.to_csv(f'processed/{tissue_name}_training.csv.gz', index=False, compression='gzip')
        print(f"   💾 Saved to: processed/{tissue_name}_training.csv.gz")
        
        all_brain_dfs.append(brain_clean)
    
    # Combine all brain tissues
    if all_brain_dfs:
        combined_brain = pd.concat(all_brain_dfs, ignore_index=True)
        combined_brain.to_csv('processed/brain_combined_training.csv.gz', index=False, compression='gzip')
        print(f"\n💾 Combined brain saved: {len(combined_brain):,} rows")
        
        # Final target gene summary
        print("\n📊 Target genes in combined brain:")
        for gene in ['TERT', 'HBB', 'NF2', 'APOL1']:
            count = combined_brain[combined_brain['gene_id'].str.contains(f'ENSG00000164362|ENSG00000244734|ENSG00000186575|ENSG00000100342')].shape[0]
            print(f"   {gene}: {count} entries")
else:
    print("⚠️ No Brain tissues found.")
