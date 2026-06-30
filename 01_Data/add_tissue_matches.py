"""
Add TERT and HBB matches from other tissues.
"""
import pandas as pd
from pathlib import Path

gtex_dir = Path.home() / 'gtex_analysis' / 'raw' / 'GTEx_Analysis_v11_eQTL'

# 1. Load Brain Caudate for TERT
caudate_path = gtex_dir / 'Brain_Caudate_basal_ganglia.v11.eQTLs.signif_pairs.parquet'
if caudate_path.exists():
    df = pd.read_parquet(caudate_path)
    tert = df[df['phenotype_id'].str.contains('ENSG00000164362')]
    print(f"TERT in Caudate: {len(tert)}")
    
    # Parse variant IDs
    tert['chrom'] = tert['variant_id'].str.split('_').str[0].str.replace('chr', '')
    tert['pos'] = tert['variant_id'].str.split('_').str[1]
    tert['ref'] = tert['variant_id'].str.split('_').str[2]
    tert['alt'] = tert['variant_id'].str.split('_').str[3]
    tert.to_csv('processed/tert_caudate_matches.csv', index=False)
    print("✅ Saved TERT matches")

# 2. Load Skin for HBB
skin_path = gtex_dir / 'Skin_Not_Sun_Exposed_Suprapubic.v11.eQTLs.signif_pairs.parquet'
if skin_path.exists():
    df = pd.read_parquet(skin_path)
    hbb = df[df['phenotype_id'].str.contains('ENSG00000244734')]
    print(f"HBB in Skin: {len(hbb)}")
    
    hbb['chrom'] = hbb['variant_id'].str.split('_').str[0].str.replace('chr', '')
    hbb['pos'] = hbb['variant_id'].str.split('_').str[1]
    hbb['ref'] = hbb['variant_id'].str.split('_').str[2]
    hbb['alt'] = hbb['variant_id'].str.split('_').str[3]
    hbb.to_csv('processed/hbb_skin_matches.csv', index=False)
    print("✅ Saved HBB matches")
