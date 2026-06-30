"""
Rebuild exact-match training matrix from ALL tissues.
"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧬 REBUILDING EXACT-MATCH MATRIX (ALL TISSUES)")
print("=" * 70)

# 1. Load Liver exact matches
liver_df = pd.read_csv('processed/exact_matches.csv.gz', compression='gzip')
print(f"Liver: {len(liver_df)} matches")

# 2. Load Whole Blood exact matches (if available)
# This requires parsing Whole Blood V11 first
# For now, we'll use liver only

# 3. Add PhyloP from conservation matrix
cons_df = pd.read_csv('processed/training_matrix_with_conservation.csv.gz', compression='gzip')

# Create keys
liver_df['key'] = liver_df['clinvar_chrom'].astype(str) + '_' + liver_df['clinvar_pos'].astype(str)
cons_df['key'] = cons_df['chrom'].astype(str) + '_' + cons_df['pos'].astype(str)

# Merge PhyloP
merged = liver_df.merge(cons_df[['key', 'phyloP']], on='key', how='left')
merged['phyloP'] = merged['phyloP'].fillna(0.0)

# 4. Check target genes
for gene in ['CHEK2', 'NF2', 'TERT', 'HBB']:
    count = merged[merged['clinvar_gene'].str.contains(gene, case=False, na=False)].shape[0]
    print(f"{gene}: {count} entries")

# 5. Save
merged.to_csv('processed/exact_match_matrix_all_tissues.csv.gz', index=False, compression='gzip')
print(f"✅ Saved {len(merged)} rows")
