import pandas as pd
from pathlib import Path

# Load training matrix
df = pd.read_csv('processed/exact_match_training_matrix.csv.gz', compression='gzip')

# Load PhyloP from conservation matrix
cons_df = pd.read_csv('processed/training_matrix_with_conservation.csv.gz', compression='gzip')

# Create merge keys
df['key'] = df['clinvar_chrom'].astype(str) + '_' + df['clinvar_pos'].astype(str)
cons_df['key'] = cons_df['chrom'].astype(str) + '_' + cons_df['pos'].astype(str)

# Merge PhyloP
merged = df.merge(cons_df[['key', 'phyloP']], on='key', how='left')

# Update PhyloP
if 'phyloP_x' in merged.columns:
    merged['phyloP'] = merged['phyloP_y'].fillna(0.0)
    merged = merged.drop(columns=['phyloP_x', 'phyloP_y'], errors='ignore')

# Save
merged.to_csv('processed/exact_match_training_matrix_fixed.csv.gz', index=False, compression='gzip')
print(f"✅ Fixed matrix saved with {len(merged)} rows")
print(f"   PhyloP non-zero: {(merged['phyloP'] != 0).sum()}")
