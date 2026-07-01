"""
Add positional features to capture variant-level signal within genes.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("🧬 ADDING POSITIONAL FEATURES")
print("=" * 70)

# Load training matrix
df = pd.read_csv('processed/training_matrix_allchr.csv.gz', compression='gzip')
print(f"Loaded {len(df):,} variants")

# 1. Distance bins (categorical)
df['dist_bin'] = pd.cut(
    df['distance_to_tss'].abs(),
    bins=[0, 100, 500, 2000, 10000],
    labels=['<100', '100-500', '500-2000', '2000-10000']
)

# 2. Position within gene (normalized)
gene_min = df.groupby('promoter_gene')['distance_to_tss'].transform('min')
gene_max = df.groupby('promoter_gene')['distance_to_tss'].transform('max')
gene_range = gene_max - gene_min
df['pos_within_gene'] = (df['distance_to_tss'] - gene_min) / gene_range.replace(0, 1)

# 3. Distance to nearest exon (if we had exon data)
# For now, use distance to TSS as proxy

# 4. TSS proximity flag
df['near_tss'] = (df['distance_to_tss'].abs() < 200).astype(int)

print(f"Added features: dist_bin, pos_within_gene, near_tss")

# Save
df.to_csv('processed/training_matrix_with_positional_features.csv.gz', index=False, compression='gzip')
print("💾 Saved to processed/training_matrix_with_positional_features.csv.gz")

# Show sample
print("\n🔍 Sample:")
print(df[['promoter_gene', 'distance_to_tss', 'dist_bin', 'pos_within_gene', 'near_tss']].head(10))
