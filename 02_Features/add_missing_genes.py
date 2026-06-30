import pandas as pd
from pathlib import Path

# Load existing training matrix
df = pd.read_csv('processed/exact_match_training_matrix.csv.gz', compression='gzip')

# Check missing genes
missing_genes = ['TERT', 'NF2', 'HBB']
print(f"Missing genes: {missing_genes}")

# Load Whole Blood exact matches (if available)
# This requires parsing Whole Blood V11 first

# For now, just report
print(f"Current genes: {df['promoter_gene'].unique()[:20]}")
