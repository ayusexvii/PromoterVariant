"""
Fix PhyloP in exact-match matrix by merging from conservation matrix.
FIXED: Creates phyloP column from merge instead of accessing it directly.
"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧬 FIXING PHYLOP IN EXACT-MATCH MATRIX (FIXED)")
print("=" * 70)

# Load matrices
exact_df = pd.read_csv('processed/unified_matrix_all_genes.csv.gz', compression='gzip')
cons_df = pd.read_csv('processed/training_matrix_with_conservation.csv.gz', compression='gzip')

print(f"Exact matrix: {len(exact_df)} rows")
print(f"Conservation matrix: {len(cons_df)} rows")

# Check if phyloP exists in cons_df
if 'phyloP' not in cons_df.columns:
    print("⚠️ phyloP not found in conservation matrix!")
    print(f"   Conservation columns: {list(cons_df.columns)}")
    # Try to find phyloP by case-insensitive search
    phylo_cols = [c for c in cons_df.columns if 'phylo' in c.lower()]
    if phylo_cols:
        print(f"   Found possible phylo columns: {phylo_cols}")
        cons_df = cons_df.rename(columns={phylo_cols[0]: 'phyloP'})
    else:
        print("❌ No phyloP column found! Setting all to 0.0")
        exact_df['phyloP'] = 0.0
        exact_df.to_csv('processed/unified_matrix_phylop_fixed.csv.gz', index=False, compression='gzip')
        print("\n✅ Saved with phyloP = 0.0")
        exit(0)

# Standardize chromosome format
exact_df['chrom_clean'] = exact_df['chrom'].astype(str).str.replace('chr', '', case=False)
cons_df['chrom_clean'] = cons_df['chrom'].astype(str).str.replace('chr', '', case=False)

# Merge PhyloP
exact_df['key'] = exact_df['chrom_clean'].astype(str) + '_' + exact_df['pos'].astype(str)
cons_df['key'] = cons_df['chrom_clean'].astype(str) + '_' + cons_df['pos'].astype(str)

# Create lookup
cons_lookup = cons_df[['key', 'phyloP']].drop_duplicates(subset=['key'])

# Merge - this creates the phyloP column
exact_df = exact_df.merge(cons_lookup, on='key', how='left')

# Fill missing and rename if needed
if 'phyloP_y' in exact_df.columns:
    exact_df['phyloP'] = exact_df['phyloP_y'].fillna(0.0)
    exact_df = exact_df.drop(columns=['phyloP_y'], errors='ignore')
elif 'phyloP_x' in exact_df.columns:
    exact_df['phyloP'] = exact_df['phyloP_x'].fillna(0.0)
    exact_df = exact_df.drop(columns=['phyloP_x'], errors='ignore')
else:
    # If phyloP column already exists, just fill
    if 'phyloP' in exact_df.columns:
        exact_df['phyloP'] = exact_df['phyloP'].fillna(0.0)
    else:
        print("❌ Could not create phyloP column!")
        exit(1)

# Check PhyloP values
print(f"\nPhyloP non-zero: {(exact_df['phyloP'] != 0).sum()}")
print(f"PhyloP mean: {exact_df['phyloP'].mean():.4f}")

# Save
exact_df.to_csv('processed/unified_matrix_phylop_fixed.csv.gz', index=False, compression='gzip')
print("\n✅ Saved: processed/unified_matrix_phylop_fixed.csv.gz")

# Check target genes with PhyloP
print("\n📊 Target genes with PhyloP:")
for gene in ['CHEK2', 'NF2', 'TERT', 'HBB']:
    sample = exact_df[exact_df['clinvar_gene'].str.contains(gene, case=False, na=False)]
    if len(sample) > 0:
        print(f"   {gene}: {len(sample)} entries, PhyloP mean: {sample['phyloP'].mean():.4f}")
    else:
        print(f"   {gene}: 0 entries")

print("\n✅ FIX COMPLETE!")
