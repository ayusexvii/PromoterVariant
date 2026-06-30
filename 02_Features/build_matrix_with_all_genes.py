#!/usr/bin/env python3
"""
Build unified exact-match matrix - PRESERVES ALL GENES.
"""
import pandas as pd
from pathlib import Path

print("=" * 75)
print("🧬 BUILDING UNIFIED MATRIX (ALL GENES PRESERVED)")
print("=" * 75)

script_dir = Path(__file__).resolve().parent

# 1. Load Liver matches
print("📂 Loading Liver matches...")
liver_df = pd.read_csv('processed/exact_matches.csv.gz', compression='gzip')
print(f"   Liver: {len(liver_df)} rows")

# 2. Load TERT
print("📂 Loading TERT...")
tert_path = script_dir / "../01_Data/processed/tert_caudate_matches.csv"
if tert_path.exists():
    tert_df = pd.read_csv(tert_path)
    # Add required columns
    tert_df['clinvar_gene'] = 'TERT'
    tert_df['clinvar_sig'] = 'Validated'
    tert_df['clinvar_chrom'] = tert_df['chrom']
    tert_df['clinvar_pos'] = tert_df['pos']
    tert_df['clinvar_ref'] = tert_df['ref']
    tert_df['clinvar_alt'] = tert_df['alt']
    tert_df['gtex_slope'] = tert_df['slope']
    tert_df['gtex_pval'] = tert_df['pval_nominal']
    tert_df['gtex_gene'] = tert_df['phenotype_id']
    print(f"   TERT: {len(tert_df)} rows")
else:
    tert_df = pd.DataFrame()

# 3. Load HBB
print("📂 Loading HBB...")
hbb_path = script_dir / "../01_Data/processed/hbb_skin_matches.csv"
if hbb_path.exists():
    hbb_df = pd.read_csv(hbb_path)
    hbb_df['clinvar_gene'] = 'HBB'
    hbb_df['clinvar_sig'] = 'Validated'
    hbb_df['clinvar_chrom'] = hbb_df['chrom']
    hbb_df['clinvar_pos'] = hbb_df['pos']
    hbb_df['clinvar_ref'] = hbb_df['ref']
    hbb_df['clinvar_alt'] = hbb_df['alt']
    hbb_df['gtex_slope'] = hbb_df['slope']
    hbb_df['gtex_pval'] = hbb_df['pval_nominal']
    hbb_df['gtex_gene'] = hbb_df['phenotype_id']
    print(f"   HBB: {len(hbb_df)} rows")
else:
    hbb_df = pd.DataFrame()

# 4. Combine ALL
print("\n🔗 Combining all matches...")
combined = pd.concat([liver_df, tert_df, hbb_df], ignore_index=True)
print(f"   Combined: {len(combined)} rows")

# 5. Check target genes
print("\n📊 Target genes in combined (before merge):")
for gene in ['CHEK2', 'NF2', 'TERT', 'HBB']:
    count = combined[combined['clinvar_gene'].fillna('').astype(str).str.contains(gene, case=False, regex=False)].shape[0]
    print(f"   {gene}: {count}")

# 6. Load promoter features
print("\n📂 Loading promoter features...")
prom_df = pd.read_csv('processed/promoter_variants_allchr.txt.gz', sep='\t', compression='gzip')
prom_df['chrom'] = prom_df['chrom'].astype(str).str.replace('chr', '', case=False)
prom_df['pos'] = prom_df['pos'].astype(int)

# 7. Merge
print("🔗 Merging with promoter features...")
combined['chrom'] = combined['clinvar_chrom'].astype(str).str.replace('chr', '', case=False)
combined['pos'] = pd.to_numeric(combined['clinvar_pos'], errors='coerce').fillna(0).astype(int)

merged = combined.merge(
    prom_df[['chrom', 'pos', 'distance_to_tss', 'promoter_gene']],
    on=['chrom', 'pos'],
    how='left'
)

# 8. Add features
merged['abs_distance'] = merged['distance_to_tss'].abs()
gene_counts = merged['promoter_gene'].value_counts().to_dict()
merged['gene_variant_count'] = merged['promoter_gene'].map(gene_counts)

# 9. Add PhyloP
cons_df = pd.read_csv('processed/training_matrix_with_conservation.csv.gz', compression='gzip')
cons_df['chrom'] = cons_df['chrom'].astype(str).str.replace('chr', '', case=False)
cons_df['pos'] = cons_df['pos'].astype(int)
cons_lookup = cons_df[['chrom', 'pos', 'phyloP']].drop_duplicates(subset=['chrom', 'pos'])
merged = merged.merge(cons_lookup, on=['chrom', 'pos'], how='left')
merged['phyloP'] = merged['phyloP'].fillna(0.0)

# 10. Final check
print("\n📊 Target genes in final matrix:")
for gene in ['CHEK2', 'NF2', 'TERT', 'HBB']:
    count = merged[merged['clinvar_gene'].fillna('').astype(str).str.contains(gene, case=False, regex=False)].shape[0]
    print(f"   {gene}: {count}")

# 11. Save
output_path = 'processed/unified_matrix_all_genes.csv.gz'
merged.to_csv(output_path, index=False, compression='gzip')
print(f"\n💾 Saved: {output_path}")
print(f"   Rows: {len(merged)}, Columns: {len(merged.columns)}")
print("=" * 75)
print("✅ COMPLETE!")
