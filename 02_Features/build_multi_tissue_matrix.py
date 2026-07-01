"""
Build unified multi-tissue training matrix with Liver + Whole Blood + Brain.
"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧬 BUILDING MULTI-TISSUE TRAINING MATRIX")
print("=" * 70)

# 1. Load Liver training matrix
liver_df = pd.read_csv('processed/training_matrix_allchr.csv.gz', compression='gzip')
liver_df['tissue'] = 'Liver'
print(f"Liver: {len(liver_df):,} rows")

# 2. Load Whole Blood
wb_path = Path('processed/whole_blood_training.csv.gz')
if wb_path.exists():
    wb_df = pd.read_csv(wb_path, compression='gzip')
    wb_df['tissue'] = 'Whole_Blood'
    wb_df['promoter_gene'] = wb_df['gene_id']
    wb_df['distance_to_tss'] = 0
    wb_df['eqtl_slope'] = wb_df['slope']
    wb_df['eqtl_pval'] = wb_df['pval']
    wb_df['source_gene'] = wb_df['gene_id']
    print(f"Whole Blood: {len(wb_df):,} rows")
else:
    wb_df = pd.DataFrame()
    print("Whole Blood: Not found")

# 3. Load Brain combined
brain_path = Path('processed/brain_combined_training.csv.gz')
if brain_path.exists():
    brain_df = pd.read_csv(brain_path, compression='gzip')
    brain_df['tissue'] = 'Brain'
    brain_df['promoter_gene'] = brain_df['gene_id']
    brain_df['distance_to_tss'] = 0
    brain_df['eqtl_slope'] = brain_df['slope']
    brain_df['eqtl_pval'] = brain_df['pval']
    brain_df['source_gene'] = brain_df['gene_id']
    print(f"Brain: {len(brain_df):,} rows")
else:
    brain_df = pd.DataFrame()
    print("Brain: Not found")

# 4. Combine
dfs_to_combine = [liver_df]
if not wb_df.empty:
    dfs_to_combine.append(wb_df)
if not brain_df.empty:
    dfs_to_combine.append(brain_df)

combined = pd.concat(dfs_to_combine, ignore_index=True)
print(f"\n✅ Combined: {len(combined):,} rows across {combined['tissue'].nunique()} tissues")

# 5. Check target genes
print("\n📊 Target genes in multi-tissue matrix:")
for gene in ['CHEK2', 'NF2', 'TERT', 'HBB', 'APOL1']:
    count = combined[combined['promoter_gene'].fillna('').astype(str).str.contains(gene, case=False, na=False)].shape[0]
    print(f"   {gene}: {count} entries")

# 6. Save
combined.to_csv('processed/multi_tissue_matrix.csv.gz', index=False, compression='gzip')
print("\n💾 Saved to: processed/multi_tissue_matrix.csv.gz")

# 7. Show tissue distribution
print("\n📊 Tissue distribution:")
print(combined['tissue'].value_counts())
