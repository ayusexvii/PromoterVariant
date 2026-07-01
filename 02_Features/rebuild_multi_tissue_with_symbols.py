"""
Rebuild multi-tissue matrix with gene symbol mapping.
"""
import pandas as pd
import json
from pathlib import Path

print("=" * 70)
print("🧬 REBUILDING MULTI-TISSUE MATRIX WITH SYMBOLS")
print("=" * 70)

# Load gene symbol mapping
with open('processed/ensembl_to_symbol.json', 'r') as f:
    ensembl_to_symbol = json.load(f)

print(f"✅ Loaded {len(ensembl_to_symbol):,} mappings")

# 1. Load Liver training matrix
liver_df = pd.read_csv('processed/training_matrix_allchr.csv.gz', compression='gzip')
liver_df['tissue'] = 'Liver'
print(f"Liver: {len(liver_df):,} rows")

# 2. Load Whole Blood
wb_path = Path('processed/whole_blood_training.csv.gz')
if wb_path.exists():
    wb_df = pd.read_csv(wb_path, compression='gzip')
    wb_df['tissue'] = 'Whole_Blood'
    # Map ENSG to gene symbol
    wb_df['promoter_gene'] = wb_df['gene_id'].str.split('.').str[0].map(ensembl_to_symbol)
    wb_df['distance_to_tss'] = 0
    wb_df['eqtl_slope'] = wb_df['slope']
    wb_df['eqtl_pval'] = wb_df['pval']
    print(f"Whole Blood: {len(wb_df):,} rows")
    
    # Check target genes
    print("   Target genes in Whole Blood:")
    for gene in ['TERT', 'HBB', 'NF2', 'APOL1']:
        count = wb_df[wb_df['promoter_gene'] == gene].shape[0]
        print(f"      {gene}: {count} entries")
else:
    wb_df = pd.DataFrame()
    print("Whole Blood: Not found")

# 3. Load Brain
brain_path = Path('processed/brain_combined_training.csv.gz')
if brain_path.exists():
    brain_df = pd.read_csv(brain_path, compression='gzip')
    brain_df['tissue'] = 'Brain'
    brain_df['promoter_gene'] = brain_df['gene_id'].str.split('.').str[0].map(ensembl_to_symbol)
    brain_df['distance_to_tss'] = 0
    brain_df['eqtl_slope'] = brain_df['slope']
    brain_df['eqtl_pval'] = brain_df['pval']
    print(f"Brain: {len(brain_df):,} rows")
    
    print("   Target genes in Brain:")
    for gene in ['TERT', 'HBB', 'NF2', 'APOL1']:
        count = brain_df[brain_df['promoter_gene'] == gene].shape[0]
        print(f"      {gene}: {count} entries")
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
print(f"\n✅ Combined: {len(combined):,} rows")

# 5. Check target genes
print("\n📊 Target genes in multi-tissue matrix (with symbols):")
for gene in ['CHEK2', 'NF2', 'TERT', 'HBB', 'APOL1']:
    count = combined[combined['promoter_gene'] == gene].shape[0]
    print(f"   {gene}: {count} entries")

# 6. Save
combined.to_csv('processed/multi_tissue_matrix_symbols.csv.gz', index=False, compression='gzip')
print("\n💾 Saved to: processed/multi_tissue_matrix_symbols.csv.gz")

# 7. Show tissue distribution
print("\n📊 Tissue distribution:")
print(combined['tissue'].value_counts())
