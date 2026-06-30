"""
Add TERT and HBB eQTLs to the gene-fallback training matrix.
"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧬 ADDING TERT AND HBB TO GENE-FALLBACK MATRIX")
print("=" * 70)

# 1. Load current gene-fallback matrix
train_df = pd.read_csv('processed/training_matrix_allchr.csv.gz', compression='gzip')
print(f"Current matrix: {len(train_df)} rows")

# 2. Load TERT from Brain Caudate
tert_path = Path("../01_Data/processed/tert_caudate_matches.csv")
if tert_path.exists():
    tert_df = pd.read_csv(tert_path)
    # Format to match training matrix
    tert_df['promoter_gene'] = 'TERT'
    tert_df['distance_to_tss'] = tert_df['start_distance']  # Use start_distance as proxy
    tert_df['chrom'] = 'chr' + tert_df['chrom'].astype(str)
    tert_df['pos'] = tert_df['pos'].astype(int)
    
    # Add to matrix
    tert_clean = tert_df[['chrom', 'pos', 'promoter_gene', 'distance_to_tss']].copy()
    print(f"TERT entries to add: {len(tert_clean)}")
else:
    print("❌ TERT file not found!")

# 3. Load HBB from Skin
hbb_path = Path("../01_Data/processed/hbb_skin_matches.csv")
if hbb_path.exists():
    hbb_df = pd.read_csv(hbb_path)
    hbb_df['promoter_gene'] = 'HBB'
    hbb_df['distance_to_tss'] = hbb_df['start_distance']
    hbb_df['chrom'] = 'chr' + hbb_df['chrom'].astype(str)
    hbb_df['pos'] = hbb_df['pos'].astype(int)
    
    hbb_clean = hbb_df[['chrom', 'pos', 'promoter_gene', 'distance_to_tss']].copy()
    print(f"HBB entries to add: {len(hbb_clean)}")
else:
    print("❌ HBB file not found!")

# 4. Combine and save
if 'tert_clean' in locals() and 'hbb_clean' in locals():
    # Add to training matrix
    # Note: These won't have eqtl_slope values, so they'd need to be added separately
    print("\n⚠️ These entries need eqtl_slope values from the exact-match matrix.")
    print("   For now, they are documented as validated but not yet in the training matrix.")
    
    # For dashboard, we could add them as validation entries
    all_entries = pd.concat([tert_clean, hbb_clean], ignore_index=True)
    all_entries.to_csv('processed/tert_hbb_validation_entries.csv.gz', index=False, compression='gzip')
    print(f"\n✅ Saved {len(all_entries)} validation entries to processed/tert_hbb_validation_entries.csv.gz")
else:
    print("\n⚠️ Could not add TERT/HBB entries.")
