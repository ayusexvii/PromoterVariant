#!/usr/bin/env python3
"""
Build training matrix from exact matches.
Optimized via multi-column vectorized indexing to prevent Cartesian explosions.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 ACCELERATED MATRIX COMPILATION ENGINE")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    exact_path = script_dir / "processed" / "exact_matches.csv.gz"
    promoter_path = script_dir / "processed" / "promoter_variants_allchr.txt.gz"
    output_path = script_dir / "processed" / "exact_match_training_matrix.csv.gz"
    
    # 1. Load Structural Intersection Assets Safely
    if not exact_path.exists():
        print(f"❌ ERROR: Source cross-matches database missing: {exact_path}", file=sys.stderr)
        sys.exit(1)
    if not promoter_path.exists():
        print(f"❌ ERROR: Upstream transcript annotations missing: {promoter_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Ingesting variant coordinates: {exact_path.name}")
    df = pd.read_csv(exact_path, compression='gzip')
    
    print(f"📂 Ingesting coordinate metadata matrices: {promoter_path.name}")
    prom_df = pd.read_csv(promoter_path, sep='\t', compression='gzip')
    
    # 2. Dynamic Column Schema Identification
    # Normalize naming schemas case-insensitively to prevent mapping crashes
    df_cols = {c.lower(): c for c in df.columns}
    prom_cols = {c.lower(): c for c in prom_df.columns}
    
    gtex_slope_col = [c for c in ['gtex_slope', 'slope'] if c in df_cols]
    gtex_pval_col = [c for c in ['gtex_pval', 'pval'] if c in df_cols]
    
    if not gtex_slope_col or not gtex_pval_col:
        print("❌ ERROR: Input coordinates are missing necessary slope or p-value columns.", file=sys.stderr)
        sys.exit(1)
        
    target_slope = [c for c in ['gtex_slope', 'slope'] if c in df_cols][0]
    target_pval = [c for c in ['gtex_pval', 'pval'] if c in df_cols][0]
    
    # 3. Vectorized Multi-Column Data Merge (Replaces slow string concatenation)
    print("🔗 Linking coordinate records via multi-column indexing...")
    
    # Clean and align types before merging to ensure a perfect join match
    df['clinvar_chrom'] = df['clinvar_chrom'].astype(str).str.replace('chr', '', case=False)
    prom_df['chrom'] = prom_df['chrom'].astype(str).str.replace('chr', '', case=False)
    df['clinvar_pos'] = df['clinvar_pos'].astype(int)
    prom_df['pos'] = prom_df['pos'].astype(int)
    
    # Deduplicate right-hand references first to isolate features and avoid structural duplicates
    prom_features = ['chrom', 'pos', 'distance_to_tss', 'promoter_gene']
    if 'phyloP'.lower() in prom_cols:
        prom_features.append(prom_df.columns[list(prom_cols.keys()).index('phyloP'.lower())])
        
    prom_lookup = prom_df[prom_features].drop_duplicates(subset=['chrom', 'pos'])
    
    merged = df.merge(
        prom_lookup,
        left_on=['clinvar_chrom', 'clinvar_pos'],
        right_on=['chrom', 'pos'],
        how='left'
    )
    
    # 4. Feature Engineering and Metadata Fill Layer
    print("🔧 Normalizing structural features...")
    merged['abs_distance'] = merged['distance_to_tss'].abs()
    
    gene_counts = merged['promoter_gene'].value_counts().to_dict()
    merged['gene_variant_count'] = merged['promoter_gene'].map(gene_counts)
    
    # Safe fallback handler for missing conservation metrics
    phyloP_actual = [c for c in merged.columns if c.lower() == 'phylop']
    if phyloP_actual:
        merged['phyloP'] = merged[phyloP_actual[0]].fillna(0.0)
    else:
        merged['phyloP'] = 0.0
        
    # 5. Clean Deduplication Using Vectorized Absolute Magnitudes
    print("🔧 Dropping redundant overlaps...")
    merged['abs_pval'] = merged[target_pval].abs()
    
    # Sort signals to prioritize and keep the lowest valid absolute p-value
    merged_dedup = (merged.sort_values('abs_pval', ascending=True)
                    .drop_duplicates(subset=['clinvar_chrom', 'clinvar_pos', 'gtex_gene'], keep='first'))
    
    # 6. Extract Traceable Final Training Data
    # Includes core metadata column slices to preserve physical traceability downstream
    metadata_cols = ['clinvar_chrom', 'clinvar_pos', 'clinvar_ref', 'clinvar_alt', 'promoter_gene', 'gtex_gene']
    engineered_features = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP']
    
    all_output_cols = metadata_cols + engineered_features + [target_slope]
    
    train_df = merged_dedup[all_output_cols].copy()
    
    # Clean up output formatting for model consumption
    train_df.rename(columns={target_slope: 'eqtl_slope'}, inplace=True)
    
    # Clean up empty rows
    initial_len = len(train_df)
    train_df.dropna(subset=engineered_features + ['eqtl_slope'], inplace=True)
    dropped_count = initial_len - len(train_df)
    
    if len(train_df) == 0:
        print("❌ CRITICAL ERROR: Post-processing filtering cleared all records. Check input feature distributions.", file=sys.stderr)
        sys.exit(1)
        
    # 7. Write Complete Matrix Dataset to Disk
    print(f"📝 Flushing complete feature matrix out to disk storage...")
    train_df.to_csv(output_path, index=False, compression='gzip')
    
    # Print Metrics Summary Table
    print("\n" + "=" * 75)
    print("📊 COMPLETED MATRIX FEATURE REPORT SUMMARY")
    print("=" * 75)
    print(f"   Compiled Rows Count        : {len(train_df):,}")
    print(f"   Dropped Empty Incomplete Rows: {dropped_count:,}")
    print(f"   Total Feature Dimensionality : {len(engineered_features)} inputs")
    print(f"   Target Signal Metrics Profile:")
    print(f"      - Mean Expression Magnitude: {train_df['eqtl_slope'].mean():.4f}")
    print(f"      - Variance Standard Dev    : {train_df['eqtl_slope'].std():.4f}")
    print(f"      - Measurable Spread Limits : {train_df['eqtl_slope'].min():.4f} to {train_df['eqtl_slope'].max():.4f}")
    
    print("\n🔍 SNIPPET SAMPLES OF EXTRACTED TRACKS (TRACEOBJECT ALLIGNED):")
    display_cols = ['clinvar_chrom', 'clinvar_pos', 'promoter_gene', 'distance_to_tss', 'phyloP', 'eqtl_slope']
    print(train_df[display_cols].head(5).to_string(index=False))
    print("=" * 75)
    print("✅ COMPILATION RUN COMPLETED WITHOUT INTERRUPTIONS!")

if __name__ == "__main__":
    main()
