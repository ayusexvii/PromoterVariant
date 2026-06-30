#!/usr/bin/env python3
"""
Build unified exact-match training matrix from ALL tissues.
FIXED: Handles different column names in tissue files.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 ACCELERATED UNIFIED MATRIX COMPILATION ENGINE (FIXED)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load available data slices
    tissue_files = {
        'Liver (V8)': (script_dir / "processed" / "exact_matches.csv.gz", True),
        'Brain Caudate (TERT)': (script_dir / "../01_Data/processed/tert_caudate_matches.csv", False),
        'Skin (HBB)': (script_dir / "../01_Data/processed/hbb_skin_matches.csv", False)
    }
    
    all_matches = []
    
    for name, (path, is_required) in tissue_files.items():
        print(f"📂 Evaluating source file channel: {name}...")
        if path.exists():
            tdf = pd.read_csv(path, compression='gzip' if path.suffix == '.gz' else None)
            
            # --- FIX: Normalize column names ---
            # Convert 'pos' to 'clinvar_pos' if needed
            if 'pos' in tdf.columns and 'clinvar_pos' not in tdf.columns:
                tdf['clinvar_pos'] = tdf['pos']
            if 'chrom' in tdf.columns and 'clinvar_chrom' not in tdf.columns:
                tdf['clinvar_chrom'] = tdf['chrom']
            
            # Inject tracking labels if missing
            if 'clinvar_gene' not in tdf.columns:
                if 'tert' in path.name.lower():
                    tdf['clinvar_gene'] = 'TERT'
                elif 'hbb' in path.name.lower():
                    tdf['clinvar_gene'] = 'HBB'
                else:
                    tdf['clinvar_gene'] = 'UNKNOWN'
            
            # Ensure gtex_gene column exists
            if 'gtex_gene' not in tdf.columns:
                tdf['gtex_gene'] = tdf.get('phenotype_id', 'NA')
            
            # Ensure slope column exists
            slope_col = [c for c in tdf.columns if c.lower() in ['slope', 'gtex_slope', 'effect_size']]
            if slope_col and 'gtex_slope' not in tdf.columns:
                tdf['gtex_slope'] = tdf[slope_col[0]]
            
            # Ensure pval column exists
            pval_col = [c for c in tdf.columns if c.lower() in ['pval_nominal', 'pval', 'p_value']]
            if pval_col and 'gtex_pval' not in tdf.columns:
                tdf['gtex_pval'] = tdf[pval_col[0]]
            
            # Fill missing columns with NA
            for col in ['clinvar_ref', 'clinvar_alt', 'clinvar_sig', 'gtex_ref', 'gtex_alt']:
                if col not in tdf.columns:
                    tdf[col] = 'NA'
            
            print(f"   📊 Ingested {len(tdf):,} rows. Columns: {list(tdf.columns)[:10]}...")
            all_matches.append(tdf)
        elif is_required:
            print(f"❌ CRITICAL ERROR: Required file missing: {path}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"   ⚠️ Optional dataset absent on disk, skipping.")
            
    # 2. Concat matrices
    print("\n🔗 Joining all matching matrices...")
    combined = pd.concat(all_matches, axis=0, ignore_index=True, sort=False)
    print(f"   Combined Data Pool footprint: {len(combined):,} raw alignment elements.")
    
    # 3. Load Promoter Features
    promoter_path = output_dir / "promoter_variants_allchr.txt.gz"
    cons_path = output_dir / "training_matrix_with_conservation.csv.gz"
    output_path = output_dir / "unified_exact_match_matrix.csv.gz"
    
    if not promoter_path.exists():
        print(f"❌ ERROR: Promoter definitions file missing: {promoter_path}", file=sys.stderr)
        sys.exit(1)
        
    print("📂 Ingesting promoter variant metrics layout...")
    prom_df = pd.read_csv(promoter_path, sep='\t', compression='gzip')
    
    # Normalize coordinate strings (FIXED: handle NaN values)
    combined['chrom'] = combined['clinvar_chrom'].astype(str).str.replace('chr', '', case=False)
    combined['pos'] = pd.to_numeric(combined['clinvar_pos'], errors='coerce').fillna(0).astype(int)
    prom_df['chrom'] = prom_df['chrom'].astype(str).str.replace('chr', '', case=False)
    prom_df['pos'] = prom_df['pos'].astype(int)
    
    # 4. Multi-Column Vectorized Merge
    print("🔗 Blending feature variables on multi-column keys...")
    prom_lookup = prom_df[['chrom', 'pos', 'distance_to_tss', 'promoter_gene']].drop_duplicates(subset=['chrom', 'pos'])
    merged = combined.merge(prom_lookup, on=['chrom', 'pos'], how='left')
    
    # 5. Add Engineered Metric Dimensions
    merged['abs_distance'] = merged['distance_to_tss'].abs()
    gene_counts = merged['promoter_gene'].value_counts().to_dict()
    merged['gene_variant_count'] = merged['promoter_gene'].map(gene_counts)
    
    # 6. Add PhyloP
    if cons_path.exists():
        print("📂 Ingesting conservation map metadata tracks...")
        cons_df = pd.read_csv(cons_path, compression='gzip')
        cons_df['chrom'] = cons_df['chrom'].astype(str).str.replace('chr', '', case=False)
        cons_df['pos'] = cons_df['pos'].astype(int)
        
        cons_lookup = cons_df[['chrom', 'pos', 'phyloP']].drop_duplicates(subset=['chrom', 'pos'])
        merged = merged.merge(cons_lookup, on=['chrom', 'pos'], how='left')
    else:
        print("   ⚠️ Warning: Conservation source file missing. Setting defaults to 0.0.")
        merged['phyloP'] = 0.0
        
    merged['phyloP'] = merged['phyloP'].fillna(0.0)
    
    # 7. Select Target Feature Schema Space
    feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP']
    metadata_cols = ['chrom', 'pos', 'clinvar_gene', 'gtex_gene']
    
    # Safely locate the eQTL slope column
    slope_col_candidate = [c for c in merged.columns if c.lower() in ['gtex_slope', 'slope', 'eqtl_slope']]
    if not slope_col_candidate:
        print("❌ CRITICAL ERROR: Target effect-size column missing from datasets.", file=sys.stderr)
        print(f"   Available columns: {list(merged.columns)}")
        sys.exit(1)
    target_slope = slope_col_candidate[0]
    
    train_df = merged[metadata_cols + [target_slope] + feature_cols].copy()
    train_df.rename(columns={target_slope: 'eqtl_slope'}, inplace=True)
    
    # 8. Clean up incomplete data
    initial_count = len(train_df)
    train_df = train_df.dropna(subset=feature_cols + ['eqtl_slope'])
    print(f"🧹 Filtered out {initial_count - len(train_df):,} rows with incomplete data fields.")
    print(f"✅ Final Training Matrix Size: {len(train_df):,} rows | {len(train_df.columns)} columns.")
    
    # 9. Verify target gene coverage
    print("\n📊 Target Genes Distribution Summary:")
    for gene in ['CHEK2', 'NF2', 'TERT', 'HBB']:
        count = train_df[train_df['clinvar_gene'].fillna('').astype(str).str.contains(gene, case=False, regex=False)].shape[0]
        print(f"   - {gene:<10}: {count:,} entries found.")
        
    # 10. Save
    train_df.to_csv(output_path, index=False, compression='gzip')
    print(f"\n💾 Saved unified tracking matrix to: {output_path}")
    print(f"   Output footprint mass: {output_path.stat().st_size / (1024**2):.2f} MB")
    print("=" * 75)
    print("✅ PROCESS COMPLETE! MULTI-TISSUE ARCHITECTURE ENCODED.")
    print("=" * 75)

if __name__ == "__main__":
    main()
