#!/usr/bin/env python3
"""
Parse GTEx V11 Parquet files to tab-separated format.
FIXED: Uses correct column names for V11.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 GTEX V11 PARQUET PARSING ENGINE (FIXED COLUMNS)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    
    # 1. Establish Input Pathways
    gtex_dir = Path.home() / "gtex_analysis" / "raw" / "GTEx_Analysis_v11_eQTL"
    parquet_path = gtex_dir / "Whole_Blood.v11.eQTLs.signif_pairs.parquet"
    
    if not parquet_path.exists():
        print(f"❌ ERROR: File not found at: {parquet_path}")
        sys.exit(1)
            
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "whole_blood_eqtl_parsed.txt"
    
    print(f"📂 Source Input: {parquet_path}")
    print(f"📝 Target Output: {output_path}")
    
    # 2. Fast Ingestion Pass
    print("📂 Loading Parquet file directly into memory...")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Ingested {len(df):,} records with {len(df.columns)} columns.")
    print(f"   Columns: {list(df.columns)}")
    
    # 3. Identify columns (V11 uses different names)
    # V11 likely has: 'variant_id', 'gene_id', 'slope', 'pval_nominal'
    # But let's check dynamically
    
    # Try common column names
    variant_col = None
    gene_col = None
    slope_col = None
    pval_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'variant' in col_lower and not variant_col:
            variant_col = col
        if 'gene' in col_lower and 'gene_id' in col_lower and not gene_col:
            gene_col = col
        if 'slope' in col_lower and not slope_col:
            slope_col = col
        if 'pval' in col_lower and not pval_col:
            pval_col = col
    
    if not variant_col:
        # Try generic column names
        if 'variant' in df.columns:
            variant_col = 'variant'
        elif 'variant_id' in df.columns:
            variant_col = 'variant_id'
        else:
            print("❌ Could not find variant column!")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
    
    if not gene_col:
        if 'gene' in df.columns:
            gene_col = 'gene'
        elif 'gene_id' in df.columns:
            gene_col = 'gene_id'
        else:
            print("❌ Could not find gene column!")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
    
    if not slope_col:
        if 'slope' in df.columns:
            slope_col = 'slope'
        else:
            # Try to find any column with slope-like values
            for col in df.columns:
                if 'effect' in col.lower() or 'slope' in col.lower():
                    slope_col = col
                    break
            if not slope_col:
                slope_col = df.columns[-3]  # fallback
    
    if not pval_col:
        if 'pval_nominal' in df.columns:
            pval_col = 'pval_nominal'
        else:
            for col in df.columns:
                if 'pval' in col.lower() or 'p_value' in col.lower():
                    pval_col = col
                    break
            if not pval_col:
                pval_col = df.columns[-2]  # fallback
    
    print(f"   Using: Variant={variant_col}, Gene={gene_col}, Slope={slope_col}, Pval={pval_col}")
    
    # 4. Vectorized Transformation
    print("🔧 Parsing variant IDs...")
    
    # Create a temporary series for parsing
    variant_series = df[variant_col].astype(str)
    
    # Split on underscore
    splits = variant_series.str.split('_')
    
    # Extract components (right-side anchoring)
    alt_series = splits.str[-2]
    ref_series = splits.str[-3]
    pos_series = splits.str[-4]
    
    # Chromosome: join all parts before the last 4
    chrom_series = splits.apply(lambda x: "_".join(x[:-4]) if len(x) >= 5 else "").str.replace('chr', '', case=False)
    
    # Clean gene IDs
    gene_series = df[gene_col].astype(str).str.split('.').str[0]
    
    # Build output dataframe
    output_df = pd.DataFrame({
        'chrom': chrom_series,
        'pos': pos_series,
        'ref': ref_series,
        'alt': alt_series,
        'gene_id': gene_series,
        'slope': df[slope_col],
        'pval': df[pval_col]
    })
    
    # Filter valid rows
    valid_mask = output_df['pos'].str.isdigit() & (output_df['chrom'] != '')
    filtered_df = output_df[valid_mask]
    
    skipped = len(df) - len(filtered_df)
    
    print(f"✅ Valid rows: {len(filtered_df):,}")
    print(f"   Skipped: {skipped:,}")
    
    # Save
    print(f"📝 Saving to: {output_path}")
    filtered_df.to_csv(output_path, sep='\t', index=False)
    
    print("\n" + "=" * 75)
    print("📊 PARSING REPORT")
    print("=" * 75)
    print(f"   Successfully Parsed  : {len(filtered_df):,} rows")
    print(f"   Skipped/Filtered Out : {skipped:,} rows")
    print(f"   Output File          : {output_path}")
    print(f"   File Size            : {output_path.stat().st_size / (1024**2):.2f} MB")
    print("=" * 75)
    print("✅ GTEX PARSING COMPLETE!")

if __name__ == "__main__":
    main()
