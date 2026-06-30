#!/usr/bin/env python3
"""
Parse GTEx V11 Parquet files to tab-separated format.
FINAL: Correct column names for V11.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 GTEX V11 PARQUET PARSING (FINAL - CORRECT COLUMNS)")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    
    # Input/Output paths
    gtex_dir = Path.home() / "gtex_analysis" / "raw" / "GTEx_Analysis_v11_eQTL"
    parquet_path = gtex_dir / "Whole_Blood.v11.eQTLs.signif_pairs.parquet"
    
    if not parquet_path.exists():
        print(f"❌ ERROR: File not found at: {parquet_path}")
        sys.exit(1)
            
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "whole_blood_eqtl_parsed.txt"
    
    print(f"📂 Source: {parquet_path}")
    print(f"📝 Output: {output_path}")
    
    # Load Parquet
    print("📂 Loading Parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns.")
    
    # V11 column names:
    # phenotype_id = gene ID (ENSG...)
    # variant_id = chr1_64764_C_T_b38
    # slope = effect size
    # pval_nominal = p-value
    
    variant_col = 'variant_id'
    gene_col = 'phenotype_id'
    slope_col = 'slope'
    pval_col = 'pval_nominal'
    
    print(f"   Using: Variant={variant_col}, Gene={gene_col}, Slope={slope_col}, Pval={pval_col}")
    
    # Parse variant IDs
    print("🔧 Parsing variant IDs...")
    variant_series = df[variant_col].astype(str)
    splits = variant_series.str.split('_')
    
    # Right-side anchoring: chr1_64764_C_T_b38
    alt_series = splits.str[-2]
    ref_series = splits.str[-3]
    pos_series = splits.str[-4]
    chrom_series = splits.apply(lambda x: "_".join(x[:-4]) if len(x) >= 5 else "").str.replace('chr', '', case=False)
    
    # Clean gene IDs (remove version suffix)
    gene_series = df[gene_col].astype(str).str.split('.').str[0]
    
    # Build output
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
    
    # Show sample
    print("\n🔍 Sample output:")
    print(filtered_df.head(10).to_string())
    print("=" * 75)
    print("✅ GTEX PARSING COMPLETE!")

if __name__ == "__main__":
    main()
