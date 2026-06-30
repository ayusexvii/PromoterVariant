#!/usr/bin/env python3
"""
Parse GTEx V11 Parquet files to tab-separated format.
Fully vectorized with vector arrays for massive speedup.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    print("=" * 75)
    print("🧬 VECTORIZED GTEX V11 PARQUET PARSING ENGINE")
    print("=" * 75)
    
    script_dir = Path(__file__).resolve().parent
    
    # 1. Establish Input Pathways
    gtex_dir = Path.home() / "gtex_analysis" / "raw" / "GTEx_Analysis_v11_eQTL"
    parquet_path = gtex_dir / "Whole_Blood.v11.eQTLs.signif_pairs.parquet"
    
    if not parquet_path.exists():
        print(f"⚠️ Primary file absent at: {parquet_path}")
        alt_path = gtex_dir / "Whole_Blood.v11.signif_variant_gene_pairs.parquet"
        if alt_path.exists():
            parquet_path = alt_path
        else:
            print("❌ CRITICAL ERROR: No valid variant Parquet sources located.", file=sys.stderr)
            sys.exit(1)
            
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "whole_blood_eqtl_parsed.txt"
    
    print(f"📂 Source Input: {parquet_path}")
    print(f"📝 Target Output: {output_path}")
    
    # 2. Fast Ingestion Pass
    print("📂 Loading Parquet file directly into memory...")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Ingested {len(df):,} records with {len(df.columns)} metric paths.")
    
    # 3. Dynamic Schema Discovery (Run Once, Not Per Row)
    variant_col = 'variant_id' if 'variant_id' in df.columns else ('variant' if 'variant' in df.columns else None)
    gene_col = 'gene_id' if 'gene_id' in df.columns else ('gene' if 'gene' in df.columns else None)
    slope_col = 'slope' if 'slope' in df.columns else 'effect_size'
    pval_col = 'pval_nominal' if 'pval_nominal' in df.columns else 'p_value'
    
    if not variant_col or not gene_col:
        print("❌ CRITICAL ERROR: Missing variant/gene core column arrays.", file=sys.stderr)
        sys.exit(1)
        
    print(f"   Using core layout mappings: Var={variant_col} | Gene={gene_col} | Slope={slope_col} | Pval={pval_col}")
    
    # 4. Vectorized Transformation Pass (No Loops)
    print("🔧 Executing parallel text vector split tracks...")
    
    # Generate vectorized string splits across the full series layout
    id_splits = df[variant_col].astype(str).str.split('_')
    
    # Safeguard against short/corrupt format fields using array masks
    valid_len_mask = id_splits.str.len() >= 5
    if not valid_len_mask.any():
        print("❌ CRITICAL ERROR: Zero variant records conform to the expected coordinate schema.", file=sys.stderr)
        sys.exit(1)
        
    # Extract coordinates directly across memory space
    alt_series = id_splits.str[-2]
    ref_series = id_splits.str[-3]
    pos_series = id_splits.str[-4]
    
    # Join prefix tokens back together to build scaffold entries safely
    chrom_series = id_splits.apply(lambda x: "_".join(x[:-4]) if len(x) >= 5 else "").str.replace('chr', '', case=False)
    
    # Clean gene identifiers without dot tracking variants
    clean_gene_series = df[gene_col].astype(str).str.split('.').str[0]
    
    # 5. Build Final Matrix Layout
    print("📊 Assembling processed matrix elements...")
    parsed_df = pd.DataFrame({
        'chrom': chrom_series,
        'pos': pos_series,
        'ref': ref_series,
        'alt': alt_series,
        'gene_id': clean_gene_series,
        'slope': df[slope_col],
        'pval': df[pval_col]
    })
    
    # Apply clean text filters simultaneously
    numeric_pos_mask = parsed_df['pos'].str.isdigit() == True
    final_mask = valid_len_mask & numeric_pos_mask
    
    skipped_count = len(df) - final_mask.sum()
    filtered_df = parsed_df[final_mask]
    
    # 6. Stream Out Clean Rows to Disk
    print(f"📝 Flushing compressed datasets out to disk storage arrays...")
    filtered_df.to_csv(output_path, sep='\t', index=False, header=True)
    
    print("\n" + "=" * 75)
    print("📊 VECTORIZED COMPLETE DISPATCH REPORT")
    print("=" * 75)
    print(f"   Successfully Parsed  : {len(filtered_df):,} rows")
    print(f"   Skipped/Filtered Out : {skipped_count:,} rows")
    print(f"   Output Saved Cleanly : {output_path}")
    print(f"   Disk File Footprint  : {output_path.stat().st_size / (1024**2):.2f} MB")
    print("=" * 75)
    print("✅ GTEX PARQUET TRANSLATION SEAMLESSLY CONCLUDED!")

if __name__ == "__main__":
    main()
