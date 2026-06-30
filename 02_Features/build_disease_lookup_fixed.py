#!/usr/bin/env python3
"""
Build disease gene lookup from ClinVar.
Prevents header contamination and supports flexible string column detection.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 REPRODUCIBLE DISEASE GENE LOOKUP (FIXED & NORMALIZED)")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / '../01_Data/processed/clinvar_regulatory.txt.gz'
    output_dir = script_dir / 'processed'
    output_path = output_dir / 'disease_genes_fixed.txt'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not input_path.exists():
        print(f"❌ ERROR: Source archive missing from disk at: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Stream-loading ClinVar variant tracks from: {input_path.name}")
    
    # Ingest file schema layout properties to determine key labels dynamically
    try:
        # Load header separately to find target columns safely by string name
        header_df = pd.read_csv(input_path, sep='\t', compression='gzip', nrows=2)
        
        # Determine target gene column name case-insensitively
        gene_col_candidates = [c for c in header_df.columns if c.lower() in ['gene', 'genesymbol', 'gene_symbol']]
        if not gene_col_candidates:
            print(f"❌ ERROR: Could not identify a valid gene column in header: {list(header_df.columns)}", file=sys.stderr)
            sys.exit(1)
        target_col = gene_col_candidates[0]
        
        # Load data safely by specifying header layout position
        df = pd.read_csv(input_path, sep='\t', compression='gzip', header=0, usecols=[target_col])
    except Exception as e:
        print(f"❌ ERROR: Failed reading data array matrix: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Strict purification layer: eliminates missing metrics, strips text padding, casts to uppercase
    pathogenic_genes = set(
        df[target_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
    )
    
    # Remove header string contamination if it bypassed filters
    pathogenic_genes.discard(target_col.upper())
    
    print(f"✅ Extracted {len(pathogenic_genes):,} pure normalized pathogenic genes.")
    print(f"   🧬 Target Verification Pass -> CHEK2 present in set: {'CHEK2' in pathogenic_genes}")
    
    # Commit cleanly to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        for gene in sorted(pathogenic_genes):
            f.write(f"{gene}\n")
            
    print(f"💾 Clean records successfully saved to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
