#!/usr/bin/env python3
"""
Extract ±50bp sequences - FIXED VERSION
Better handling of chromosome names and coordinates.
"""
import py2bit
import pandas as pd
import time
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 SEQUENCE EXTRACTION - FIXED")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    twobit_path = script_dir / '../01_Data/raw/hg38.2bit'
    input_path = script_dir / 'processed' / 'promoter_variants_allchr.txt.gz'
    output_path = script_dir / 'processed' / 'variant_sequences_fixed.csv.gz'
    
    if not twobit_path.exists():
        print(f"❌ ERROR: Genome file missing at: {twobit_path}")
        sys.exit(1)
    if not input_path.exists():
        print(f"❌ ERROR: Input file missing at: {input_path}")
        sys.exit(1)
        
    print("📂 Loading reference genome...")
    tb = py2bit.open(str(twobit_path))
    chroms = set(tb.chroms().keys())
    print(f"✅ Loaded {len(chroms)} chromosomes")
    
    print(f"📂 Loading variants...")
    df = pd.read_csv(input_path, sep='\t', compression='gzip')
    print(f"✅ Loaded {len(df):,} variants")
    
    # Determine column names
    chrom_col = 'chrom' if 'chrom' in df.columns else ('Chromosome' if 'Chromosome' in df.columns else None)
    pos_col = 'pos' if 'pos' in df.columns else ('Start' if 'Start' in df.columns else ('End' if 'End' in df.columns else None))
    ref_col = 'ref' if 'ref' in df.columns else ('Ref' if 'Ref' in df.columns else None)
    alt_col = 'alt' if 'alt' in df.columns else ('Alt' if 'Alt' in df.columns else None)
    
    print(f"🎯 Using: {chrom_col}, {pos_col}, {ref_col}, {alt_col}")
    
    ref_seqs = []
    alt_seqs = []
    errors = 0
    
    start_time = time.time()
    total = len(df)
    
    for i, row in df.iterrows():
        chrom_raw = str(row[chrom_col])
        chrom = f"chr{chrom_raw}" if not chrom_raw.startswith('chr') else chrom_raw
        
        pos = int(row[pos_col])
        ref = str(row[ref_col]) if pd.notna(row[ref_col]) else 'N'
        alt = str(row[alt_col]) if pd.notna(row[alt_col]) else 'N'
        
        # Use 0-based coordinates for 2bit
        start = max(0, pos - 50)
        end = pos + 51  # Half-open: includes position 50
        
        if chrom not in chroms:
            ref_seqs.append("N" * 101)
            alt_seqs.append("N" * 101)
            errors += 1
            continue
        
        try:
            seq = tb.sequence(chrom, start, end)
            if len(seq) >= 101:
                ref_seq = seq[:101].upper()
                # Create alt sequence
                alt_seq = list(ref_seq)
                if len(alt_seq) > 50:
                    alt_seq[50] = alt
                alt_seq = "".join(alt_seq)
                ref_seqs.append(ref_seq)
                alt_seqs.append(alt_seq)
            else:
                ref_seqs.append("N" * 101)
                alt_seqs.append("N" * 101)
                errors += 1
        except Exception as e:
            ref_seqs.append("N" * 101)
            alt_seqs.append("N" * 101)
            errors += 1
        
        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1:,}/{total:,}...")
    
    tb.close()
    
    df['ref_sequence'] = ref_seqs
    df['alt_sequence'] = alt_seqs
    df.to_csv(output_path, index=False, compression='gzip')
    
    print(f"\n✅ Saved {len(df):,} sequences")
    print(f"   Errors: {errors}")
    print(f"   Time: {time.time() - start_time:.2f}s")
    
    # Show sample
    sample = df.iloc[0]
    print(f"\n🔍 Sample:")
    print(f"   Ref: {sample['ref_sequence'][:50]}...{sample['ref_sequence'][50]}...{sample['ref_sequence'][51:]}")
    print(f"   Alt: {sample['alt_sequence'][:50]}...{sample['alt_sequence'][50]}...{sample['alt_sequence'][51:]}")
    
    print("\n✅ COMPLETE!")

if __name__ == "__main__":
    main()
