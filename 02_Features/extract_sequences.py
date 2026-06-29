#!/usr/bin/env python3
"""
Extract ±50bp sequences (ref and alt alleles) for all variants.
Engineered with corrected 0-based indexing limits and strand tracking logic.
"""
import py2bit
import pandas as pd
import time
import sys
from pathlib import Path

def reverse_complement(dna_str: str) -> str:
    """Compute the reverse complement of a DNA sequence string."""
    comp = str.maketrans('ATCGN', 'TAGCN')
    return dna_str.translate(comp)[::-1]

def main():
    print("=" * 70)
    print("🧬 HARDWARE-OPTIMIZED TRANSCRIPTION ALIGNED SEQUENCE EXTRACTOR")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    twobit_path = script_dir / '../01_Data/raw/hg38.2bit'
    input_path = script_dir / 'processed' / 'promoter_variants_allchr.txt.gz'
    output_path = script_dir / 'processed' / 'variant_sequences.csv.gz'
    
    # 1. Establish file attachments safely
    if not twobit_path.exists():
        print(f"❌ ERROR: Genome source hg38.2bit track missing at: {twobit_path}", file=sys.stderr)
        sys.exit(1)
    if not input_path.exists():
        print(f"❌ ERROR: Target overlay input file missing at: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    print("📂 Connecting to hg38.2bit memory map reference stream...")
    tb = py2bit.open(str(twobit_path))
    
    print(f"📂 Unpacking variant matrix: {input_path}")
    df = pd.read_csv(input_path, sep='\t', compression='gzip')
    print(f"✅ Loaded {len(df):,} variants into processing RAM.")
    
    # 2. Determine structural schema properties
    chrom_col = 'chrom' if 'chrom' in df.columns else ('Chromosome' if 'Chromosome' in df.columns else None)
    pos_col = 'pos' if 'pos' in df.columns else ('Start' if 'Start' in df.columns else ('End' if 'End' in df.columns else None))
    ref_col = 'ref' if 'ref' in df.columns else ('Ref' if 'Ref' in df.columns else None)
    alt_col = 'alt' if 'alt' in df.columns else ('Alt' if 'Alt' in df.columns else None)
    strand_col = 'strand' if 'strand' in df.columns else ('Strand' if 'Strand' in df.columns else None)
    
    if not all([chrom_col, pos_col, ref_col, alt_col]):
        print(f"❌ Missing critical columns in input file! Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
        
    print(f"🎯 Indices aligned: Chrom: '{chrom_col}', Pos: '{pos_col}', Ref: '{ref_col}', Alt: '{alt_col}'")
    
    # 3. Memory Array Allocation Execution Pass
    ref_seqs = []
    alt_seqs = []
    errors = 0
    zero_pos = 0
    start_time = time.time()
    total = len(df)
    
    twobit_chroms = set(tb.chroms().keys())
    
    # itertuples drops overhead processing steps entirely
    for row in df.itertuples(index=True):
        idx = row.Index
        
        raw_chrom = str(getattr(row, chrom_col))
        chrom = raw_chrom if raw_chrom.startswith('chr') else f"chr{raw_chrom}"
        
        # Pull Strand Orientation when tracking target profiles
        strand = str(getattr(row, strand_col)) if strand_col else '+'
        if strand not in ['+', '-']:
            strand = '+'
            
        try:
            pos_1based = int(getattr(row, pos_col))
        except (ValueError, TypeError):
            pos_1based = 0
            
        if pos_1based <= 0:
            zero_pos += 1
            ref_seqs.append("N" * 101)
            alt_seqs.append("N" * 101)
            continue
            
        # Extract individual base identifiers
        ref_allele = str(getattr(row, ref_col)).upper() if pd.notna(getattr(row, ref_col)) else 'N'
        alt_allele = str(getattr(row, alt_col)).upper() if pd.notna(getattr(row, alt_col)) else 'N'
        
        # MATHEMATICAL FIX: For ±50bp window centered exactly on mutated nucleotide base (total 101bp)
        # 1-based coordinate maps to 0-based tracking system at position pos_1based - 1
        pos_0based = pos_1based - 1
        start_idx = max(0, pos_0based - 50)
        end_idx = pos_0based + 51  # Half-open limits ensure range total width == 101
        
        if chrom not in twobit_chroms:
            errors += 1
            ref_seqs.append("N" * 101)
            alt_seqs.append("N" * 101)
            continue
            
        try:
            # Query fast sequence extraction layer
            ref_seq = tb.sequence(chrom, start_idx, end_idx).upper()
            
            if len(ref_seq) == 101:
                alt_list = list(ref_seq)
                center_offset = 50
                
                # Insert variant structural substitution mutation safely
                alt_list[center_offset] = alt_allele
                alt_seq = "".join(alt_list)
                
                # BIOLOGICAL CORRECTION: Compute reverse complements for antisense target profiles
                if strand == '-':
                    ref_seq = reverse_complement(ref_seq)
                    alt_seq = reverse_complement(alt_seq)
            else:
                ref_seq = "N" * 101
                alt_seq = "N" * 101
                errors += 1
        except Exception:
            ref_seq = "N" * 101
            alt_seq = "N" * 101
            errors += 1
            
        ref_seqs.append(ref_seq)
        alt_seqs.append(alt_seq)
        
        if (idx + 1) % 25000 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            print(f"   ⚡ Cache Velocity Profile: Processed {idx+1:,}/{total:,} sequences... ({elapsed:.1f}s)")
            
    tb.close()
    
    # 4. Commit transformations and export safely
    df['ref_sequence'] = ref_seqs
    df['alt_sequence'] = alt_seqs
    
    print(f"📝 Compacting output vector to file: {output_path}")
    df.to_csv(output_path, index=False, compression='gzip')
    
    # Diagnostic Breakdown display
    print("\n" + "=" * 70)
    print("📊 QUALITY METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Sequences Preserved   : {len(df):,}")
    print(f"System Operational Errors   : {errors}")
    print(f"Missing Coordinate Faults   : {zero_pos}")
    print(f"Processing Velocity Elapsed : {time.time() - start_time:.2f}s")
    print("-" * 70)
    
    print("\n🔍 Sequence Alignment Previews (First 3 Extracted Tracks):")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        gene = row['promoter_gene'] if 'promoter_gene' in df.columns else 'Unknown'
        strand_display = row[strand_col] if strand_col else '+'
        print(f"   [{i+1}] Gene Symbol: {gene} (Strand: {strand_display})")
        print(f"       Ref (Center Base 50): {row['ref_sequence'][:45]}...{row['ref_sequence'][50]}...{row['ref_sequence'][56:]}")
        print(f"       Alt (Center Base 50): {row['alt_sequence'][:45]}...{row['alt_sequence'][50]}...{row['alt_sequence'][56:]}")
    print("=" * 70)
    print("✅ SEQUENCING EXTRACTION STEP RUN STABLE AND COMPLETE!")

if __name__ == "__main__":
    main()
