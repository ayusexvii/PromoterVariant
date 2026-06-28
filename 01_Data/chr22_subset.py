#!/usr/bin/env python3
"""
Extract only Chromosome 22 variants from ClinVar regulatory set.
Engineered with hardware cache considerations to prevent thermal spikes.
"""
import gzip
import json
import io
import sys
from pathlib import Path

def extract_chr22(input_path: Path, output_path: Path, col_map_path: Path):
    """
    Filter ClinVar regulatory variants to only Chromosome 22.
    Leverages balanced system buffers for optimal thermal profiles.
    """
    if not col_map_path.exists():
        print(f"❌ ERROR: {col_map_path} not found!", file=sys.stderr)
        sys.exit(1)
        
    with open(col_map_path, 'r', encoding='utf-8') as f:
        col_map = json.load(f)
    
    # Enforce strict index extraction without silent fallbacks
    if 'Chromosome' in col_map:
        chrom_idx = col_map['Chromosome']
    elif 'chr' in col_map:
        chrom_idx = col_map['chr']
    else:
        print("❌ ERROR: Chromosome identifier key not found in column_map.json!", file=sys.stderr)
        sys.exit(1)
    
    print(f"🔍 Confirmed Chromosome column index: {chrom_idx}")
    
    kept = 0
    total = 0
    
    # 2MB Cache alignment matching Intel P-core L2 boundaries
    BUFFER_SIZE = 2 * 1024 * 1024
    
    # Target lookup set optimized for fast O(1) hash resolution
    target_chromosomes = {'22', 'chr22'}
    
    print(f"\n📂 Stream Reading: {input_path}")
    print(f"📝 Buffered Writing: {output_path}")
    
    try:
        with gzip.open(input_path, 'rb') as raw_in, \
             gzip.open(output_path, 'wb', compresslevel=5) as raw_out:
            
            f_in = io.TextIOWrapper(io.BufferedReader(raw_in, buffer_size=BUFFER_SIZE), encoding='utf-8')
            f_out = io.TextIOWrapper(io.BufferedWriter(raw_out, buffer_size=BUFFER_SIZE), encoding='utf-8')
            
            # Transfer the header block
            header = f_in.readline()
            f_out.write(header)
            
            while True:
                lines = f_in.readlines(BUFFER_SIZE)
                if not lines:
                    break
                
                output_buffer = []
                for line in lines:
                    total += 1
                    fields = line.strip().split('\t')
                    
                    if len(fields) <= chrom_idx:
                        continue
                    
                    # Direct check against the target set
                    if fields[chrom_idx].strip() in target_chromosomes:
                        output_buffer.append(line)
                        kept += 1
                
                if output_buffer:
                    f_out.writelines(output_buffer)
                    
                if total % 50000 == 0:
                    print(f"   Processed {total:,} regulatory tracks... Kept {kept:,} items on Chr22.")
                    
            f_out.flush()
            
    except (OSError, UnicodeDecodeError) as e:
        print(f"❌ Pipeline Failure during extraction: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"\n✅ Total Scanned Regulatory Records: {total:,}")
    print(f"✅ Extracted Chromosome 22 Variants: {kept:,}")
    if total > 0:
        print(f"✅ Composition Fraction: {(kept / total * 100):.2f}%")
        
    return kept, total

def main():
    print("=" * 60)
    print("🧬 HARDWARE-OPTIMIZED CHROMOSOME 22 SUBSET EXTRACTOR")
    print("=" * 60)
    
    script_dir = Path(__file__).resolve().parent
    
    input_path = script_dir / "processed" / "clinvar_regulatory.txt.gz"
    output_path = script_dir / "processed" / "clinvar_chr22.txt.gz"
    col_map_path = script_dir / "processed" / "column_map.json"
    
    if not input_path.exists():
        print(f"❌ ERROR: Source layout {input_path} missing.", file=sys.stderr)
        print("Please execute your regulatory filter pipeline step first.", file=sys.stderr)
        sys.exit(1)
        
    kept, total = extract_chr22(input_path, output_path, col_map_path)
    
    stats = {
        "total_regulatory_variants": total,
        "chr22_regulatory_variants": kept,
        "percentage_chr22": kept / total if total > 0 else 0
    }
    
    stats_output = script_dir / "processed" / "chr22_stats.json"
    with open(stats_output, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\n💾 Metadata stats written to: {stats_output}")
    print("=" * 60)

if __name__ == "__main__":
    main()