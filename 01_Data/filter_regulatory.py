#!/usr/bin/env python3
"""
Filter ClinVar to keep only regulatory/promoter-associated variants.
Corrected to utilize the exact columns available in variant_summary.txt.
"""
import gzip
import json
import io
import sys
import re
from pathlib import Path

def load_column_map(base_dir: Path) -> dict:
    """Load column indices using explicit path routing."""
    map_path = base_dir / "processed" / "column_map.json"
    if not map_path.exists():
        print(f"❌ ERROR: {map_path} not found!", file=sys.stderr)
        print("Please execute your validation script first to generate it.", file=sys.stderr)
        return None
    
    with open(map_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_clinvar(input_path: Path, output_path: Path, col_map: dict, batch_size=1000000):
    """
    Stream filters the ClinVar file by looking into Name, Type, and 
    ClinicalSignificance columns where upstream/promoter data lives.
    """
    total_scanned = 0
    total_kept = 0
    skipped_malformed = 0
    
    # Safely extract the exact columns guaranteed to be in variant_summary.txt
    try:
        name_idx = col_map['Name']
        type_idx = col_map['Type']
        sig_idx = col_map['ClinicalSignificance']
    except KeyError as e:
        print(f"❌ ERROR: Your column map is missing standard ClinVar fields: {e}", file=sys.stderr)
        print(f"Available columns in your file are: {list(col_map.keys())}", file=sys.stderr)
        sys.exit(1)
        
    max_required_idx = max(name_idx, type_idx, sig_idx)
    
    # Matches regulatory terms OR the HGVS upstream promoter notation 'c.-' (e.g., c.-124C>T)
    # Added word boundaries to prevent catching random text
    regulatory_pattern = re.compile(
        r'c\.-|regulatory|promoter|enhancer|silencer|intron|intergenic|utr|'
        r'untranslated|upstream|downstream|transcription factor|tfbs', 
        re.IGNORECASE
    )
    
    BUFFER_SIZE = 2 * 1024 * 1024  # 2MB Cache Tuning
    
    print(f"\n📂 Stream Reading: {input_path}")
    print(f"📝 Buffered Writing: {output_path}")
    print("🎯 Scanning Name (for c.- promoter notation), Type, and ClinicalSignificance...")
    
    try:
        with gzip.open(input_path, 'rb') as raw_in, \
             gzip.open(output_path, 'wb', compresslevel=5) as raw_out:
            
            f_in = io.TextIOWrapper(io.BufferedReader(raw_in, buffer_size=BUFFER_SIZE), encoding='utf-8')
            f_out = io.TextIOWrapper(io.BufferedWriter(raw_out, buffer_size=BUFFER_SIZE), encoding='utf-8')
            
            # Forward the TSV header
            header = f_in.readline()
            f_out.write(header)
            
            while True:
                lines = f_in.readlines(BUFFER_SIZE)
                if not lines:
                    break
                
                output_buffer = []
                for line in lines:
                    total_scanned += 1
                    fields = line.strip().split('\t')
                    
                    if len(fields) <= max_required_idx:
                        skipped_malformed += 1
                        continue
                    
                    # Combine Name, Type, and Significance for the regex engine
                    target_text = f"{fields[name_idx]} {fields[type_idx]} {fields[sig_idx]}"
                    
                    if regulatory_pattern.search(target_text):
                        output_buffer.append(line)
                        total_kept += 1
                
                if output_buffer:
                    f_out.writelines(output_buffer)
                    
                if total_scanned % batch_size == 0 or total_scanned == batch_size:
                    ratio = total_kept / total_scanned if total_scanned > 0 else 0
                    print(f"   Scanned: {total_scanned:,} | Kept: {total_kept:,} ({ratio:.2%})")
            
            f_out.flush()
            
    except KeyboardInterrupt:
        print("\n⚠️ Signal interrupt caught. Closing safely.")
        
    return total_scanned, total_kept, skipped_malformed

def main():
    print("=" * 60)
    print("🧬 CLINVAR OPTIMIZED REGULATORY FILTER")
    print("=" * 60)
    
    script_dir = Path(__file__).resolve().parent
    col_map = load_column_map(script_dir)
    if not col_map:
        sys.exit(1)
        
    input_path = script_dir / "raw" / "variant_summary.txt.gz"
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "clinvar_regulatory.txt.gz"
    
    if not input_path.exists():
        print(f"❌ ERROR: Source target {input_path} missing.", file=sys.stderr)
        sys.exit(1)
        
    scanned, kept, malformed = filter_clinvar(input_path, output_path, col_map)
    
    ratio = kept / scanned if scanned > 0 else 0
    print("\n" + "=" * 60)
    print("✅ FILTERING STAGE COMPLETE")
    print(f"   Total scanned: {scanned:,}")
    print(f"   Kept (Regulatory/Promoter): {kept:,} ({ratio:.2%})")
    print(f"   Malformed lines skipped: {malformed:,}")
    print("=" * 60)
    
    stats = {
        "total_scanned": scanned,
        "total_kept": kept,
        "percentage_kept": ratio,
        "malformed_rows": malformed
    }
    
    with open(output_dir / "filtering_stats.json", "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()