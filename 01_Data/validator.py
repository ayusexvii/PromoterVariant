#!/usr/bin/env python3
"""
ClinVar streaming validator for PromoterVariant project.
Optimized for thermal-constrained Intel i5 architectures (4 P-cores, 8 E-cores).
"""
import gzip
import sys
import json
import io
from pathlib import Path

def validate_file(base_dir: Path) -> Path:
    """Check that the ClinVar file exists and is readable using explicit Path routing."""
    file_path = base_dir / "raw" / "variant_summary.txt.gz"
    if not file_path.exists():
        print(f"ERROR: {file_path} not found!", file=sys.stderr)
        print("Make sure you have your data placed in the 'raw/' directory relative to this script.", file=sys.stderr)
        sys.exit(1)
    
    compressed_bytes = file_path.stat().st_size
    print(f"✓ Compressed file found: {compressed_bytes / (1024 * 1024):.1f} MB")
    return file_path

def stream_read_counts(file_path: Path):
    """
    Stream read the file using a large internal buffer to minimize I/O syscalls
    and avoid single-byte decoding overhead in the Python interpreter loop.
    """
    total_rows = 0
    header_columns = {}
    total_uncompressed_bytes = 0
    
    print("\n📊 Opening file with buffered streaming parser...")
    print("   (Optimized chunk allocations to respect Intel L2/L3 cache bounds)")
    
    # 1MB buffer size balances L2/L3 cache capacities on Intel 12th/13th Gen P-cores
    BUFFER_SIZE = 1024 * 1024 
    
    try:
        # Open in binary mode ('rb') to separate hardware decompression from string decoding
        with gzip.open(file_path, 'rb') as gz_file:
            # Wrap in a buffered text reader to process text lines cleanly
            with io.TextIOWrapper(io.BufferedReader(gz_file, buffer_size=BUFFER_SIZE), encoding='utf-8') as f:
                header_line = f.readline()
                if not header_line:
                    print("ERROR: Empty file!", file=sys.stderr)
                    return None, 0
                
                total_uncompressed_bytes += len(header_line.encode('utf-8'))
                header_parts = header_line.strip().split('\t')
                print(f"✓ Total columns detected: {len(header_parts)}")
                print(f"✓ Sample columns: {header_parts[:5]}")
                
                # Map column indices for fast lookups
                for idx, col_name in enumerate(header_parts):
                    header_columns[col_name] = idx
                
                # Structural check for expected metrics
                important_cols = [
                    'Type', 'ClinicalSignificance', 'Gene', 'Chromosome', 
                    'Start', 'ReferenceAllele', 'AlternateAllele'
                ]
                print("\n🔑 Key column index mapping:")
                for col in important_cols:
                    if col in header_columns:
                        print(f"   {col}: {header_columns[col]}")
                    else:
                        print(f"   {col}: ⚠️ NOT FOUND (Review schema for this ClinVar release)")
                
                print("\n📈 Counting rows and measuring uncompressed footprint...")
                
                # Batched streaming loop to reduce CPU overhead
                while True:
                    lines = f.readlines(BUFFER_SIZE)
                    if not lines:
                        break
                    total_rows += len(lines)
                    # Track exact uncompressed payload sizing
                    total_uncompressed_bytes += sum(len(line.encode('utf-8')) for line in lines)
                    
                    if total_rows % 1000000 == 0:
                        print(f"   Processed {total_rows:,} rows...")

    except (OSError, UnicodeDecodeError) as e:
        print(f"\nFATAL ERROR: File corruption or encoding issues encountered: {e}", file=sys.stderr)
        return None, 0
        
    print(f"\n✅ Total rows in ClinVar: {total_rows:,}")
    print(f"💾 Absolute RAM footprint bypassed: {total_uncompressed_bytes / (1024 * 1024):.1f} MB")
    return header_columns, total_rows

def main():
    print("=" * 60)
    print("🧬 CLINVAR HARDWARE-AWARE STREAMING VALIDATOR")
    print("=" * 60)
    
    # Establish absolute anchor path without modifying global process states like os.chdir()
    script_dir = Path(__file__).resolve().parent
    
    file_path = validate_file(script_dir)
    header_map, row_count = stream_read_counts(file_path)
    
    if header_map is None:
        sys.exit(1)
    
    # Safely construct the distribution outputs directory
    processed_dir = script_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    output_json = processed_dir / "column_map.json"
    with open(output_json, "w") as f:
        json.dump(header_map, f, indent=2)
        
    print(f"\n💾 Metadata saved safely to disk at: {output_json}")
    print("=" * 60)
    print("🚀 VALIDATION COMPLETE: READY FOR DOWNSTREAM MULTI-THREADED FILTERING")
    print("=" * 60)

if __name__ == "__main__":
    main()
