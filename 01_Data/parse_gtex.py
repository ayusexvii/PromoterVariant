#!/usr/bin/env python3
"""
Parse GTEx V8 Liver eQTL variant IDs to coordinates.
Uses right-side anchor splitting to protect non-standard chromosome scaffolds.
"""
import gzip
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 HARDWARE-ACCELERATED GTEX COORD PARSING ENGINE")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "raw" / "Liver.v8.signif_variant_gene_pairs.txt.gz"
    output_dir = script_dir / "processed"
    output_path = output_dir / "liver_eqtl_parsed.txt"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not input_path.exists():
        print(f"❌ ERROR: Missing target source file element: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Stream-loading compressed archive: {input_path.name} ({input_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"📝 Initializing target destination  : {output_path}")
    
    total = 0
    skipped = 0
    
    # Using larger, optimized file buffering system channels
    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8', buffering=64*1024) as f_out:
        # Write clean table columns layout
        f_out.write("chrom\tpos\tref\talt\tgene_id\tslope\tpval\n")
        
        # Ingest and verify header tracks 
        header = f_in.readline().strip()
        header_cols = header.split('\t')
        print(f"✅ Verified source table shape width: {len(header_cols)} metrics columns.")
        
        for line in f_in:
            fields = line.strip().split('\t')
            
            # Avoid processing partial or truncated network lines
            if len(fields) < 8:
                skipped += 1
                continue
                
            variant_id = fields[0]
            parts = variant_id.split('_')
            
            # FIXED: Split from the right to protect scaffold names with internal underscores
            if len(parts) >= 5:
                alt = parts[-2]
                ref = parts[-3]
                pos = parts[-4]
                chrom = "_".join(parts[:-4]).replace('chr', '')
            else:
                skipped += 1
                continue
                
            # Verify that position values are strictly numerical characters
            if not pos.isdigit():
                skipped += 1
                continue
                
            # Isolate stable gene ID without variable dot suffix modifications
            gene_id = fields[1].split('.')[0]
            
            pval = fields[6]
            slope = fields[7]
            
            f_out.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{gene_id}\t{slope}\t{pval}\n")
            total += 1
            
            if total % 500000 == 0:
                print(f"   Parsed {total:,} lines successfully...")
                
    print(f"\n" + "=" * 70)
    print("📊 DATA INGESTION PARSING REPORT COMPLETED")
    print("=" * 70)
    print(f"   Total valid records compiled : {total:,}")
    print(f"   Skipped/Malformed lines drops: {skipped:,}")
    print(f"   Destination path target lock : {output_path}")
    print(f"   Output file footprint mass   : {output_path.stat().st_size / (1024**2):.1f} MB")
    print("=" * 70)

if __name__ == "__main__":
    main()
