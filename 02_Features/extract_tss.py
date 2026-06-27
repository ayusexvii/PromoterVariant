#!/usr/bin/env python3
"""
Extract Transcription Start Sites (TSS) from GENCODE annotation.
Creates promoter regions: TSS ± 2kb. Fixed for 0-based BED edge conditions.
"""
import gzip
import io
import sys
import re
from pathlib import Path

def extract_tss_from_gencode(gtf_path: Path, output_path: Path, chromosome="22", promoter_window=2000):
    """
    Extract TSS from GENCODE GTF file and generate a validated 0-based BED file.
    Engineered to eliminate negative index errors on edge-proximal genes.
    """
    kept_genes = 0
    total_genes = 0
    skipped_not_protein = 0
    skipped_no_name = 0
    
    # Pre-compile structural regex patterns for fast execution inside the main loop
    biotype_pattern = re.compile(r'gene_(?:type|biotype)\s+"protein_coding"')
    name_pattern = re.compile(r'gene_name\s+"([^"]+)"')
    
    # GENCODE uses the 'chr' identifier prefix uniformly
    chrom_name = f"chr{chromosome}"
    BUFFER_SIZE = 2 * 1024 * 1024  # 2MB optimized system buffer
    
    print(f"📂 Stream Reading: {gtf_path}")
    print(f"📝 Buffered Writing: {output_path}")
    print(f"🎯 Target Target: Protein-coding features on {chrom_name}")
    
    try:
        with gzip.open(gtf_path, 'rb') as raw_in, \
             open(output_path, 'w', encoding='utf-8', buffering=BUFFER_SIZE) as f_out:
            
            f_in = io.TextIOWrapper(io.BufferedReader(raw_in, buffer_size=BUFFER_SIZE), encoding='utf-8')
            
            # Formal 6-column BED schema layout tracking track metadata header
            f_out.write("#chrom\tstart\tend\tgene_name\ttss\tstrand\n")
            
            for line in f_in:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                # Filter out irrelevant chromosomes and non-gene records quickly
                if fields[0] != chrom_name or fields[2] != 'gene':
                    continue
                    
                total_genes += 1
                info_block = fields[8]
                
                # Validate protein coding designation using pre-compiled regex matrix
                if not biotype_pattern.search(info_block):
                    skipped_not_protein += 1
                    continue
                
                # Extract clean gene name designations
                name_match = name_pattern.search(info_block)
                if not name_match:
                    skipped_no_name += 1
                    continue
                gene_name = name_match.group(1)
                
                # Parse biological coordinates (GTF is 1-based inclusive structure)
                start_coord = int(fields[3])
                end_coord = int(fields[4])
                strand = fields[6]
                
                # Biological TSS mapping rules
                tss = start_coord if strand == '+' else end_coord
                
                # CRITICAL FIX: Convert from 1-based GTF to 0-based BED first, THEN clamp to floor zero
                # Spans from (TSS - 1) - window up to TSS + window
                bed_start = max(0, (tss - 1) - promoter_window)
                bed_end = tss + promoter_window
                
                f_out.write(f"{chrom_name}\t{bed_start}\t{bed_end}\t{gene_name}\t{tss}\t{strand}\n")
                kept_genes += 1
                
                if kept_genes % 200 == 0:
                    print(f"   Mapped {kept_genes} protein-coding promoter regions...")
                    
    except (OSError, UnicodeDecodeError) as e:
        print(f"❌ Pipeline Failure during file extraction processing: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"\n📊 Extraction Breakdown Summary:")
    print(f"   Total source genome entities encountered: {total_genes:,}")
    print(f"   Protein-coding loci successfully preserved: {kept_genes:,}")
    print(f"   Non-protein-coding types bypassed: {skipped_not_protein:,}")
    print(f"   Loci missing string identification names: {skipped_no_name:,}")
    
    return kept_genes

def main():
    print("=" * 70)
    print("🧬 HARDWARE-BUFFERED GENCODE TSS PROMOTER EXTRACTOR")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    
    # Establish explicit path anchors inside the primary working root structure
    gtf_path = script_dir / "raw" / "gencode.v46.basic.annotation.gtf.gz"
    output_bed = script_dir / "processed" / "chr22_tss.bed"
    
    # Create target processed directories dynamically
    output_bed.parent.mkdir(exist_ok=True)
    
    if not gtf_path.exists():
        print(f"❌ ERROR: GENCODE target annotation vector path missing at: {gtf_path}", file=sys.stderr)
        print("Please place the target file inside your 'raw/' staging path or verify symlinks.", file=sys.stderr)
        sys.exit(1)
        
    kept_genes = extract_tss_from_gencode(
        gtf_path=gtf_path,
        output_path=output_bed,
        chromosome="22",
        promoter_window=2000
    )
    
    print("\n" + "=" * 70)
    print(f"✅ PROCESS COMPLETE: {kept_genes} clean promoter maps exported.")
    print(f"   Target Deliverable Location: {output_bed}")
    print("=" * 70)

if __name__ == "__main__":
    main()