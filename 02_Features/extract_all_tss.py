#!/usr/bin/env python3
"""
Extract TSS ± 2kb promoters for ALL protein-coding genes from GENCODE.
Output: BED file ready for promoter overlap across the full genome.
"""

import gzip
import re
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_all_promoters(
    gtf_path: str = "../01_Data/raw/gencode.v46.basic.annotation.gtf.gz",
    output_path: str = "processed/all_tss.bed"
):
    """Extract protein-coding gene promoters (TSS ± 2kb)."""
    logger.info("=" * 80)
    logger.info("🧬 EXTRACTING ALL PROTEIN-CODING PROMOTERS")
    logger.info("=" * 80)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    gene_count = 0
    protein_coding_count = 0
    chrom_counter = Counter()
    
    # Regex for fast attribute extraction
    gene_name_re = re.compile(r'gene_name "([^"]+)"')
    gene_type_re = re.compile(r'gene_type "([^"]+)"')
    gene_biotype_re = re.compile(r'gene_biotype "([^"]+)"')
    
    logger.info(f"📂 Reading: {gtf_path}")
    
    with gzip.open(gtf_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # BED header
        f_out.write("#chrom\tstart\tend\tgene_name\ttss\tstrand\n")
        
        for line_num, line in enumerate(f_in, 1):
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            
            gene_count += 1
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            info = fields[8]
            
            # Check protein-coding
            gene_type_match = gene_type_re.search(info)
            gene_biotype_match = gene_biotype_re.search(info)
            
            gene_type = gene_type_match.group(1) if gene_type_match else ""
            gene_biotype = gene_biotype_match.group(1) if gene_biotype_match else ""
            
            if gene_type != "protein_coding" and gene_biotype != "protein_coding":
                continue
            
            # Extract gene name
            gene_name_match = gene_name_re.search(info)
            if not gene_name_match:
                continue
            gene_name = gene_name_match.group(1)
            
            # Determine TSS
            tss = start if strand == '+' else end
            
            # Promoter: TSS ± 2kb
            prom_start = max(0, tss - 2000)
            prom_end = tss + 2000
            
            # Write to BED
            f_out.write(f"{chrom}\t{prom_start}\t{prom_end}\t{gene_name}\t{tss}\t{strand}\n")
            
            protein_coding_count += 1
            chrom_counter[chrom] += 1
            
            if protein_coding_count % 2000 == 0:
                logger.info(f"   Extracted {protein_coding_count:,} promoters...")
    
    logger.info(f"\n✅ Total genes scanned     : {gene_count:,}")
    logger.info(f"✅ Protein-coding promoters: {protein_coding_count:,}")
    logger.info(f"💾 Output saved: {output_path}")
    
    # Chromosome summary
    logger.info("\n📊 Chromosome Distribution:")
    for chrom, count in sorted(chrom_counter.items(), key=lambda x: (x[0].replace('chr','').isdigit() and int(x[0].replace('chr','')) or 99, x[0])):
        logger.info(f"   {chrom:<6}: {count:,} promoters")
    
    logger.info("\n✅ ALL TSS EXTRACTION COMPLETE!")
    return protein_coding_count


if __name__ == "__main__":
    extract_all_promoters()
