#!/usr/bin/env python3
"""
Full-Genome Promoter Overlap using PyRanges - SIMPLIFIED
"""
import pyranges as pr
import pandas as pd
import gzip
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_promoters(bed_path="processed/all_tss.bed"):
    """Load promoters from BED file."""
    logger.info(f"📂 Loading promoters: {bed_path}")
    
    # Read manually to handle header
    rows = []
    with open(bed_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                rows.append({
                    'Chromosome': parts[0],
                    'Start': int(parts[1]),
                    'End': int(parts[2]),
                    'gene_name': parts[3],
                    'TSS': int(parts[4]),
                    'strand': parts[5]
                })
    
    df = pd.DataFrame(rows)
    prom = pr.PyRanges(df)
    logger.info(f"✅ Loaded {len(prom):,} promoters")
    return prom

def load_variants(variant_path="../01_Data/processed/clinvar_regulatory.txt.gz"):
    """Load variants as PyRanges."""
    logger.info(f"📂 Loading variants: {variant_path}")
    
    rows = []
    with gzip.open(variant_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        for i, line in enumerate(f):
            fields = line.strip().split('\t')
            if len(fields) < 20:
                continue
            
            chrom_raw = fields[18].strip()
            pos_str = fields[19].strip()
            
            # Skip sex chromosomes
            if chrom_raw in ['X', 'Y', 'MT'] or not pos_str.isdigit():
                continue
            
            chrom = f"chr{chrom_raw}"
            pos = int(pos_str)
            
            rows.append({
                'Chromosome': chrom,
                'Start': pos,
                'End': pos + 1,
                'Ref': fields[21] if len(fields) > 21 else '',
                'Alt': fields[22] if len(fields) > 22 else '',
                'Gene': fields[4] if len(fields) > 4 else '',
                'Name': fields[2] if len(fields) > 2 else '',
                'Type': fields[1] if len(fields) > 1 else '',
                'ClinSig': fields[6] if len(fields) > 6 else '',
                'raw_line': line.strip()
            })
            
            if (i + 1) % 20000 == 0:
                logger.info(f"   Loaded {i+1:,} variants...")
    
    var = pr.PyRanges(pd.DataFrame(rows))
    logger.info(f"✅ Loaded {len(var):,} variants")
    return var

def main():
    logger.info("=" * 80)
    logger.info("🧬 FULL GENOME OVERLAP (PYRANGES)")
    logger.info("=" * 80)
    
    # Load data
    prom = load_promoters()
    var = load_variants()
    
    # Overlap: variant inside promoter
    logger.info("🔍 Performing overlap...")
    start = time.time()
    
    overlap = prom.join(var, how='inner', slack=0)
    
    elapsed = time.time() - start
    logger.info(f"✅ Overlap completed in {elapsed:.2f} seconds")
    
    # Convert to DataFrame
    df = overlap.df.copy()
    logger.info(f"✅ Found {len(df):,} overlaps")
    
    # Calculate distance to TSS
    df['distance_to_tss'] = df['Start'] - df['TSS']
    
    # Save
    df.to_csv('processed/promoter_variants_allchr.txt.gz', sep='\t', index=False, compression='gzip')
    
    logger.info(f"💾 Saved to: processed/promoter_variants_allchr.txt.gz")
    logger.info(f"   Unique genes: {df['gene_name'].nunique():,}")
    logger.info(f"   Unique variants: {df['Start'].nunique():,}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 OVERLAP SUMMARY")
    print("=" * 80)
    print(f"Promoters loaded     : {len(prom):,}")
    print(f"Variants loaded      : {len(var):,}")
    print(f"Overlaps found       : {len(df):,}")
    print(f"Unique genes hit     : {df['gene_name'].nunique():,}")
    print(f"Unique variants hit  : {df['Start'].nunique():,}")

if __name__ == "__main__":
    main()
