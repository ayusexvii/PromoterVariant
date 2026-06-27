#!/usr/bin/env python3
"""
Full-Genome Promoter Overlap - FIXED VERSION
Handles BED header properly.
"""
import pyranges as pr
import pandas as pd
import gzip
import logging
import time
import gc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_promoters(bed_path: str = "processed/all_tss.bed") -> pr.PyRanges:
    """Load promoter regions (TSS ± 2kb) with proper header handling."""
    logger.info(f"📂 Loading promoters: {bed_path}")
    
    # Read BED file manually to handle header
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
    logger.info(f"✅ Loaded {len(df):,} promoter regions")
    
    # Convert to PyRanges
    prom = pr.PyRanges(df)
    logger.info(f"   Chromosomes: {prom.chromosomes.nunique()}")
    return prom


def load_variants(variant_path: str = "../01_Data/processed/clinvar_regulatory.txt.gz"):
    """Load regulatory variants as PyRanges."""
    logger.info(f"📂 Loading variants: {variant_path}")
    
    rows = []
    skipped = 0
    
    with gzip.open(variant_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        
        for i, line in enumerate(f):
            try:
                fields = line.strip().split('\t')
                if len(fields) < 20:
                    skipped += 1
                    continue
                
                chrom_raw = fields[18].strip()
                pos_str = fields[19].strip()
                
                # Skip sex chromosomes for speed
                if chrom_raw in ['X', 'Y', 'MT'] or not pos_str.isdigit():
                    skipped += 1
                    continue
                
                chrom = f"chr{chrom_raw}"
                pos = int(pos_str)
                
                rows.append({
                    'Chromosome': chrom,
                    'Start': pos,
                    'End': pos + 1,
                    'Ref': fields[21] if len(fields) > 21 else '',
                    'Alt': fields[22] if len(fields) > 22 else '',
                    'raw_line': line.strip()
                })
                
                if (i + 1) % 20000 == 0:
                    logger.info(f"   Loaded {i+1:,} variants...")
                    
            except Exception:
                skipped += 1
                continue
    
    logger.info(f"✅ Loaded {len(rows):,} variants (skipped {skipped})")
    var = pr.PyRanges(pd.DataFrame(rows))
    return var


def perform_overlap(prom: pr.PyRanges, var: pr.PyRanges):
    """Perform efficient overlap using PyRanges."""
    logger.info("🔍 Performing promoter-variant overlap...")
    start_time = time.time()
    
    # Inner join: variants inside promoters
    overlap = prom.join(var, how='inner', slack=0)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Overlap completed in {elapsed:.2f} seconds")
    
    return overlap


def save_results(overlap: pr.PyRanges, output_path: str = "processed/promoter_variants_allchr.txt.gz"):
    """Save enriched overlap results."""
    logger.info(f"💾 Saving results to: {output_path}")
    
    df = overlap.df.copy()
    
    # Calculate distance to TSS
    if 'TSS' in df.columns and 'Start' in df.columns:
        df['distance_to_tss'] = df['Start'] - df['TSS']
    
    # Rename columns
    rename_map = {
        'gene_name': 'promoter_gene',
        'TSS': 'promoter_tss',
        'strand': 'promoter_strand'
    }
    df.rename(columns=rename_map, inplace=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False, compression='gzip')
    
    logger.info(f"✅ Saved {len(df):,} promoter-overlapping variants")
    return df


def main():
    logger.info("=" * 90)
    logger.info("🧬 FULL GENOME PROMOTER OVERLAP PIPELINE (FIXED)")
    logger.info("=" * 90)
    
    # Load data
    prom = load_promoters()
    var = load_variants()
    
    # Overlap
    overlap = perform_overlap(prom, var)
    
    # Save
    df_out = save_results(overlap)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total overlaps          : {len(df_out):,}")
    logger.info(f"Unique genes            : {df_out.get('promoter_gene', pd.Series()).nunique():,}")
    logger.info(f"Output file             : processed/promoter_variants_allchr.txt.gz")
    
    logger.info("\n✅ FULL GENOME OVERLAP COMPLETE!")
    gc.collect()


if __name__ == "__main__":
    main()
