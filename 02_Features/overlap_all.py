#!/usr/bin/env python3
"""
Full-Genome Promoter Overlap: Regulatory Variants × All TSS ± 2kb Promoters
Uses PyRanges for efficient interval operations.
"""

import pyranges as pr
import pandas as pd
import gzip
import logging
import time
import gc
from pathlib import Path
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_promoters(bed_path: str = "processed/all_tss.bed") -> pr.PyRanges:
    """Load promoter regions (TSS ± 2kb)."""
    logger.info(f"📂 Loading promoters: {bed_path}")
    try:
        prom = pr.read_bed(bed_path)
        logger.info(f"✅ Loaded {len(prom):,} promoter regions across {prom.chromosomes.nunique()} chromosomes")
        return prom
    except Exception as e:
        logger.error(f"Failed to load promoters: {e}")
        raise


def load_variants(variant_path: str = "../01_Data/processed/clinvar_regulatory.txt.gz") -> Tuple[pr.PyRanges, list]:
    """Stream and load regulatory variants as PyRanges."""
    logger.info(f"📂 Loading variants: {variant_path}")
    
    rows = []
    skipped = 0
    
    try:
        with gzip.open(variant_path, 'rt', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            
            for i, line in enumerate(f):
                try:
                    fields = line.strip().split('\t')
                    if len(fields) < 20:
                        skipped += 1
                        continue
                    
                    chrom_raw = fields[18].strip()
                    pos_str = fields[19].strip()
                    
                    if chrom_raw in ['X', 'Y', 'MT'] or not pos_str.isdigit():
                        skipped += 1
                        continue
                    
                    chrom = f"chr{chrom_raw}"
                    pos = int(pos_str)
                    
                    rows.append({
                        'Chromosome': chrom,
                        'Start': pos,
                        'End': pos + 1,
                        'Ref': fields[3] if len(fields) > 3 else '',
                        'Alt': fields[4] if len(fields) > 4 else '',
                        'promoter_gene': '',  # will be filled during overlap
                        'raw_line': line.strip()
                    })
                    
                    if (i + 1) % 20000 == 0:
                        logger.info(f"   Loaded {i+1:,} variants... (skipped {skipped})")
                        
                except Exception:
                    skipped += 1
                    continue
                    
    except FileNotFoundError:
        logger.error(f"Variant file not found: {variant_path}")
        raise
    
    logger.info(f"✅ Loaded {len(rows):,} valid variants (skipped {skipped})")
    df = pd.DataFrame(rows)
    var_pr = pr.PyRanges(df)
    
    return var_pr, rows


def perform_overlap(prom: pr.PyRanges, var: pr.PyRanges) -> pr.PyRanges:
    """Perform efficient overlap using PyRanges."""
    logger.info("🔍 Performing promoter-variant overlap...")
    start_time = time.time()
    
    # Inner join: variants inside promoters
    overlap = var.join(prom, how='inner', suffix="_prom")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Overlap completed in {elapsed:.2f} seconds")
    logger.info(f"   Found {len(overlap):,} overlaps")
    
    return overlap


def save_results(overlap: pr.PyRanges, output_path: str = "processed/promoter_variants_allchr.txt.gz"):
    """Save enriched overlap results."""
    logger.info(f"💾 Saving results to: {output_path}")
    
    df = overlap.df.copy()
    
    # Calculate distance to TSS
    if 'Start_prom' in df.columns and 'End_prom' in df.columns:
        # Approximate TSS as middle or use original logic
        df['distance_to_tss'] = df['Start'] - ((df['Start_prom'] + df['End_prom']) // 2)
    else:
        df['distance_to_tss'] = 0
    
    # Rename columns for clarity
    rename_map = {
        'gene_name': 'promoter_gene',
        'TSS': 'promoter_tss',
        'strand': 'promoter_strand'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Select useful columns
    keep_cols = [col for col in [
        'Chromosome', 'Start', 'End', 'Ref', 'Alt', 'promoter_gene',
        'distance_to_tss', 'promoter_tss', 'promoter_strand'
    ] if col in df.columns]
    
    df_out = df[keep_cols].copy()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, sep='\t', index=False, compression='gzip')
    
    logger.info(f"✅ Saved {len(df_out):,} promoter-overlapping variants")
    logger.info(f"   Unique genes hit: {df_out['promoter_gene'].nunique() if 'promoter_gene' in df_out.columns else 'N/A'}")
    
    return df_out


def main():
    logger.info("=" * 90)
    logger.info("🧬 FULL GENOME PROMOTER OVERLAP PIPELINE")
    logger.info("=" * 90)
    
    # Load data
    prom = load_promoters()
    var, _ = load_variants()
    
    # Overlap
    overlap = perform_overlap(prom, var)
    
    # Save
    df_out = save_results(overlap)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total overlaps          : {len(df_out):,}")
    logger.info(f"Unique promoter genes   : {df_out.get('promoter_gene', pd.Series()).nunique():,}")
    logger.info(f"Output file             : processed/promoter_variants_allchr.txt.gz")
    
    logger.info("\n✅ FULL GENOME OVERLAP COMPLETE!")
    gc.collect()


if __name__ == "__main__":
    main()
