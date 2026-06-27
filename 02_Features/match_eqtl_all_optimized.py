#!/usr/bin/env python3
"""
Full-Genome GTEx Matching - Optimized & Memory-Aware
"""

import gzip
import pandas as pd
import logging
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_gtex_index(gtex_path: str = "../01_Data/raw/Liver.v8.signif_variant_gene_pairs.txt.gz") -> Dict[str, List[dict]]:
    """Build gene → list of eQTL records (memory efficient)."""
    logger.info(f"📂 Building GTEx index from: {gtex_path}")
    
    gene_to_eqtls: Dict[str, List[dict]] = defaultdict(list)
    total = 0

    with gzip.open(gtex_path, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_idx = {col: i for i, col in enumerate(header)}
        
        for i, line in enumerate(f):
            try:
                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue
                
                gene_id = fields[col_idx.get('gene_id', 1)].split('.')[0]
                slope = float(fields[col_idx.get('slope', 7)])
                pval = float(fields[col_idx.get('pval_nominal', 6)])
                
                gene_to_eqtls[gene_id].append({
                    'slope': slope,
                    'pval': pval
                })
                
                total += 1
                if total % 500000 == 0:
                    logger.info(f"   Indexed {total:,} eQTLs...")
                    gc.collect()
                    
            except (ValueError, IndexError, KeyError):
                continue

    logger.info(f"✅ GTEx index ready: {len(gene_to_eqtls):,} genes")
    return gene_to_eqtls


def load_gene_mapping(map_path: str = "gene_ensembl_map.txt") -> Dict[str, List[str]]:
    """Symbol → list of Ensembl IDs."""
    symbol_to_ens: Dict[str, List[str]] = defaultdict(list)
    
    with open(map_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                symbol_to_ens[parts[0].strip()].append(parts[1].strip())
    
    logger.info(f"✅ Loaded mapping for {len(symbol_to_ens):,} gene symbols")
    return symbol_to_ens


def main():
    logger.info("=" * 90)
    logger.info("🧬 FULL GENOME GTEx MATCHING (Optimized)")
    logger.info("=" * 90)
    
    # 1. Load indexes
    gene_to_eqtls = load_gtex_index()
    symbol_to_ens = load_gene_mapping()
    
    # 2. Load promoter variants
    promoter_path = "processed/promoter_variants_allchr.txt.gz"
    logger.info(f"📂 Loading promoter variants: {promoter_path}")
    
    df = pd.read_csv(promoter_path, sep='\t', compression='gzip', low_memory=False)
    logger.info(f"✅ Loaded {len(df):,} promoter-variant overlaps")
    
    # 3. Matching
    logger.info("🔍 Matching to GTEx eQTLs...")
    matched_records = []
    processed = 0
    
    for _, row in df.iterrows():
        gene_symbol = str(row.get('promoter_gene', row.get('gene_name', ''))).strip()
        if not gene_symbol:
            processed += 1
            continue
        
        ens_ids = symbol_to_ens.get(gene_symbol, [])
        
        for ens in ens_ids:
            if ens in gene_to_eqtls:
                for eq in gene_to_eqtls[ens]:
                    matched_records.append({
                        'chrom': row.get('Chromosome', row.get('chrom', '')),
                        'pos': row.get('Start', row.get('pos', 0)),
                        'ref': row.get('Ref', ''),
                        'alt': row.get('Alt', ''),
                        'promoter_gene': gene_symbol,
                        'distance_to_tss': row.get('distance_to_tss', 0),
                        'eqtl_slope': eq['slope'],
                        'eqtl_pval': eq['pval']
                    })
        
        processed += 1
        if processed % 10000 == 0:
            logger.info(f"   Processed {processed:,}/{len(df):,} overlaps...")
            gc.collect()
    
    # 4. Create final DataFrame
    out_df = pd.DataFrame(matched_records)
    logger.info(f"✅ Generated {len(out_df):,} matched records")
    
    # Optional: Keep best (lowest p-value) per variant-gene pair
    if len(out_df) > 100000:
        logger.info("Deduplicating (keeping best eQTL per variant-gene)...")
        out_df = out_df.sort_values('eqtl_pval').drop_duplicates(
            subset=['chrom', 'pos', 'promoter_gene'], keep='first'
        )
        logger.info(f"   After dedup: {len(out_df):,} rows")
    
    # 5. Save
    output_path = "processed/training_matrix_allchr.csv.gz"
    out_df.to_csv(output_path, index=False, compression='gzip')
    logger.info(f"💾 Final training matrix saved: {output_path}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 FULL GENOME MATCH SUMMARY")
    logger.info("="*80)
    logger.info(f"Input overlaps          : {len(df):,}")
    logger.info(f"Matched records         : {len(out_df):,}")
    logger.info(f"Unique genes            : {out_df['promoter_gene'].nunique() if 'promoter_gene' in out_df.columns else 0:,}")
    logger.info(f"Unique variants         : {out_df['pos'].nunique() if 'pos' in out_df.columns else 0:,}")
    logger.info("\n✅ FULL GENOME GTEx MATCHING COMPLETE!")


if __name__ == "__main__":
    main()
