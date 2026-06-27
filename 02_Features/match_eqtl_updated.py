#!/usr/bin/env python3
"""
GTEx eQTL Matching with Gene Symbol Mapping (Optimized)
Matches promoter variants (GRCh38) to GTEx Liver eQTLs (GRCh37).
"""

import gzip
import json
import gc
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_gene_symbol_map(map_path: str = "gene_ensembl_map.txt") -> Dict[str, str]:
    """Load gene symbol → Ensembl ID mapping."""
    symbol_map: Dict[str, str] = {}
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    symbol_map[parts[0].strip()] = parts[1].strip()
        logger.info(f"✅ Loaded {len(symbol_map):,} gene symbol → Ensembl mappings")
    except FileNotFoundError:
        logger.error(f"Gene mapping file not found: {map_path}")
        raise
    return symbol_map


def load_gtex_data(gtex_path: str) -> Tuple[Dict[str, Dict], Dict[str, list]]:
    """Load GTEx into memory (acceptable size ~11MB compressed)."""
    logger.info(f"📂 Loading GTEx data: {gtex_path}")
    
    by_variant: Dict[str, Dict] = {}
    by_gene: Dict[str, list] = defaultdict(list)
    total = 0

    try:
        with gzip.open(gtex_path, 'rt', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            col_idx = {col: i for i, col in enumerate(header)}
            
            for i, line in enumerate(f):
                try:
                    fields = line.strip().split('\t')
                    if len(fields) < 5:
                        continue

                    variant_id = fields[col_idx['variant_id']]
                    gene_id = fields[col_idx['gene_id']].split('.')[0]
                    tss_distance = int(fields[col_idx['tss_distance']])
                    slope = float(fields[col_idx['slope']])
                    pval = float(fields[col_idx['pval_nominal']])

                    eqtl_data = {
                        'slope': slope,
                        'pval': pval,
                        'gene_id': gene_id,
                        'tss_distance': tss_distance
                    }

                    by_variant[variant_id] = eqtl_data
                    by_gene[gene_id].append(eqtl_data)

                    total += 1
                    if total % 500000 == 0:
                        logger.info(f"Loaded {total:,} eQTLs...")
                        gc.collect()
                except (ValueError, IndexError, KeyError):
                    continue

    except FileNotFoundError:
        logger.error(f"GTEx file not found: {gtex_path}")
        raise

    logger.info(f"✅ GTEx loaded: {len(by_variant):,} variants | {len(by_gene):,} genes")
    return by_variant, by_gene


def build_clinvar_variant_id(fields: list) -> str:
    """Safely build GTEx-style variant ID."""
    try:
        chrom = fields[18].replace('chr', '') if len(fields) > 18 else ''
        pos = fields[19] if len(fields) > 19 else ''
        ref = fields[3] if len(fields) > 3 else ''   # Usually REF is earlier
        alt = fields[4] if len(fields) > 4 else ''   # Usually ALT is earlier
        
        alt = alt.split(',')[0] if ',' in alt else alt
        ref = ref.split(',')[0] if ',' in ref else ref
        
        return f"chr{chrom}_{pos}_{ref}_{alt}_b37"
    except Exception:
        return ""


def match_variants(
    variant_path: str,
    gtex_by_variant: Dict,
    gtex_by_gene: Dict,
    symbol_map: Dict[str, str],
    output_path: str
) -> Tuple[int, int, int, int]:
    """Main matching logic."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    total = exact_matched = gene_matched = unmatched = 0

    logger.info(f"Processing promoter variants: {variant_path}")

    with gzip.open(variant_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', compresslevel=6, encoding='utf-8') as f_out:
        
        header = f_in.readline().strip()
        new_header = header + '\teqtl_slope\teqtl_pval\teqtl_gene_id\tmatch_type\n'
        f_out.write(new_header)
        
        for line_num, line in enumerate(f_in, start=2):
            total += 1
            try:
                fields = line.strip().split('\t')
                gene_symbol = fields[-4] if len(fields) >= 4 else ''
                distance_to_tss = int(fields[-1]) if len(fields) > 0 and fields[-1].lstrip('-').isdigit() else 0
                
                variant_id = build_clinvar_variant_id(fields)
                
                if variant_id and variant_id in gtex_by_variant:
                    eq = gtex_by_variant[variant_id]
                    match_type = 'exact'
                    exact_matched += 1
                else:
                    ensembl_id = symbol_map.get(gene_symbol)
                    if ensembl_id and ensembl_id in gtex_by_gene:
                        candidates = gtex_by_gene[ensembl_id]
                        # Closest TSS distance
                        best = min(candidates, key=lambda x: abs(x['tss_distance'] - distance_to_tss))
                        eq = best
                        match_type = 'gene_fallback'
                        gene_matched += 1
                    else:
                        eq = None
                        match_type = 'unmatched'
                        unmatched += 1
                
                if eq:
                    out_line = (line.strip() + 
                              f"\t{eq['slope']}\t{eq['pval']}\t{eq['gene_id']}\t{match_type}\n")
                else:
                    out_line = line.strip() + "\tNA\tNA\tNA\tunmatched\n"
                f_out.write(out_line)
                
                if total % 100 == 0:
                    logger.info(f"Processed {total:,} | Exact: {exact_matched} | Gene: {gene_matched}")
                    
            except Exception as e:
                logger.warning(f"Error on line {line_num}: {e}")
                continue

    gc.collect()
    return exact_matched, gene_matched, unmatched, total


def main():
    logger.info("=" * 80)
    logger.info("🧬 GTEx eQTL MATCHING PIPELINE - OPTIMIZED")
    logger.info("=" * 80)

    symbol_map = load_gene_symbol_map("gene_ensembl_map.txt")
    
    gtex_by_variant, gtex_by_gene = load_gtex_data(
        "../01_Data/raw/Liver.v8.signif_variant_gene_pairs.txt.gz"
    )
    
    exact, gene_fb, unmatched, total = match_variants(
        variant_path="processed/promoter_variants_chr22.txt.gz",
        gtex_by_variant=gtex_by_variant,
        gtex_by_gene=gtex_by_gene,
        symbol_map=symbol_map,
        output_path="processed/training_matrix_chr22.txt.gz"
    )
    
    stats = {
        "total_variants": total,
        "exact_matches": exact,
        "gene_fallback_matches": gene_fb,
        "unmatched": unmatched,
        "total_matched": exact + gene_fb,
        "match_rate_pct": round((exact + gene_fb) / total * 100, 2) if total > 0 else 0
    }
    
    with open("processed/eqtl_match_stats.json", "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("MATCHING SUMMARY")
    logger.info(f"Total variants          : {total:,}")
    logger.info(f"Exact matches           : {exact:,}")
    logger.info(f"Gene fallback matches   : {gene_fb:,}")
    logger.info(f"Total matched           : {exact + gene_fb:,} ({stats['match_rate_pct']}%)")
    logger.info("="*60)
    logger.info("💾 Output: processed/training_matrix_chr22.txt.gz")


if __name__ == "__main__":
    main()