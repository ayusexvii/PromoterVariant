#!/usr/bin/env python3
"""
GTEx eQTL Matching Module (Phase 2) - Enhanced Gene Symbol Support
"""

import gzip
import json
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def build_gtex_indexes(gtex_path: str) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Build three indexes:
    1. variant_id → record
    2. ensembl_id → records (exact)
    3. gene_symbol → records (normalized fallback)
    """
    logger.info(f"Building eQTL indexes from: {gtex_path}")
    
    var_index: Dict[str, Dict] = {}
    ensembl_index: Dict[str, List[Dict]] = {}
    symbol_index: Dict[str, List[Dict]] = {}
    total = 0
    chr22_count = 0

    try:
        with gzip.open(gtex_path, 'rt', encoding='utf-8') as f:
            header_line = f.readline().strip()
            header = [h.strip() for h in header_line.split('\t')]
            col_idx = {col: i for i, col in enumerate(header)}
            
            for line_num, line in enumerate(f, start=2):
                try:
                    fields = line.strip().split('\t')
                    if len(fields) < 5:
                        continue
                    
                    variant_id = fields[col_idx.get('variant_id', 0)]
                    gene_id_raw = fields[col_idx.get('gene_id', 1)]
                    gene_id = gene_id_raw.split('.')[0]  # ENSG...
                    
                    # Try to extract symbol if present (some GTEx versions include it)
                    gene_symbol = ""
                    if '|' in gene_id_raw:
                        gene_symbol = gene_id_raw.split('|')[-1].strip()
                    # Or take from gene_name column if available
                    elif 'gene_name' in col_idx:
                        gene_symbol = fields[col_idx['gene_name']].strip()
                    
                    slope = float(fields[col_idx.get('slope', 3)])
                    pval = float(fields[col_idx.get('pval_nominal', 4)])
                    tss_dist = int(fields[col_idx.get('tss_distance', 2)]) if 'tss_distance' in col_idx else 0
                    
                    record = {
                        'variant_id': variant_id,
                        'gene_id': gene_id,
                        'gene_symbol': gene_symbol.upper() if gene_symbol else "",
                        'slope': slope,
                        'pval_nominal': pval,
                        'tss_distance': tss_dist
                    }
                    
                    # Primary indexes
                    var_index[variant_id] = record
                    
                    if gene_id not in ensembl_index:
                        ensembl_index[gene_id] = []
                    ensembl_index[gene_id].append(record)
                    
                    if gene_symbol:
                        sym_upper = gene_symbol.upper()
                        if sym_upper not in symbol_index:
                            symbol_index[sym_upper] = []
                        symbol_index[sym_upper].append(record)
                    
                    total += 1
                    if total % 100000 == 0:
                        gc.collect()
                    if 'chr22' in variant_id or '_22_' in variant_id:
                        chr22_count += 1
                        
                except Exception:
                    continue

    except FileNotFoundError:
        raise FileNotFoundError(f"GTEx file not found: {gtex_path}")

    logger.info(f"✅ Indexed {len(var_index):,} variants")
    logger.info(f"✅ Indexed {len(ensembl_index):,} Ensembl genes")
    logger.info(f"✅ Indexed {len(symbol_index):,} gene symbols for fallback")
    return var_index, ensembl_index, symbol_index


def construct_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Construct GTEx b37 variant ID."""
    chrom = str(chrom).replace('chr', '')
    return f"chr{chrom}_{pos}_{ref}_{alt}_b37"


def normalize_gene(gene: str) -> str:
    """Normalize gene symbol for matching."""
    if not gene:
        return ""
    return gene.strip().upper().split('.')[0]  # Remove versions if any


def match_variant(
    variant: Dict[str, Any],
    var_index: Dict[str, Dict],
    ensembl_index: Dict[str, List[Dict]],
    symbol_index: Dict[str, List[Dict]]
) -> Optional[Dict]:
    """Enhanced two-stage + symbol fallback matching."""
    chrom = str(variant.get('chrom', '')).replace('chr', '')
    pos = variant.get('pos')
    ref = variant.get('ref', '')
    alt = variant.get('alt', '')
    gene_raw = variant.get('promoter_gene') or variant.get('gene', '')
    gene_norm = normalize_gene(gene_raw)

    if not chrom or not pos:
        return None

    # 1. Exact variant ID
    var_id = construct_variant_id(chrom, pos, ref, alt)
    if var_id in var_index:
        match = var_index[var_id].copy()
        match['match_type'] = 'variant_id'
        match['match_score'] = 1.0
        return match

    # 2. Ensembl ID fallback (if ClinVar has it)
    if gene_norm.startswith('ENSG'):
        if gene_norm in ensembl_index:
            candidates = ensembl_index[gene_norm]
            best = min(candidates, key=lambda x: abs(x.get('tss_distance', 999999)))
            best = best.copy()
            best['match_type'] = 'ensembl_fallback'
            best['match_score'] = 0.8
            return best

    # 3. Gene Symbol fallback (main improvement)
    if gene_norm and gene_norm in symbol_index:
        candidates = symbol_index[gene_norm]
        if candidates:
            best = min(candidates, key=lambda x: abs(x.get('tss_distance', 999999)))
            best = best.copy()
            best['match_type'] = 'gene_symbol'
            best['match_score'] = max(0.0, 1.0 - abs(best.get('tss_distance', 0)) / 10000)
            return best

    return None


def main(
    promoter_path: str = "processed/promoter_variants_chr22.txt.gz",
    gtex_path: str = "../01_Data/raw/Liver.v8.signif_variant_gene_pairs.txt.gz",
    output_path: str = "processed/training_matrix_chr22.txt.gz"
) -> None:
    """Main pipeline with enhanced gene matching."""
    logger.info("=" * 70)
    logger.info("🧬 GTEx eQTL MATCHING (Enhanced Gene Symbol Support)")
    logger.info("=" * 70)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    var_index, ensembl_index, symbol_index = build_gtex_indexes(gtex_path)

    matched_count = 0
    total = 0
    match_types: Dict[str, int] = {'variant_id': 0, 'ensembl_fallback': 0, 'gene_symbol': 0, 'no_match': 0}

    logger.info(f"Processing: {promoter_path}")

    with gzip.open(promoter_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', compresslevel=6, encoding='utf-8') as f_out:
        
        header_line = f_in.readline().strip()
        new_header = header_line + '\teqtl_slope\teqtl_pval\tmatch_type\tmatch_score\tmatched_gene\n'
        f_out.write(new_header)
        
        for line_num, line in enumerate(f_in, start=2):
            total += 1
            try:
                fields = line.strip().split('\t')
                # Flexible parsing
                variant = {
                    'chrom': fields[18].replace('chr','') if len(fields) > 18 else '',
                    'pos': int(fields[19]) if len(fields) > 19 else 0,
                    'ref': fields[3] if len(fields) > 3 else '',
                    'alt': fields[4] if len(fields) > 4 else '',
                    'promoter_gene': fields[-4] if len(fields) > 4 else ''  # promoter_gene column
                }
                
                match = match_variant(variant, var_index, ensembl_index, symbol_index)
                
                if match:
                    out_line = (
                        line.strip() +
                        f"\t{match['slope']:.6f}\t{match['pval_nominal']:.2e}\t"
                        f"{match['match_type']}\t{match['match_score']:.4f}\t"
                        f"{match.get('gene_symbol', match.get('gene_id', ''))}\n"
                    )
                    f_out.write(out_line)
                    matched_count += 1
                    match_types[match['match_type']] += 1
                else:
                    match_types['no_match'] += 1
                    
            except Exception as e:
                logger.warning(f"Error line {line_num}: {e}")
                continue

    match_rate = (matched_count / total * 100) if total > 0 else 0
    stats = {
        "stage": "eqtl_matching",
        "total_promoter_variants": total,
        "matched": matched_count,
        "match_rate_pct": round(match_rate, 2),
        "match_types": match_types,
        "expected_range": "20-50% with symbol fallback"
    }
    
    with open("processed/eqtl_match_stats.json", "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"✅ Matching complete: {matched_count}/{total} ({match_rate:.1f}%)")
    for mtype, cnt in match_types.items():
        logger.info(f"   • {mtype}: {cnt}")
    logger.info(f"💾 Output: {output_path}")


if __name__ == "__main__":
    main()