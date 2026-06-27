#!/usr/bin/env python3
"""
Promoter-Variant Overlap Module (Phase 1)
Memory-efficient streaming overlap of ClinVar regulatory variants with TSS ± 2kb promoters.
"""

import gzip
import json
import gc
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_promoters(bed_path: str) -> List[Dict[str, Any]]:
    """
    Load promoter regions from BED file.
    Expected columns: chrom, start, end, gene, tss, strand
    """
    promoters: List[Dict[str, Any]] = []
    try:
        with open(bed_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                fields = line.split('\t')
                if len(fields) < 4:
                    logger.warning(f"Skipping malformed promoter line {i}")
                    continue
                try:
                    promoters.append({
                        'chrom': fields[0].replace('chr', '').strip(),
                        'start': int(fields[1]),
                        'end': int(fields[2]),
                        'gene': fields[3],
                        'tss': int(fields[4]) if len(fields) > 4 else int(fields[1]) + 2000,
                        'strand': fields[5] if len(fields) > 5 else '+'
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing promoter line {i}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Promoter BED file not found: {bed_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading promoters: {e}") from e

    logger.info(f"✅ Loaded {len(promoters)} promoter regions")
    return promoters


def overlap_variants_with_promoters(
    variant_path: str,
    promoter_path: str,
    output_path: str
) -> Tuple[int, int]:
    """
    Stream ClinVar variants and find overlaps with promoters.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    promoters = load_promoters(promoter_path)
    
    total = 0
    kept = 0

    logger.info(f"📂 Processing variants: {variant_path}")
    logger.info(f"📝 Writing output: {output_path}")

    try:
        with gzip.open(variant_path, 'rt', encoding='utf-8') as f_in, \
             gzip.open(output_path, 'wt', compresslevel=6, encoding='utf-8') as f_out:
            
            header = f_in.readline().strip()
            new_header = header + '\tpromoter_gene\tpromoter_tss\tpromoter_strand\tdistance_to_tss\n'
            f_out.write(new_header)
            
            for line_num, line in enumerate(f_in, start=2):
                total += 1
                try:
                    fields = line.strip().split('\t')
                    if len(fields) <= 19:
                        continue
                    
                    chrom_num = fields[18].strip()
                    pos_str = fields[19].strip()
                    
                    chrom = chrom_num.replace('chr', '')
                    try:
                        pos = int(pos_str)
                    except ValueError:
                        continue
                    
                    for prom in promoters:
                        if chrom == prom['chrom'] and prom['start'] <= pos <= prom['end']:
                            distance = pos - prom['tss']
                            out_line = (
                                line.strip() +
                                f"\t{prom['gene']}\t{prom['tss']}\t"
                                f"{prom['strand']}\t{distance}\n"
                            )
                            f_out.write(out_line)
                            kept += 1
                            break
                    
                    if total % 500 == 0:
                        logger.info(f"Processed {total:,} variants | Kept: {kept:,}")
                
                except Exception as e:
                    logger.warning(f"Error on line {line_num}: {e}")
                    continue

    except FileNotFoundError as e:
        logger.error(f"Input file missing: {e}")
        raise

    del promoters
    gc.collect()

    fraction = kept / total if total > 0 else 0
    logger.info(f"✅ Overlap complete: {kept:,}/{total:,} ({fraction:.2%})")
    return kept, total


def save_stats(kept: int, total: int, output_dir: str = "processed") -> None:
    """Save statistics."""
    Path(output_dir).mkdir(exist_ok=True)
    stats = {
        "stage": "promoter_overlap",
        "total_variants": total,
        "promoter_variants": kept,
        "fraction_in_promoters": round(kept / total, 4) if total > 0 else 0,
        "expected_range": "40-50%"
    }
    stats_path = Path(output_dir) / "overlap_stats.json"
    with open(stats_path, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"💾 Stats saved: {stats_path}")


def main(
    variant_path: str = "../01_Data/processed/clinvar_chr22.txt.gz",
    promoter_path: str = "processed/chr22_tss.bed",        # ← FIXED: Local path
    output_path: str = "processed/promoter_variants_chr22.txt.gz"
) -> None:
    """Main pipeline entry point."""
    logger.info("=" * 70)
    logger.info("🧬 PROMOTER-VARIANT OVERLAP PIPELINE")
    logger.info("=" * 70)
    
    kept, total = overlap_variants_with_promoters(
        variant_path=variant_path,
        promoter_path=promoter_path,
        output_path=output_path
    )
    
    save_stats(kept, total)
    
    print("\n" + "="*60)
    print("VERIFICATION CHECKPOINTS")
    print(f"Total variants processed : {total:,}")
    print(f"Variants in promoters    : {kept:,}")
    print(f"Match rate               : {kept/total:.2%}" if total > 0 else "N/A")
    print("="*60)


if __name__ == "__main__":
    main()