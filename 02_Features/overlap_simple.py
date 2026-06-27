#!/usr/bin/env python3
"""
Simple Full-Genome Overlap using Pandas (Memory-Efficient)
"""
import pandas as pd
import gzip
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🧬 FULL GENOME OVERLAP (SIMPLE PANDAS)")
print("=" * 80)

# 1. Load promoters
logger.info("📂 Loading promoters...")
promoters = []
with open('processed/all_tss.bed', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            promoters.append({
                'chrom': parts[0],
                'start': int(parts[1]),
                'end': int(parts[2]),
                'gene_name': parts[3],
                'tss': int(parts[4]),
                'strand': parts[5]
            })

prom_df = pd.DataFrame(promoters)
logger.info(f"✅ Loaded {len(prom_df):,} promoters")

# 2. Load variants
logger.info("📂 Loading variants...")
variants = []
with gzip.open('../01_Data/processed/clinvar_regulatory.txt.gz', 'rt') as f:
    header = f.readline().strip().split('\t')
    for i, line in enumerate(f):
        fields = line.strip().split('\t')
        if len(fields) < 20:
            continue
        chrom_raw = fields[18].strip()
        pos_str = fields[19].strip()
        if chrom_raw in ['X', 'Y', 'MT'] or not pos_str.isdigit():
            continue
        variants.append({
            'chrom': f"chr{chrom_raw}",
            'pos': int(pos_str),
            'ref': fields[21] if len(fields) > 21 else '',
            'alt': fields[22] if len(fields) > 22 else '',
            'gene': fields[4] if len(fields) > 4 else '',
            'name': fields[2] if len(fields) > 2 else '',
            'type': fields[1] if len(fields) > 1 else '',
            'clinsig': fields[6] if len(fields) > 6 else '',
            'raw': line.strip()
        })
        if (i + 1) % 20000 == 0:
            logger.info(f"   Loaded {i+1:,} variants...")

var_df = pd.DataFrame(variants)
logger.info(f"✅ Loaded {len(var_df):,} variants")

# 3. Group promoters by chromosome for faster lookup
logger.info("🔍 Performing overlap...")
prom_by_chrom = {chrom: group for chrom, group in prom_df.groupby('chrom')}

overlaps = []
total_variants = len(var_df)

for idx, variant in var_df.iterrows():
    chrom = variant['chrom']
    pos = variant['pos']
    
    if chrom not in prom_by_chrom:
        continue
    
    # Find promoters containing this position
    prom_group = prom_by_chrom[chrom]
    mask = (prom_group['start'] <= pos) & (pos <= prom_group['end'])
    matching_promoters = prom_group[mask]
    
    for _, prom in matching_promoters.iterrows():
        overlaps.append({
            'chrom': chrom,
            'pos': pos,
            'ref': variant['ref'],
            'alt': variant['alt'],
            'variant_gene': variant['gene'],
            'variant_name': variant['name'],
            'variant_type': variant['type'],
            'clinsig': variant['clinsig'],
            'promoter_gene': prom['gene_name'],
            'promoter_tss': prom['tss'],
            'promoter_strand': prom['strand'],
            'distance_to_tss': pos - prom['tss']
        })
    
    if (idx + 1) % 5000 == 0:
        logger.info(f"   Processed {idx+1:,}/{total_variants:,} variants...")

overlap_df = pd.DataFrame(overlaps)
logger.info(f"✅ Found {len(overlap_df):,} overlaps")

# 4. Save
overlap_df.to_csv('processed/promoter_variants_allchr.txt.gz', sep='\t', index=False, compression='gzip')
logger.info(f"💾 Saved to: processed/promoter_variants_allchr.txt.gz")

# Summary
print("\n" + "=" * 80)
print("📊 OVERLAP SUMMARY")
print("=" * 80)
print(f"Promoters loaded      : {len(prom_df):,}")
print(f"Variants loaded       : {len(var_df):,}")
print(f"Overlaps found        : {len(overlap_df):,}")
print(f"Unique genes hit      : {overlap_df['promoter_gene'].nunique():,}")
print(f"Unique variants hit   : {overlap_df['pos'].nunique():,}")

print("\n✅ TASK 2 COMPLETE!")
