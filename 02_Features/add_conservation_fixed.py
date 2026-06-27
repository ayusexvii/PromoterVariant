#!/usr/bin/env python3
"""
Add PhyloP conservation scores to training matrix.
Fixed version - no syntax errors.
"""
import pyBigWig
import pandas as pd
import logging
import time
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("🧬 PHYLOP CONSERVATION MATRIX GENERATOR (FIXED)")
    print("=" * 80)
    
    script_dir = Path(__file__).resolve().parent
    matrix_path = script_dir / 'processed' / 'training_matrix_allchr.csv.gz'
    bw_path = script_dir / '../01_Data/raw/hg38.phyloP100way.bw'
    output_path = script_dir / 'processed' / 'training_matrix_with_conservation.csv.gz'
    
    # 1. Load training matrix
    if not matrix_path.exists():
        logger.error(f"❌ Missing training matrix: {matrix_path}")
        sys.exit(1)
        
    logger.info("📂 Loading training matrix...")
    train_df = pd.read_csv(matrix_path, compression='gzip')
    logger.info(f"✅ Loaded {len(train_df):,} rows with {len(train_df.columns)} columns")
    
    # 2. Get coordinates
    if 'chrom' in train_df.columns and 'pos' in train_df.columns:
        chrom_col, pos_col = 'chrom', 'pos'
    elif 'Chromosome' in train_df.columns and 'Start' in train_df.columns:
        chrom_col, pos_col = 'Chromosome', 'Start'
    elif 'Chromosome' in train_df.columns and 'End' in train_df.columns:
        chrom_col, pos_col = 'Chromosome', 'End'
    else:
        logger.error(f"❌ No chrom/pos columns found. Columns: {list(train_df.columns)}")
        sys.exit(1)
        
    logger.info(f"🎯 Using: {chrom_col}, {pos_col}")
    
    # 3. Open BigWig
    if not bw_path.exists():
        logger.error(f"❌ BigWig file not found: {bw_path}")
        sys.exit(1)
        
    logger.info("📂 Opening PhyloP BigWig...")
    bw = pyBigWig.open(str(bw_path))
    bw_chroms = set(bw.chroms().keys())
    logger.info(f"✅ BigWig loaded: {len(bw_chroms)} chromosomes")
    
    # 4. Query conservation
    logger.info("🔍 Querying conservation scores...")
    scores = []
    errors = 0
    zero_scores = 0
    start_time = time.time()
    total = len(train_df)
    
    for idx, row in train_df.iterrows():
        # Get chromosome
        raw_chrom = str(row[chrom_col])
        if '.' in raw_chrom:
            raw_chrom = raw_chrom.split('.')[0]
        chrom = raw_chrom if raw_chrom.startswith('chr') else f"chr{raw_chrom}"
        
        # Get position (convert to 0-based for BigWig)
        try:
            pos_1based = int(row[pos_col])
            pos_0based = pos_1based - 1
        except (ValueError, TypeError):
            scores.append(0.0)
            errors += 1
            continue
        
        # Query
        if pos_0based >= 0 and chrom in bw_chroms:
            try:
                val = bw.values(chrom, pos_0based, pos_0based + 1)
                if val and len(val) > 0 and val[0] is not None:
                    score = float(val[0])
                    scores.append(score)
                    if score == 0.0:
                        zero_scores += 1
                else:
                    scores.append(0.0)
                    zero_scores += 1
            except Exception:
                scores.append(0.0)
                errors += 1
        else:
            scores.append(0.0)
            zero_scores += 1
        
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            logger.info(f"   Processed {idx+1:,}/{total:,} variants... ({elapsed:.1f}s)")
    
    bw.close()
    
    # 5. Add scores and save
    train_df['phyloP'] = scores
    train_df.to_csv(output_path, index=False, compression='gzip')
    logger.info(f"💾 Saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PHYLOP SUMMARY")
    print("=" * 80)
    print(f"Total Variants      : {len(train_df):,}")
    print(f"Mean PhyloP         : {train_df['phyloP'].mean():.4f}")
    print(f"Std PhyloP          : {train_df['phyloP'].std():.4f}")
    print(f"Min PhyloP          : {train_df['phyloP'].min():.4f}")
    print(f"Max PhyloP          : {train_df['phyloP'].max():.4f}")
    print(f"Non-zero scores     : {(train_df['phyloP'] != 0).sum():,}")
    print(f"Errors              : {errors}")
    print(f"Zero scores         : {zero_scores}")
    print("=" * 80)
    print("✅ CONSERVATION SCORES ADDED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
