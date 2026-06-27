#!/usr/bin/env python3
"""
Clean Feature Matrix Builder - Remove Data Leakage
Keeps only features available BEFORE seeing GTEx labels.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_clean_features(
    input_path: str = "processed/feature_matrix_chr22.csv.gz",
    output_path: str = "processed/clean_features_chr22.csv.gz"
):
    """Create clean feature matrix without leakage."""
    logger.info("=" * 80)
    logger.info("🧹 BUILDING CLEAN FEATURE MATRIX (No Leakage)")
    logger.info("=" * 80)
    
    try:
        df = pd.read_csv(input_path, compression='gzip', low_memory=False)
        logger.info(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
        
        # Define safe, non-leaky features (known before GTEx)
        clean_feature_list = [
            'distance_to_tss',
            'abs_distance',
            'distance_squared',
            'log_distance',
            'is_downstream',
            'dist_bin_very_close',
            'dist_bin_close',
            'dist_bin_medium',
            'dist_bin_far',
            'gene_variant_count'
        ]
        
        # Find which ones actually exist
        available_features = [col for col in clean_feature_list if col in df.columns]
        missing = [col for col in clean_feature_list if col in clean_feature_list and col not in df.columns]
        
        if missing:
            logger.warning(f"Missing expected columns: {missing}")
        
        # Create clean DataFrame
        clean_df = df[available_features].copy()
        
        logger.info(f"✅ Clean matrix ready: {clean_df.shape[1]} features × {len(clean_df)} samples")
        logger.info(f"   Features: {list(clean_df.columns)}")
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(output_path, index=False, compression='gzip')
        
        logger.info(f"💾 Saved clean features to: {output_path}")
        logger.info(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
        
        # Quick stats
        print("\n" + "="*60)
        print("CLEAN MATRIX SUMMARY")
        print("="*60)
        print(f"Rows          : {len(clean_df):,}")
        print(f"Features      : {clean_df.shape[1]}")
        print(f"Columns       : {list(clean_df.columns)}")
        print("\n✅ Clean matrix created successfully!")
        
        return clean_df
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error creating clean matrix: {e}")
        raise


if __name__ == "__main__":
    build_clean_features()
