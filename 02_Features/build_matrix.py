#!/usr/bin/env python3
"""
Feature Matrix Builder for Promoter Variant ML Training
"""

import gzip
import json
import gc
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_matrix(input_path: str) -> pd.DataFrame:
    """Load and clean the matched training matrix."""
    logger.info(f"📂 Loading training matrix: {input_path}")
    
    try:
        # For small Chr22 we can load fully; later switch to chunks for full genome
        df = pd.read_csv(input_path, sep='\t', compression='gzip', low_memory=False)
        
        logger.info(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Filter matched variants
        if 'eqtl_slope' in df.columns:
            df_matched = df[df['eqtl_slope'] != 'NA'].copy()
        else:
            logger.warning("No 'eqtl_slope' column found. Using all rows.")
            df_matched = df.copy()
        
        # Safe type conversion
        for col in ['eqtl_slope', 'eqtl_pval']:
            if col in df_matched.columns:
                df_matched[col] = pd.to_numeric(df_matched[col], errors='coerce')
        
        if 'distance_to_tss' in df_matched.columns:
            df_matched['distance_to_tss'] = pd.to_numeric(df_matched['distance_to_tss'], errors='coerce').astype('Int64')
        
        logger.info(f"✅ Matched variants: {len(df_matched):,} ({len(df_matched)/len(df):.2%} of total)")
        return df_matched
        
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {e}")
        raise


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build rich feature matrix."""
    logger.info("🔧 Engineering features...")
    
    features = pd.DataFrame(index=df.index)
    
    # === Core Distance Features ===
    if 'distance_to_tss' in df.columns:
        dist = df['distance_to_tss'].fillna(0).astype(float)
        features['distance_to_tss'] = dist
        features['abs_distance'] = dist.abs()
        features['distance_squared'] = dist ** 2
        features['log_distance'] = np.log1p(features['abs_distance'])
        
        # Direction
        features['is_downstream'] = (dist > 0).astype(int)
        
        # Bins
        features['dist_bin_very_close'] = (features['abs_distance'] <= 100).astype(int)
        features['dist_bin_close'] = ((features['abs_distance'] > 100) & (features['abs_distance'] <= 500)).astype(int)
        features['dist_bin_medium'] = ((features['abs_distance'] > 500) & (features['abs_distance'] <= 2000)).astype(int)
        features['dist_bin_far'] = (features['abs_distance'] > 2000).astype(int)
    
    # === Gene-level features ===
    if 'promoter_gene' in df.columns:
        gene_col = 'promoter_gene'
    elif 'gene' in df.columns:
        gene_col = 'gene'
    else:
        gene_col = None
    
    if gene_col:
        gene_counts = df[gene_col].value_counts()
        features['gene_variant_count'] = df[gene_col].map(gene_counts)
        # One-hot top genes (optional, low cardinality on chr22)
        top_genes = df[gene_col].value_counts().head(20).index
        for g in top_genes:
            features[f'gene_{g}'] = (df[gene_col] == g).astype(int)
    
    # === eQTL Quality ===
    if 'eqtl_pval' in df.columns:
        features['log_pval'] = -np.log10(df['eqtl_pval'].fillna(1) + 1e-300)
        features['significant_eqtl'] = (df['eqtl_pval'] < 1e-5).astype(int)
    
    # === Variant Context (if available) ===
    if 'ref' in df.columns and 'alt' in df.columns:
        features['is_transition'] = ((df['ref'] == 'A') & (df['alt'] == 'G')) | \
                                    ((df['ref'] == 'G') & (df['alt'] == 'A')) | \
                                    ((df['ref'] == 'C') & (df['alt'] == 'T')) | \
                                    ((df['ref'] == 'T') & (df['alt'] == 'C'))
    
    logger.info(f"✅ Built feature matrix: {features.shape[1]} features × {len(features)} samples")
    return features, df['eqtl_slope'].values


def main(
    input_path: str = "processed/training_matrix_chr22.txt.gz",
    output_dir: str = "processed"
) -> None:
    """Main feature engineering pipeline."""
    logger.info("=" * 80)
    logger.info("🧬 FEATURE MATRIX BUILDER FOR PROMOTER VARIANTS")
    logger.info("=" * 80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_training_matrix(input_path)
    
    # Build features
    X, y = build_features(df)
    
    # Save
    X.to_csv(f"{output_dir}/feature_matrix_chr22.csv.gz", index=False, compression='gzip')
    pd.Series(y, name='eqtl_slope').to_csv(f"{output_dir}/target_chr22.csv.gz", index=False, compression='gzip')
    
    # Summary stats
    summary: Dict[str, Any] = {
        "total_samples": len(X),
        "feature_count": X.shape[1],
        "features": list(X.columns),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "sparse_features": int((X == 0).all().sum())
    }
    
    with open(f"{output_dir}/feature_matrix_summary.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Samples          : {len(X):,}")
    logger.info(f"Features         : {X.shape[1]}")
    logger.info(f"Target mean      : {np.mean(y):.4f} ± {np.std(y):.4f}")
    logger.info(f"Target range     : [{np.min(y):.4f}, {np.max(y):.4f}]")
    logger.info(f"\n💾 Files saved in: {output_dir}/")
    logger.info("   • feature_matrix_chr22.csv.gz")
    logger.info("   • target_chr22.csv.gz")
    logger.info("   • feature_matrix_summary.json")


if __name__ == "__main__":
    main()