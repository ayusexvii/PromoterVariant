#!/usr/bin/env python3
"""
Feature Matrix Builder - FIXED VERSION
Properly handles 'NA' strings in eqtl_slope column.
"""
import gzip
import json
import pandas as pd
import numpy as np
import os

print("=" * 70)
print("🧬 FEATURE MATRIX BUILDER (FIXED)")
print("=" * 70)

# Load the training matrix
input_path = "processed/training_matrix_chr22.txt.gz"
if not os.path.exists(input_path):
    input_path = "../02_Features/processed/training_matrix_chr22.txt.gz"

print(f"📂 Loading: {input_path}")

df = pd.read_csv(input_path, sep='\t', compression='gzip', low_memory=False)
print(f"✅ Loaded {len(df):,} rows")

# Fix: Convert 'NA' strings to actual NaN
df['eqtl_slope'] = df['eqtl_slope'].replace('NA', np.nan)
df['eqtl_slope'] = pd.to_numeric(df['eqtl_slope'], errors='coerce')

# Filter to matched variants (with actual slopes)
df_matched = df[df['eqtl_slope'].notna()].copy()
print(f"✅ Matched variants (with slopes): {len(df_matched):,}")

if len(df_matched) == 0:
    print("\n❌ ERROR: No matched variants found!")
    print("   Check that 'eqtl_slope' column has numeric values.")
    print("   Current column values:", df['eqtl_slope'].unique()[:10])
    exit(1)

# Convert other columns
df_matched['eqtl_pval'] = pd.to_numeric(df_matched['eqtl_pval'], errors='coerce')
df_matched['distance_to_tss'] = pd.to_numeric(df_matched['distance_to_tss'], errors='coerce')

# Build features
print("🔧 Building features...")

dist = df_matched['distance_to_tss'].fillna(0).astype(float)
features = pd.DataFrame()

# Distance features
features['distance_to_tss'] = dist
features['abs_distance'] = dist.abs()
features['distance_squared'] = dist ** 2
features['log_distance'] = np.log1p(features['abs_distance'])
features['is_downstream'] = (dist > 0).astype(int)

# Distance bins
features['dist_bin_close'] = (features['abs_distance'] <= 500).astype(int)
features['dist_bin_medium'] = ((features['abs_distance'] > 500) & (features['abs_distance'] <= 2000)).astype(int)
features['dist_bin_far'] = (features['abs_distance'] > 2000).astype(int)

# Gene features
gene_counts = df_matched['promoter_gene'].value_counts()
features['gene_variant_count'] = df_matched['promoter_gene'].map(gene_counts)

# eQTL quality
features['log_pval'] = -np.log10(df_matched['eqtl_pval'].fillna(1) + 1e-300)
features['significant_eqtl'] = (df_matched['eqtl_pval'] < 1e-5).astype(int)

# Target
target = df_matched['eqtl_slope'].values

print(f"✅ Built {features.shape[1]} features for {len(features)} variants")

# Create processed directory
os.makedirs("processed", exist_ok=True)

# Save
features.to_csv("processed/feature_matrix_chr22.csv.gz", index=False, compression='gzip')
pd.Series(target, name='eqtl_slope').to_csv("processed/target_chr22.csv.gz", index=False, compression='gzip')

# Summary
summary = {
    "total_samples": len(features),
    "feature_count": features.shape[1],
    "features": list(features.columns),
    "target_mean": float(np.mean(target)) if not np.isnan(np.mean(target)) else None,
    "target_std": float(np.std(target)) if not np.isnan(np.std(target)) else None,
    "target_min": float(np.min(target)) if not np.isnan(np.min(target)) else None,
    "target_max": float(np.max(target)) if not np.isnan(np.max(target)) else None
}

with open("processed/feature_matrix_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"✅ Samples: {len(features):,}")
print(f"✅ Features: {features.shape[1]}")
print(f"✅ Target mean: {np.mean(target):.4f}")
print(f"✅ Target std: {np.std(target):.4f}")
print(f"✅ Target range: {np.min(target):.4f} to {np.max(target):.4f}")

print("\n💾 Files saved in: processed/")
print("   • feature_matrix_chr22.csv.gz")
print("   • target_chr22.csv.gz")
print("   • feature_matrix_summary.json")

print("\n✅ TASK 4 COMPLETE!")
