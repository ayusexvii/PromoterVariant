#!/usr/bin/env python3
"""
Extract motif disruption features for all variants using k-mer shifts.
"""
import pandas as pd
import numpy as np
import time
from collections import Counter
from pathlib import Path
import sys

def compute_kmer_shifts(ref_seq: str, alt_seq: str, k=4) -> dict:
    center_start, center_end = 45, 56
    ref_window = ref_seq[center_start:center_end]
    alt_window = alt_seq[center_start:center_end]
    
    ref_kmers = Counter([ref_window[i:i+k] for i in range(len(ref_window)-k+1)])
    alt_kmers = Counter([alt_window[i:i+k] for i in range(len(alt_window)-k+1)])
    
    all_kmers = set(ref_kmers.keys()) | set(alt_kmers.keys())
    shifts = {}
    for kmer in all_kmers:
        shifts[kmer] = alt_kmers.get(kmer, 0) - ref_kmers.get(kmer, 0)
    return shifts

def motif_disruption_score(ref_seq: str, alt_seq: str, k=4) -> dict:
    shifts = compute_kmer_shifts(ref_seq, alt_seq, k)
    max_shift = max(shifts.values()) if shifts else 0
    sum_shift = sum(shifts.values())
    num_affected = sum(1 for v in shifts.values() if v != 0)
    total_kmers = len(shifts)
    novel_kmers = sum(1 for v in shifts.values() if v > 0)
    lost_kmers = sum(1 for v in shifts.values() if v < 0)
    
    return {
        'max_shift': max_shift,
        'sum_shift': sum_shift,
        'num_affected': num_affected,
        'novel_kmers': novel_kmers,
        'lost_kmers': lost_kmers,
        'disruption_intensity': sum_shift / total_kmers if total_kmers > 0 else 0,
    }

def main():
    print("=" * 70)
    print("🧬 EXTRACTING MOTIF FEATURES - FULL DATASET")
    print("=" * 70)
    
    input_path = Path("processed/variant_sequences_fixed.csv.gz")
    output_path = Path("processed/motif_features_full.csv.gz")
    
    if not input_path.exists():
        print(f"❌ ERROR: {input_path} not found")
        sys.exit(1)
    
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path, compression='gzip')
    print(f"✅ Loaded {len(df):,} variants")
    
    start_time = time.time()
    
    features_list = []
    total = len(df)
    
    for i, row in df.iterrows():
        ref_seq = str(row.ref_sequence)
        alt_seq = str(row.alt_sequence)
        
        if 'N' in ref_seq or 'N' in alt_seq:
            features_list.append({
                'max_shift': 0,
                'sum_shift': 0,
                'num_affected': 0,
                'novel_kmers': 0,
                'lost_kmers': 0,
                'disruption_intensity': 0.0,
            })
        else:
            features = motif_disruption_score(ref_seq, alt_seq, k=4)
            features_list.append(features)
        
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"   Processed {i+1:,}/{total:,}... ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    
    # Add features to dataframe
    feat_df = pd.DataFrame(features_list)
    for col in feat_df.columns:
        df[col] = feat_df[col].values
    
    print(f"\n✅ Processed {len(df):,} variants in {elapsed:.2f}s")
    print(f"💾 Saving to: {output_path}")
    df.to_csv(output_path, index=False, compression='gzip')
    
    print("\n📊 Feature Summary:")
    print(df[['max_shift', 'num_affected', 'disruption_intensity']].describe())
    print("\n✅ COMPLETE!")

if __name__ == "__main__":
    main()
