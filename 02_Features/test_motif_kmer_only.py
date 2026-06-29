#!/usr/bin/env python3
"""
Motif disruption scoring using k-mer shifts only.
This is faster and more reliable than PWM scoring.
"""
import pandas as pd
import numpy as np
import time
from collections import Counter
from pathlib import Path

def compute_kmer_shifts(ref_seq: str, alt_seq: str, k=4) -> dict:
    """Compute k-mer frequency shifts between ref and alt sequences."""
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
    """Compute motif disruption features from k-mer shifts."""
    shifts = compute_kmer_shifts(ref_seq, alt_seq, k)
    
    # Features
    max_shift = max(shifts.values()) if shifts else 0
    sum_shift = sum(shifts.values())
    num_affected = sum(1 for v in shifts.values() if v != 0)
    total_kmers = len(shifts)
    
    # Novel k-mers (appear in alt but not ref)
    novel_kmers = sum(1 for v in shifts.values() if v > 0)
    lost_kmers = sum(1 for v in shifts.values() if v < 0)
    
    # Identify specific disrupted k-mers (top 3)
    changed = [k for k, v in shifts.items() if v != 0]
    top_changed = changed[:3] if changed else []
    
    return {
        'max_shift': max_shift,
        'sum_shift': sum_shift,
        'num_affected': num_affected,
        'novel_kmers': novel_kmers,
        'lost_kmers': lost_kmers,
        'disruption_intensity': sum_shift / total_kmers if total_kmers > 0 else 0,
        'top_changed_kmers': ','.join(top_changed)
    }

def main():
    print("=" * 70)
    print("🧬 MOTIF DISRUPTION - K-MER ONLY (Fast & Reliable)")
    print("=" * 70)
    
    input_path = Path("processed/variant_sequences_fixed.csv.gz")
    output_path = Path("processed/test_motif_results_500.csv.gz")
    
    df = pd.read_csv(input_path, compression='gzip')
    test_df = df.head(500).copy()
    print(f"Testing on {len(test_df)} variants")
    
    start_time = time.time()
    
    records = []
    for row in test_df.itertuples():
        ref_seq = str(row.ref_sequence)
        alt_seq = str(row.alt_sequence)
        
        if 'N' in ref_seq or 'N' in alt_seq:
            records.append({
                'max_shift': 0,
                'sum_shift': 0,
                'num_affected': 0,
                'novel_kmers': 0,
                'lost_kmers': 0,
                'disruption_intensity': 0.0,
                'top_changed_kmers': ''
            })
        else:
            features = motif_disruption_score(ref_seq, alt_seq, k=4)
            records.append(features)
    
    elapsed = time.time() - start_time
    print(f"✅ Processed in {elapsed:.2f}s")
    
    result_df = pd.DataFrame(records)
    test_df[['max_shift', 'sum_shift', 'num_affected', 'novel_kmers', 'lost_kmers', 'disruption_intensity']] = result_df
    
    print("\n📊 Feature Statistics:")
    print(test_df[['max_shift', 'num_affected', 'disruption_intensity']].describe())
    
    print("\n🔍 Sample results (first 5):")
    print(test_df[['max_shift', 'num_affected', 'disruption_intensity', 'top_changed_kmers']].head())
    
    test_df.to_csv(output_path, index=False, compression='gzip')
    
    estimated_full = elapsed * (len(df) / 500)
    print(f"\n⏱️ Estimated full runtime: {estimated_full:.1f}s ({estimated_full/60:.1f} min)")
    print("✅ COMPLETE!")

if __name__ == "__main__":
    main()
