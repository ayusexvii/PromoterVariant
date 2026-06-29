#!/usr/bin/env python3
"""
Test motif disruption with fixed PWM scoring
"""
import pandas as pd
import numpy as np
import time
from collections import Counter
from Bio import motifs
from Bio.Seq import Seq
from pathlib import Path

def calculate_pwm_delta_simple(ref_seq: str, alt_seq: str, motifs_list) -> float:
    """Simplified PWM delta calculation."""
    if not motifs_list or len(ref_seq) < 101:
        return 0.0
    
    ref_seq_bio = Seq(ref_seq)
    alt_seq_bio = Seq(alt_seq)
    
    max_delta = 0.0
    for motif in motifs_list[:10]:  # Test with first 10 motifs
        try:
            # Get the PWM as a simple position weight matrix
            pwm = motif.pwm
            # Scan for best match
            for start in range(len(ref_seq) - len(motif) + 1):
                ref_score = pwm[ref_seq_bio[start:start+len(motif)]]
                alt_score = pwm[alt_seq_bio[start:start+len(motif)]]
                delta = abs(ref_score - alt_score)
                if delta > max_delta:
                    max_delta = delta
        except Exception as e:
            continue
    return float(max_delta)

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

def main():
    print("=" * 70)
    print("🧬 MOTIF TEST - FIXED PWM SCORING")
    print("=" * 70)
    
    input_path = Path("processed/variant_sequences_fixed.csv.gz")
    motif_file = Path("../01_Data/raw/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
    
    df = pd.read_csv(input_path, compression='gzip')
    test_df = df.head(10).copy()
    print(f"Testing on {len(test_df)} variants")
    
    # Load motifs
    motifs_list = []
    if motif_file.exists():
        with open(motif_file) as f:
            motifs_list = list(motifs.parse(f, "jaspar"))
        print(f"Loaded {len(motifs_list)} motifs")
    
    # Process
    records = []
    for row in test_df.itertuples():
        ref_seq = str(row.ref_sequence)
        alt_seq = str(row.alt_sequence)
        
        shifts = compute_kmer_shifts(ref_seq, alt_seq, k=4)
        max_shift = max(shifts.values()) if shifts else 0
        num_affected = sum(1 for v in shifts.values() if v != 0)
        
        pwm_delta = calculate_pwm_delta_simple(ref_seq, alt_seq, motifs_list)
        
        records.append({
            'max_kmer_shift': max_shift,
            'num_kmer_affected': num_affected,
            'pwm_delta': pwm_delta
        })
    
    result_df = pd.DataFrame(records)
    print("\n📊 Results:")
    print(result_df)
    print("\n📊 Stats:")
    print(result_df.describe())

if __name__ == "__main__":
    main()
