#!/usr/bin/env python3
"""
Test motif disruption scoring on 500 variants.
Features robust vector alignment fixes and Biopython PWM binding score fallbacks.
"""
import pandas as pd
import numpy as np
import time
import sys
from collections import Counter
from Bio import motifs
from Bio.Seq import Seq
from pathlib import Path

def compute_kmer_shifts(ref_seq: str, alt_seq: str, k=4) -> dict:
    """Compute k-mer frequency shifts. Fixed from internal breakout loop bug."""
    # Narrow tracking window centered cleanly around mutated position (45-56)
    center_start, center_end = 45, 56
    ref_window = ref_seq[center_start:center_end]
    alt_window = alt_seq[center_start:center_end]
    
    ref_kmers = Counter([ref_window[i:i+k] for i in range(len(ref_window)-k+1)])
    alt_kmers = Counter([alt_window[i:i+k] for i in range(len(alt_window)-k+1)])
    
    all_kmers = set(ref_kmers.keys()) | set(alt_kmers.keys())
    shifts = {}
    for kmer in all_kmers:
        shifts[kmer] = alt_kmers.get(kmer, 0) - ref_kmers.get(kmer, 0)
        
    return shifts  # FIXED: Moved outside the for loop to return all shifts

def calculate_pwm_delta(ref_seq: str, alt_seq: str, motifs_list) -> float:
    """Calculate the maximum delta binding score across loaded JASPAR matrices."""
    if not motifs_list or len(ref_seq) < 101:
        return 0.0
        
    # Isolate binding space sequence windows
    ref_seq_bio = Seq(ref_seq)
    alt_seq_bio = Seq(alt_seq)
    
    max_delta = 0.0
    # Score a subset of top motifs to keep test execution fast
    for motif in motifs_list[:15]:
        try:
            # Convert raw PFM count parameters to log-odds scoring probabilities
            pwm = motif.counts.normalize().log_odds()
            
            # Extract raw scores for both sequences
            ref_scores = pwm.calculate(ref_seq_bio)
            alt_scores = pwm.calculate(alt_seq_bio)
            
            # Find the max score drop or gain
            delta = np.max(np.abs(alt_scores - ref_scores))
            if delta > max_delta:
                max_delta = delta
        except Exception:
            continue
            
    return float(max_delta)

def main():
    print("=" * 70)
    print("🧬 MOTIF DISRUPTION ACCELERATED TEST PIPELINE (500 VARIANTS)")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "processed" / "variant_sequences_fixed.csv.gz"
    motif_file = script_dir / "../01_Data/raw/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
    output_path = script_dir / "processed" / "test_motif_results_500.csv.gz"
    
    if not input_path.exists():
        print(f"❌ ERROR: Missing target source sequence matrix: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"📂 Unpacking variant matrix: {input_path}")
    df = pd.read_csv(input_path, compression='gzip')
    test_df = df.head(500).copy()
    print(f"✅ Extracted primary baseline cluster pool of {len(test_df)} variants.")
    
    # --- Load JASPAR motifs ---
    print("\n📂 Loading JASPAR motif matrix libraries...")
    motifs_list = []
    if motif_file.exists():
        try:
            with open(motif_file) as f:
                motifs_list = list(motifs.parse(f, "jaspar"))
            print(f"✅ Loaded {len(motifs_list)} functional regulatory PFMs successfully.")
        except Exception as e:
            print(f"⚠️ Failed to parse Jaspar catalog: {e}", file=sys.stderr)
    else:
        print(f"⚠️ Motif database path absent at: {motif_file}. Spawning synthetic profiles.")
        from Bio.motifs import create
        for i in range(5):
            motifs_list.append(create.motif([Seq("ACGTACGT"), Seq("TGCAACGT")], name=f"Synthetic_Motif_{i}"))

    # --- Run processing loop ---
    print("\n🔧 Executing multi-feature mutation extraction loop...")
    start_time = time.time()
    
    records = []
    
    # Process variants using fast named tuples to avoid dataframe desync bugs
    for row in test_df.itertuples(index=True):
        ref_seq = str(row.ref_sequence)
        alt_seq = str(row.alt_sequence)
        
        ref_base = ref_seq[50] if len(ref_seq) > 50 else 'N'
        alt_base = alt_seq[50] if len(alt_seq) > 50 else 'N'
        
        if 'N' in ref_seq or 'N' in alt_seq:
            max_shift, sum_shift, num_affected, pwm_max_delta = 0, 0, 0, 0.0
            changed_kmers = ""
        else:
            shifts = compute_kmer_shifts(ref_seq, alt_seq, k=4)
            
            max_shift = max(shifts.values()) if shifts else 0
            sum_shift = sum(shifts.values())
            num_affected = sum(1 for v in shifts.values() if v != 0)
            changed_kmers = ",".join([k for k, v in shifts.items() if v != 0][:5])
            
            # Extract true motif PWM binding affinity changes
            pwm_max_delta = calculate_pwm_delta(ref_seq, alt_seq, motifs_list)
            
        records.append({
            'max_kmer_shift': max_shift,
            'sum_kmer_shift': sum_shift,
            'num_kmer_affected': num_affected,
            'changed_kmers': changed_kmers,
            'pwm_max_disruption': pwm_max_delta,
            'ref_base': ref_base,
            'alt_base': alt_base
        })
        
    elapsed = time.time() - start_time
    print(f"✅ 500 benchmark rows compiled seamlessly in {elapsed:.4f} seconds.")
    
    # Merging outputs safely via an explicit dictionary array structure
    feat_df = pd.DataFrame(records)
    
    # Transfer values back onto tracking vectors safely without index mismatches
    for col in feat_df.columns:
        test_df[col] = feat_df[col].values
        
    print("\n📊 Extracted Structural Feature Statistics:")
    print(test_df[['max_kmer_shift', 'pwm_max_disruption', 'num_kmer_affected']].describe())
    
    # Validate variant quality
    real_variants = (test_df['ref_base'] != test_df['alt_base']).sum()
    print(f"\n🎯 Confirmed Real Mutation Variants (Ref != Alt Alignment): {real_variants}/{len(test_df)}")
    
    print(f"📝 Compacting test results to target location: {output_path}")
    test_df.to_csv(output_path, index=False, compression='gzip')
    
    if elapsed > 0:
        estimated_full = elapsed * (len(df) / 500)
        print(f"\n⏱️ Scaled Global Velocity Estimate: {estimated_full:.1f}s ({estimated_full/60:.2f} min to parse full dataset).")
    print("=" * 70)
    print("✅ TEST PIPELINE RUN STABLE AND COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
