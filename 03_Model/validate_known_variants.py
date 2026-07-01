#!/usr/bin/env python3
"""
Validate known literature variants against gene-fallback model.
Honest verification utilizing mathematical sign checking.
"""
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("🧬 GENE-FALLBACK VALIDATION ENGINE — DIRECTIONAL VERIFICATION")
    print("=" * 80)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Trained Checkpoint Assets
    model_paths = [
        output_dir / "full_honest_model.pkl",
        output_dir / "full_model.pkl",
        output_dir / "final_exact_model.pkl"
    ]
    
    model = None
    for path in model_paths:
        if path.exists():
            model = joblib.load(path)
            print(f"✅ Loaded model: {path.name}")
            break
            
    if model is None:
        print("❌ CRITICAL ERROR: Model weights files missing on disk!", file=sys.stderr)
        sys.exit(1)

    # 2. Load Evaluation Data Matrix
    data_path = script_dir / "../02_Features/processed/training_matrix_allchr.csv.gz"
    if not data_path.exists():
        print(f"❌ CRITICAL ERROR: Feature matrix table missing at: {data_path}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(data_path, compression='gzip')
    print(f"✅ Loaded {len(df):,} evaluation base variant rows.")

    # 3. Dynamic Calculation Layer
    df['abs_distance'] = df['distance_to_tss'].abs()
    
    # Locate the tissue alignment gene identification column variation
    gene_candidates = [c for c in df.columns if c in ['promoter_gene', 'gtex_gene', 'clinvar_gene']]
    if not gene_candidates:
        print("❌ CRITICAL ERROR: Missing variant gene mapping identifier tracks.", file=sys.stderr)
        sys.exit(1)
    gene_col = gene_candidates[0]
    
    gene_counts = df[gene_col].value_counts().to_dict()
    df['gene_variant_count'] = df[gene_col].map(gene_counts)

    # 4. Generate Inference Pass
    feature_names = ['distance_to_tss', 'abs_distance', 'gene_variant_count']
    X = df[feature_names].copy()
    
    df['predicted_slope'] = model.predict(X)
    print("✅ Model inference generation pass complete.")

    # 5. Define Reference Literature Validation Baseline Sets
    variants = [
        {'gene': 'TERT', 'expected': 'up', 'variant': 'c.-124 G>A', 'lit': 'Horn et al., 2013'},
        {'gene': 'TERT', 'expected': 'up', 'variant': 'c.-146 C>T', 'lit': 'Huang et al., 2013'},
        {'gene': 'HBB', 'expected': 'down', 'variant': 'c.-87 C>G', 'lit': 'Orkin et al., 1983'},
        {'gene': 'HBG1', 'expected': 'up', 'variant': 'HPFH deletion', 'lit': 'Collins & Stoeckert, 2023'},
        {'gene': 'LCT', 'expected': 'up', 'variant': 'rs4988235', 'lit': 'Enattah et al., 2002'}
    ]

    print("\n" + "=" * 80)
    print("📊 LITERARY DIRECTIONAL VALIDATION ANALYSIS")
    print("=" * 80)
    
    results = []
    match_count = 0
    eval_count = 0

    for v in variants:
        gene = v['gene']
        expected = v['expected']
        variant_name = v['variant']
        lit = v['lit']

        # Isolate rows linked to the targeted locus sequence
        gene_df = df[df[gene_col].fillna('').astype(str).str.upper() == gene.upper()]

        if len(gene_df) == 0:
            print(f"\n❌ {variant_name} ({gene})")
            print(f"   Literature Ref: {lit}")
            print(f"   Expected Dir  : {expected}")
            print(f"   Status Marker : NOT PRESENT IN ACTIVE TRANSCRIPT DATASET (Tissue Expression Gate)")
            
            results.append({
                'variant': variant_name, 'gene': gene, 'expected': expected, 'literature': lit,
                'in_dataset': False, 'mean_slope': 'N/A', 'median_slope': 'N/A',
                'predicted_direction': 'N/A', 'match': 'N/A', 'interpretation': 'Tissue Mismatch'
            })
            continue

        eval_count += 1
        mean_slope = float(gene_df['predicted_slope'].mean())
        median_slope = float(gene_df['predicted_slope'].median())
        std_slope = float(gene_df['predicted_slope'].std()) if len(gene_df) > 1 else 0.0

        # FIXED: Pure sign-based directional check (guards against arbitrary boundary failures)
        # Uses a minimal envelope (1e-4) to safely capture neutral models
        if mean_slope > 0.0001:
            predicted_dir = 'up'
        elif mean_slope < -0.0001:
            predicted_dir = 'down'
        else:
            predicted_dir = 'neutral'

        # Match verification evaluate step
        is_match = (predicted_dir == expected)
        if is_match:
            match = '✅ MATCH'
            match_count += 1
        else:
            match = '❌ MISMATCH'

        # Set biological interpretation strings
        if mean_slope > 0.2:
            interp = 'Strong/Moderate positive up-regulation effect'
        elif mean_slope > 0.0:
            interp = 'Weak positive variant effect'
        elif mean_slope < -0.2:
            interp = 'Strong/Moderate negative down-regulation effect'
        else:
            interp = 'Weak negative variant effect'

        print(f"\n{match} -> {variant_name} ({gene})")
        print(f"   Literature Ref: {lit}")
        print(f"   Expected Dir  : {expected}")
        print(f"   Calculated Beta : {mean_slope:.5f} ± {std_slope:.5f} (Median: {median_slope:.5f})")
        print(f"   Inferred Vector : {predicted_dir}")
        print(f"   Biological Unit : {interp}")

        results.append({
            'variant': variant_name, 'gene': gene, 'expected': expected, 'literature': lit,
            'in_dataset': True, 'mean_slope': f"{mean_slope:.4f}", 'median_slope': f"{median_slope:.4f}",
            'predicted_direction': predicted_dir, 'match': match, 'interpretation': interp
        })

    # 6. Build and Export Validation Reports Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION COMPLETE MATRIX OVERVIEW")
    print("=" * 80)
    
    match_rate = (match_count / eval_count * 100) if eval_count > 0 else 0.0
    print(f"   Total Evaluated Expressed Genes : {eval_count} / {len(variants)}")
    print(f"   Identified Directional Matches  : {match_count} / {eval_count}")
    print(f"   Effective Predictive Match Rate : {match_rate:.1f}%")

    results_path = output_dir / 'validation_results.csv'
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\n💾 Validation matrix file successfully exported to: {results_path.name}")

    if eval_count == 0:
        conclusion = "⚠️ Tissue-Mismatch: Non-expressed gene indicators — cannot validate direction."
    elif match_rate >= 75.0:
        conclusion = "✅ Robust Verification: Gene-fallback captures accurate structural directions."
    elif match_rate >= 40.0:
        conclusion = "📊 Moderate Verification: Variable signal captured — check baseline limits."
    else:
        conclusion = "❌ Validation Rejection: Model outputs diverge from established biology."
        
    print(f"🔍 FINAL STUDY CONCLUSION: {conclusion}")
    print("=" * 80)

    with open(output_dir / 'validation_conclusion.txt', 'w', encoding='utf-8') as f:
        f.write(conclusion)

if __name__ == "__main__":
    main()
