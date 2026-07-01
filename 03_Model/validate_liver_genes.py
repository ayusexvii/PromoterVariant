"""
Validate liver-expressed disease genes in the dataset.
"""
import pandas as pd
import joblib
from pathlib import Path

print("=" * 70)
print("🧬 VALIDATING LIVER-EXPRESSED DISEASE GENES")
print("=" * 70)

# Load model
model = joblib.load('processed/full_honest_model.pkl')
print(f"✅ Loaded model")

# Load data
df = pd.read_csv('../02_Features/processed/training_matrix_allchr.csv.gz', compression='gzip')
print(f"✅ Loaded {len(df):,} variants")

# Build features
df['abs_distance'] = df['distance_to_tss'].abs()
gene_counts = df['promoter_gene'].value_counts().to_dict()
df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)

# Predict
X = df[['distance_to_tss', 'abs_distance', 'gene_variant_count']]
df['predicted_slope'] = model.predict(X)

# Target genes (liver-expressed, disease-associated)
target_genes = ['CHEK2', 'NF2', 'ARSA', 'APOL1']

print("\n📊 GENE-LEVEL VALIDATION (Liver-Expressed Disease Genes)")
print("=" * 70)

results = []
for gene in target_genes:
    gene_df = df[df['promoter_gene'] == gene]
    
    if len(gene_df) == 0:
        print(f"\n{gene}: NOT IN DATASET ❌")
        continue
    
    mean_slope = gene_df['predicted_slope'].mean()
    median_slope = gene_df['predicted_slope'].median()
    std_slope = gene_df['predicted_slope'].std()
    
    # Direction
    if mean_slope > 0.1:
        direction = "Upregulation"
        emoji = "⬆️"
    elif mean_slope < -0.1:
        direction = "Downregulation"
        emoji = "⬇️"
    else:
        direction = "Neutral"
        emoji = "➡️"
    
    results.append({
        'gene': gene,
        'variants': len(gene_df),
        'mean_slope': mean_slope,
        'median_slope': median_slope,
        'std_slope': std_slope,
        'direction': direction
    })
    
    print(f"\n{gene}:")
    print(f"   Variants: {len(gene_df):,}")
    print(f"   Mean slope: {mean_slope:.4f} ± {std_slope:.4f}")
    print(f"   Median slope: {median_slope:.4f}")
    print(f"   Predicted direction: {emoji} {direction}")

# Summary
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
for r in results:
    print(f"   {r['gene']:<8}: {r['variants']:>6} variants, {r['direction']} ({r['mean_slope']:.4f})")
