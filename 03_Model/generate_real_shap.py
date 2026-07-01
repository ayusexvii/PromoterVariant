"""
Generate real SHAP values for the dashboard.
"""
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

print("=" * 70)
print("🧬 GENERATING REAL SHAP VALUES")
print("=" * 70)

# Load model and data
model = joblib.load('processed/positional_model.pkl')
df = pd.read_csv('../02_Features/processed/training_matrix_with_positional_features.csv.gz', compression='gzip')
print(f"✅ Loaded {len(df):,} rows")

# Prepare features
df['abs_distance'] = df['distance_to_tss'].abs()
gene_counts = df['promoter_gene'].value_counts().to_dict()
df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)

feature_cols = ['distance_to_tss', 'abs_distance', 'gene_variant_count']
X = df[feature_cols].sample(5000, random_state=42)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Feature importance
imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

imp_df.to_csv('processed/real_shap_importance.csv', index=False)
print("\n📊 SHAP Feature Importance:")
print(imp_df)

print("\n💾 Saved to: processed/real_shap_importance.csv")
