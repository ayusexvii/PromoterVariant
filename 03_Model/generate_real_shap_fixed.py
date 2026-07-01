"""
Generate real SHAP values for the dashboard.
FIXED: check_additivity in shap_values(), not TreeExplainer.
"""
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

print("=" * 70)
print("🧬 GENERATING REAL SHAP VALUES (FIXED)")
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

# Sample for SHAP (use 2000 for speed)
X_sample = df[feature_cols].sample(2000, random_state=42)

print(f"✅ Using {len(X_sample)} samples for SHAP")

# SHAP - check_additivity in shap_values()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# Feature importance
imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n📊 SHAP Feature Importance:")
print(imp_df)

# Save
imp_df.to_csv('processed/real_shap_importance.csv', index=False)
print("\n💾 Saved to: processed/real_shap_importance.csv")

# Also save sample data for dashboard compatibility
imp_df.rename(columns={'Feature': 'feature'}).to_csv('processed/shap_dashboard_data.csv', index=False)
print("💾 Also saved as: processed/shap_dashboard_data.csv")

print("\n✅ COMPLETE!")
