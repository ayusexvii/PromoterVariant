#!/usr/bin/env python3
"""
SHAP Analysis for Model Interpretability
Works with HistGradientBoostingRegressor
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("🔬 SHAP ANALYSIS - FIXED")
print("=" * 70)

# Load data
print("📂 Loading data...")
df = pd.read_csv('processed/feature_matrix_chr22.csv.gz', compression='gzip')
y = df['eqtl_slope'].values.ravel()
X = df.drop(columns=['eqtl_slope'], errors='ignore')

# Clean
mask = ~np.isnan(y)
X = X[mask].reset_index(drop=True)
y = y[mask]
print(f"✅ Loaded {len(X):,} samples with {X.shape[1]} features")

# Load model
model = joblib.load('processed/baseline_model.pkl')
print("✅ Model loaded")

# For HistGradientBoosting, use a smaller background sample
background_size = min(30, len(X))
background = X.sample(background_size, random_state=42)
print(f"✅ Background: {background_size} samples")

# Use TreeSHAP with validation_data parameter
print("\n🔧 Computing SHAP values (TreeSHAP)...")
explainer = shap.TreeExplainer(model, data=background, model_output='raw')
shap_values = explainer.shap_values(X)

print(f"✅ SHAP values computed: {len(shap_values)} samples")

# SHAP Summary Plot
print("\n📊 Generating SHAP summary...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False, max_display=10)
plt.tight_layout()
plt.savefig('processed/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Summary plot saved: processed/shap_summary.png")

# SHAP Bar Plot
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig('processed/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Bar plot saved: processed/shap_bar.png")

# Feature importance from SHAP
feature_importance = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': feature_importance
}).sort_values('shap_importance', ascending=False)

print("\n" + "=" * 70)
print("📊 TOP 10 SHAP FEATURES")
print("=" * 70)
for i, row in feature_importance_df.head(10).iterrows():
    print(f"   {row['feature']:<30} : {row['shap_importance']:.5f}")

# Save
feature_importance_df.to_csv('processed/shap_feature_importance.csv', index=False)
print("\n💾 SHAP feature importance saved: processed/shap_feature_importance.csv")

# Also save as feature_importance.csv for compatibility
feature_importance_df.rename(columns={'shap_importance': 'importance'}).to_csv('processed/feature_importance.csv', index=False)
print("💾 Also saved as: processed/feature_importance.csv")

print("\n✅ TASK 3 COMPLETE!")
