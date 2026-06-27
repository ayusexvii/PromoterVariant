#!/usr/bin/env python3
"""
Train model with PhyloP conservation scores (4 features)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import json

print("=" * 80)
print("🧬 TRAINING WITH PHYLOP CONSERVATION")
print("=" * 80)

# Load data with conservation
df = pd.read_csv('../02_Features/processed/training_matrix_with_conservation.csv.gz', compression='gzip')
print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

# Build features
df['abs_distance'] = df['distance_to_tss'].abs()

# Gene count
gene_counts = df['promoter_gene'].value_counts().to_dict()
df['gene_variant_count'] = df['promoter_gene'].map(gene_counts)

# Features (4 features now!)
X = df[['distance_to_tss', 'abs_distance', 'gene_variant_count', 'phyloP']].copy()
y = df['eqtl_slope'].values

# Clean
mask = X.notna().all(axis=1) & ~np.isnan(y)
X, y = X[mask], y[mask]

print(f"✅ Clean data: {len(X):,} samples, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train
model = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 80)
print("📊 MODEL WITH PHYLOP")
print("=" * 80)
print(f"   Test R²: {r2:.4f}")
print(f"   Test MAE: {mae:.4f}")

cv = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f"   CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

if hasattr(model, 'feature_importances_'):
    imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print("\n🔝 FEATURE IMPORTANCES:")
    for _, row in imp.iterrows():
        print(f"   {row['feature']:<25}: {row['importance']:.4f}")

# Save
joblib.dump(model, 'processed/conservation_model.pkl')

metrics = {
    "r2_test": float(r2),
    "mae": float(mae),
    "cv_r2_mean": float(cv.mean()),
    "cv_r2_std": float(cv.std()),
    "samples": len(X),
    "features": X.shape[1]
}
with open('processed/conservation_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n💾 Model saved: processed/conservation_model.pkl")
print("\n✅ TASK 3 COMPLETE!")
