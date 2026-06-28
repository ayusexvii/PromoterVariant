# Data Leakage in ML

## Definition
Data leakage occurs when information from outside the training dataset is used to train the model.

## Our Example
Using `log_pval` (eQTL p-value) to predict `eqtl_slope` (effect size) was leakage because:
- Both come from the same statistical test
- p-value is calculated USING the slope
- It's circular reasoning

## Detection
- CV R² (0.276) << Test R² (0.689) → Classic sign of leakage
- Removing leakage dropped R² from 0.689 to 0.521 (24.4% drop)

## Lesson
Always remove features that are derived from your target variable.
