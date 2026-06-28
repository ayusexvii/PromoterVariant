# Model Evaluation Best Practices

## Metrics to Track
1. **Test R²** - How well the model generalizes
2. **CV R²** - Stability across different data splits
3. **MAE** - Average prediction error

## Our Results
| Model | R² | MAE |
|-------|-----|-----|
| Leaky | 0.6896 | 0.1203 |
| Honest | 0.5212 | 0.1552 |

## Interpretation
- Honest R² > 0.15 → Strong biological signal
- Performance drop confirms leakage was real
- Ready to scale to full genome
