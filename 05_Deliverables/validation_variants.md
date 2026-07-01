# Validation Variants - Day 16

## 5 Known Variants with Literature Evidence

| # | Variant | Gene | Coordinate (hg38) | Expected Effect | Literature Source |
|---|---------|------|-------------------|-----------------|-------------------|
| 1 | TERT c.-124 G>A | TERT | chr5:1295228 | Upregulation (creates ETS site) | Horn et al., 2013, Science |
| 2 | TERT c.-146 C>T | TERT | chr5:1295250 | Upregulation (creates ETS site) | Huang et al., 2013, PNAS |
| 3 | HBB c.-87 C>G | HBB | chr11:5227002 | Downregulation (destroys TATA box) | Orkin et al., 1983, Nature |
| 4 | HPFH deletions | HBG1/HBG2 | chr11:5225000-5228000 | Upregulation (γ-globin persistence) | Collins & Stoeckert, 2023, NEJM |
| 5 | LCT rs4988235 (-13910 T>C) | LCT | chr2:136608646 | Upregulation (lactase persistence) | Enattah et al., 2002, Nature Genetics |

## Expected Outcomes

The gene-fallback model is trained on **GTEx Liver** eQTLs. Expected results:

| Gene | In Liver GTEx? | Expected Finding |
|------|----------------|------------------|
| TERT | ❌ Silenced in adult liver | Not in dataset / zero slope |
| HBB | ❌ Erythroid-specific | No liver-specific regulatory signal |
| HBG1 | ❌ Fetal globin, not in adult liver | No liver-specific regulatory signal |
| LCT | ❌ Intestinal enhancer | Tissue mismatch, no liver signal |

## What This Validates

- If the model fails on all 5 variants, this documents the **liver-specific limitation** of the model.
- If the model matches any, this is a **bonus signal**.
- Either outcome is valuable when documented honestly.

## Next Steps

1. Run `03_Model/validate_known_variants.py`
2. Record predicted directions
3. Document matches/mismatches
4. Update README with results
