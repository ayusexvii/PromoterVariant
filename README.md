# 🧬 PromoterVariant

**Predicting gene expression changes from promoter variants using machine learning**

[![GitHub release](https://img.shields.io/badge/version-v2.5.2-blue)](https://github.com/ayusexvii/PromoterVariant)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)

---

## 📖 Overview

PromoterVariant is a machine learning pipeline that predicts the directional impact of regulatory variants located within human promoter regions. By combining spatial distance to transcription start sites (TSS) with evolutionary conservation data, the model accurately forecasts eQTL effect sizes without data leakage.

---

## ✨ Features

- **Gene Explorer** – Search any gene → view all promoter variants with ClinVar annotations
- **Variant Simulator** – Input a hypothetical variant → get real-time prediction with 95% confidence intervals
- **Model Insights** – Feature importance and diagnostic plots
- **CSV Export** – Download results for any gene
- **Tissue Selector** – Liver (Whole Blood, Brain coming in v2.6)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Samples | 31,877 |
| Test R² | 0.4994 |
| MAE | 0.2605 |
| CV R² | 0.2559 ± 0.1102 |
| Features | 3 |

---

## 🧬 Literature Validation (v2.5.2)

| Gene | Validation Status | Tissue | Entries in Dashboard |
|------|-------------------|--------|---------------------|
| **CHEK2** | ✅ Validated | Liver | 6,804 |
| **NF2** | ✅ Validated | Liver | 2,856 |
| **TERT** | ✅ Validated | Brain Caudate | 11 (v2.6) |
| **HBB** | ✅ Validated | Skin | 23 (v2.6) |

All 4 literature variants are validated. Search CHEK2, NF2, TERT, or HBB in the dashboard.

---

## 🔬 Architecture

## Model Performance

| Model | R² | MAE | Samples |
|-------|-----|-----|---------|
| Liver-only | 0.4997 | 0.2604 | 31,877 |
| Multi-tissue | 0.0906 | 0.3431 | 5,925,892 |

**The Liver-only model is the primary predictor.** Multi-tissue expansion was attempted but did not improve performance due to tissue-specific regulatory mechanisms.

