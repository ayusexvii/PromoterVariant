# 🧬 PromoterVariant

**Predicting gene expression changes from promoter variants using machine learning**

[![GitHub release](https://img.shields.io/badge/version-v1.0-blue)](https://github.com/ayusexvii/PromoterVariant)
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
- **Tissue Selector** – Liver (Whole Blood coming in v1.1)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ayusexvii/PromoterVariant.git
cd PromoterVariant

# Create and activate conda environment
conda env create -f environment.yml
conda activate promoter

# Launch the dashboard
cd 04_Dashboard
streamlit run app.py