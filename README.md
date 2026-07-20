# EventHorizon Demand

> **Retail demand forecasting validated with academic-grade rigor.**  
> Part of the **EventHorizon** platform. Built to reduce waste and stock-outs.

---

# Why This Exists

Small and medium retailers often estimate next week's demand based on intuition or simple heuristics.

That often results in:

- 📦 **Excess inventory**, increasing storage costs and product waste.
- ❌ **Stock-outs**, causing lost sales and dissatisfied customers.

**EventHorizon Demand** replaces guesswork with a **statistically validated forecasting system** built for real-world retail operations.

The project:

- 📈 Achieves a **30% improvement** over the standard seasonal baseline (**WAPE 30.9% vs. 44.4%**).
- 🏬 Produces statistically significant improvements across **7 out of 7 stores** (one store exhibits a smaller margin, transparently documented).
- 🔬 Uses the same **honest validation methodology** developed for the EventHorizon crypto research.

> The objective is not simply to build an accurate model—it is to provide results that remain trustworthy under rigorous statistical evaluation.

---

# Features

| Feature | Description |
|---------|-------------|
| 📊 Accurate Forecasting | Reduces WAPE by **30%** compared to the common lag-7 seasonal baseline. |
| 🏬 Multi-Store Support | One model serves multiple stores using a `store_code` feature to capture store-specific behavior. |
| 📈 Rich Temporal Features | Includes yearly seasonality (`lag_365`), weekly seasonality, rolling statistics and exponential weighted averages. |
| 🔬 Honest Statistical Validation | Walk-forward evaluation, block bootstrap by date, paired gap bootstrap and confidence intervals. |
| ⚡ Lightweight Deployment | The trained **LightGBM** model is approximately **600 KB**, requiring only CPU inference. |
| 🔌 Easy Integration | Load with `joblib`, provide a DataFrame or CSV, and obtain predictions immediately. |

---

# Performance

| Metric | Seasonal Baseline (lag-7) | EventHorizon Demand | Improvement |
|--------|---------------------------:|--------------------:|------------:|
| **WAPE** | **44.4%** | **30.9%** | **30%** |
| **MAE** | 29.54 units | 18.77 units | 36% |

> **Evaluation Dataset:** M5 Walmart Forecasting Dataset (2015–2016 evaluation period)

Validation included:

- ✅ Walk-forward evaluation
- ✅ Block bootstrap by date
- ✅ Paired gap bootstrap
- ✅ Confidence intervals
- ✅ Per-store statistical analysis

---

# Quick Start

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("modelo_demanda_m5.pkl")

# Required features
features = [
    "store_code",
    "dia_seno",
    "dia_cos",
    "mes_seno",
    "mes_cos",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_365",
    "ewma_7",
    "ewma_28",
    "rolling_std_7",
    "fim_de_semana",
    "evento",
    "sell_price",
    "price_change",
    "snap"
]

# Load new sales data
df = pd.read_csv("your_store_sales.csv")

# Generate predictions
df["predicted_sales"] = model.predict(df[features])
```

---

# Repository Structure

```text
eventhorizon-demand/
│
├── model/
│   ├── modelo_demanda_m5.pkl      # Trained LightGBM model (~600 KB)
│   └── metricas_finais.json       # Metrics, confidence intervals and per-store analysis
│
├── notebooks/
│   └── m5_research.ipynb          # Complete research pipeline
│
├── src/
│   ├── features.py                # Feature engineering
│   └── validation.py              # Walk-forward, bootstrap and evaluation
│
└── README.md
```

---

# Validation Methodology

The evaluation protocol was designed to prevent optimistic estimates commonly found in time-series forecasting.

| Method | Purpose |
|---------|----------|
| **Walk-Forward Split** | Simulates real production forecasting by respecting temporal order. |
| **Block Bootstrap** | Estimates confidence intervals while preserving temporal dependence. |
| **Paired Gap Bootstrap** | Measures whether improvements over the baseline are statistically significant. |
| **Confidence Intervals** | Every reported metric includes uncertainty estimation. |
| **Per-Store Analysis** | Performance evaluated individually for every retail location. |

---

# Philosophy

| Principle | Description |
|-----------|-------------|
| 🎯 **Rigor over Hype** | Every reported improvement is accompanied by confidence intervals and transparent limitations. |
| 🔬 **Honest Validation** | No random shuffling, no data leakage, no cherry-picked results. |
| 🛠️ **Practical Deployment** | A compact LightGBM model that can run on virtually any machine. |
| 📊 **Transparent Metrics** | Evaluation scripts and methodology remain fully reproducible. |

---

# Use Cases

- 🛒 Supermarkets forecasting daily product demand
- 👕 Fashion retailers planning seasonal inventory
- 📦 E-commerce inventory replenishment
- 🚚 Distribution centers estimating weekly shipments
- 🏪 Multi-store retail forecasting
- 📈 Supply chain demand planning

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Model | LightGBM |
| Language | Python |
| Data Processing | pandas, NumPy |
| Validation | honest-validation-toolkit |
| Model Serialization | joblib |

---

# Links

| Project | Link |
|---------|------|
| 🚀 EventHorizon-AI Hub | *(add repository link)* |
| 📈 Crypto Research | *(add repository link)* |
| 📦 EventHorizon Demand | *(this repository)* |
| 🧪 honest-validation-toolkit | *(add repository link)* |

---

# License

MIT License.
