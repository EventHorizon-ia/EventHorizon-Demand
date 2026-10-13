# EventHorizon Demand

> **Retail demand forecasting with honest validation.**  
> Part of the **EventHorizon** platform. Built to reduce waste and stock-outs.

---

## Why This Exists

Small and medium retailers often estimate next week’s demand based on
intuition or simple heuristics.

That often results in:

-  **Excess inventory**, increasing storage costs and product waste.
-  **Stock-outs**, causing lost sales and dissatisfied customers.

**EventHorizon Demand** replaces guesswork with a statistically validated
forecasting system built for real-world retail operations.

The current MVP:

-  Achieves **31.03% overall WAPE** on the M5 retail benchmark.
-  Outperforms the naive lag-7 baseline in **7 out of 7 stores**.
-  Does **not** require `lag_365`, making it more suitable for use cases
  with shorter sales history.
-  Uses temporal validation, block bootstrap and paired gap-bootstrap
  intervals.

> The objective is not simply to build an accurate model. It is to provide
> results that remain trustworthy under rigorous statistical evaluation.

---

## Current MVP Result

| Metric | EventHorizon Demand MVP |
|---|---:|
| Overall WAPE | **31.03%** |
| Validation setup | Temporal holdout: train ≤ 2014, validate ≥ 2015 |
| Feature set | No `lag_365` |
| Store-level result | Better than the naive lag-7 baseline in 7/7 stores |
| Market validation | Ongoing |

### Store-level WAPE

| Store | EventHorizon Demand MVP | Naive lag-7 baseline | Paired WAPE gap |
|---|---:|---:|---:|
| CA_1 | 31.49% | 46.95% | 10.10–22.00 pp |
| CA_2 | 35.04% | 44.93% | 5.69–14.85 pp |
| CA_3 | 28.72% | 40.38% | 6.54–17.44 pp |
| TX_1 | 28.50% | 42.03% | 9.06–19.33 pp |
| TX_2 | 28.10% | 43.91% | 10.84–21.72 pp |
| WI_1 | 40.97% | 50.59% | 4.71–15.17 pp |
| WI_2 | 37.19% | 52.12% | 10.48–20.05 pp |

The intervals are 95% percentile intervals from a moving block bootstrap with
seven-day blocks and 1,000 resamples. The gap is calculated as the paired
difference between the naive lag-7 baseline and EventHorizon Demand.

> These results establish technical model performance, not business ROI.

A previous legacy model achieved 30.93% WAPE, but it used `lag_365` and was
trained before a weekend-feature bug fix. It is retained only as historical
context and must not be interpreted as the current MVP’s performance.

---

## Features

| Feature | Description |
|---|---|
|  Demand forecasting | Predicts future retail demand from historical sales and contextual features. |
|  Multi-store support | Uses `store_code` to learn store-specific behavior. |
|  Temporal features | Uses short-term lags, weekly and yearly seasonality, rolling statistics and exponential weighted averages. |
|  No `lag_365` requirement | The MVP does not require a full year of historical sales. |
|  Honest validation | Uses temporal holdout validation, block bootstrap and paired gap bootstrap. |
| Lightweight deployment | The trained LightGBM model is compact and supports CPU inference. |
|  Easy integration | Load with `joblib`, provide a DataFrame with the required features and obtain predictions. |

---

## Quick Start

```python
import joblib
import pandas as pd

model = joblib.load("model/modelo_demanda_mvp.pkl")

features = [
    "store_code",
    "dia_seno",
    "dia_cos",
    "mes_seno",
    "mes_cos",
    "lag_7",
    "lag_14",
    "lag_28",
    "ewma_7",
    "ewma_28",
    "rolling_std_7",
    "fim_de_semana",
    "evento",
    "sell_price",
    "price_change",
    "snap"
]

df = pd.read_csv("your_store_sales.csv")

df["predicted_sales"] = model.predict(df[features])
```

> The input DataFrame must contain the required features in the same order
> and with the same preprocessing used during training.

---



## Validation Methodology

The MVP was evaluated with two complementary temporal protocols.

### Temporal holdout

The main reported result trains on data through 2014 and evaluates on data
from 2015 onward:

| Metric | EventHorizon Demand MVP |
|---|---:|
| Overall WAPE | **31.03%** |
| Training period | Through 2014 |
| Evaluation period | 2015 onward |
| Feature set | No `lag_365` |
| Baseline | Naive lag-7 |

### Walk-forward stability

To assess temporal stability, the model was retrained annually and evaluated
on the following year:

| Test year | Protocol |
|---|---|
| 2013 | Train on earlier years; evaluate on 2013 |
| 2014 | Train through 2013; evaluate on 2014 |
| 2015 | Train through 2014; evaluate on 2015 |
| 2016 | Train through 2015; evaluate on 2016 |

This expanding-window walk-forward protocol retrains the model for each test
year, using only information available before that year.

### Uncertainty and significance

Uncertainty was estimated with a moving block bootstrap using seven-day blocks
and 1,000 resamples.

For each store, the paired WAPE gap between the naive lag-7 baseline and
EventHorizon Demand was computed within each bootstrap resample. This paired
approach evaluates whether the baseline improvement remains consistent under
temporal resampling.

### Honest limitations

- The main 31.03% WAPE result comes from a temporal holdout evaluation.
- The annual walk-forward analysis assesses temporal stability, but its
  yearly results should not be pooled as a single out-of-sample benchmark.
- Results are specific to the M5 Walmart benchmark.
- Technical accuracy does not establish business ROI.

### Honest limitations

- This is a temporal-holdout result, not a walk-forward validation across
  multiple time origins.
- The validation period was used for model evaluation and reporting.
- Results are specific to the M5 Walmart benchmark and do not guarantee
  performance on another retailer, market or data distribution.
- Technical accuracy does not establish business ROI.

---

## Philosophy

| Principle | Description |
|---|---|
|  **Rigor over hype** | Results are accompanied by validation methodology and limitations. |
|  **Honest validation** | No random shuffling, no data leakage and no cherry-picked results. |
|  **Practical deployment** | A compact LightGBM model that can run on CPU. |
|  **Transparent metrics** | Evaluation methodology and results are documented openly. |

---

## Use Cases

-  Supermarkets forecasting daily product demand
-  Fashion retailers planning seasonal inventory
-  E-commerce inventory replenishment
-  Distribution centers estimating weekly shipments
-  Multi-store retail forecasting
-  Supply-chain demand planning

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | LightGBM |
| Language | Python |
| Data processing | pandas, NumPy |
| Validation | Block bootstrap and paired gap bootstrap |
| Model serialization | joblib |

---

## Links

| Project | URL |
|---|---|
|  EventHorizon-AI Hub | [github.com/EventHorizon-ia/EventHorizon](https://github.com/EventHorizon-ia/EventHorizon) |
|  Demand Research | [github.com/EventHorizon-ia/demand-m5](https://github.com/EventHorizon-ia/demand-m5) |
|  Honest Validation Toolkit | [github.com/EventHorizon-ia/honest-validation-toolkit](https://github.com/EventHorizon-ia/honest-validation-toolkit) |
|  Crypto Research | [github.com/EventHorizon-ia/crypto-h0-edge](https://github.com/EventHorizon-ia/crypto-h0-edge) |

---

## License

MIT License.
