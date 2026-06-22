# Customer Churn Prediction

## Business Problem

Customer retention is one of the most important challenges businesses face.
Acquiring new customers is often more expensive than retaining existing ones,
making churn prediction a high-value business problem.

This project builds a machine learning model to predict whether a customer is
likely to leave a telecoms company based on demographics, account information,
and service usage. The goal is not only a working predictive model but identifying
the factors that drive churn so retention efforts can be targeted where they matter most.

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| File | Rows | Description |
|------|------|-------------|
| Telco-Customer-Churn.csv | 7,043 | Customer demographics, account info, services subscribed, and churn label |

> Raw data files are not tracked in this repository.
> Download from Kaggle and place the CSV in `data/raw/`.

**Target variable:** `Churn` — whether a customer left (Yes / No)

---


## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                        # Original CSVs (not tracked by git)
│   └── processed/                  # Cleaned outputs (not tracked by git)
│
├── docs/
│   └── project_journal.md          # Day-by-day decisions and learnings
│
├── images/                         # Charts saved from notebooks
├── models/                         # Saved model files (not tracked by git)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_model_improvement.ipynb
|   |── 07_tuning.ipynb
│   └── 08_evaluation.ipynb
│
├── src/
│   └── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md

```

---

---

## Results

| Metric | Default threshold (0.5) | Recommended threshold (0.35) |
|--------|------------------------|------------------------------|
| ROC-AUC | 0.8404 | 0.8404 (threshold-invariant) |
| Precision (churners) | 0.630 | 0.553 |
| Recall (churners) | 0.519 | 0.738 |
| F1 (churners) | 0.569 | 0.632 |
| Churners caught | 194 / 374 | 276 / 374 |

**Final model:** LightGBM (Optuna tuned, 5-fold CV, 100 trials)
**Recommended threshold:** 0.35 — raises recall from 51.9% to 73.8%, catching 81 additional churners

---

## Model Progression

| Stage | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic Regression (baseline) | 0.8348 | ConvergenceWarning — scale mismatch |
| Logistic Regression + StandardScaler | 0.8340 | Converged cleanly via Pipeline |
| LightGBM (default) | 0.8344 | No tuning |
| Random Forest (200 trees) | 0.8217 | No tuning |
| **LightGBM (Optuna tuned)** | **0.8404** | 100 trials · 5-fold CV · best model |

---

## Key Findings from EDA

- **Contract type** is the strongest categorical churn driver — month-to-month: 42.7% vs two-year: 2.8%
- **Fiber optic** customers churn at 41.9% — more than double DSL at 19.0%
- **Electronic check** payment has the highest churn rate at 45.3%
- **Senior citizens** churn at 41.7% vs 23.6% for non-seniors
- **First 12 months are the highest-risk window** — 47.4% churn rate, falling to 9.5% after 4 years
- **Churned customers pay more monthly but less in total** — they leave before spend accumulates

---

## Data Cleaning Summary

| Step | Column(s) | Issue | Decision |
|------|-----------|-------|----------|
| Drop rows | TotalCharges | 11 blank strings not caught by `.isnull()` | Dropped — all had tenure = 0, imputing is fabrication |
| Drop column | customerID | Identifier, no signal | Dropped |
| Binary encode | Partner, Dependents, PhoneService, PaperlessBilling, Churn | Yes/No strings | Yes → 1, No → 0 |
| Collapse to binary | 7 service columns | Yes / No / No internet service | Yes → 1, all else → 0 |
| Binary encode | gender | Male/Female string | Male → 1, Female → 0 |
| One-hot encode | InternetService, Contract, PaymentMethod | Multi-class, no ordinal order | get_dummies → 10 new columns |
| Reorder | Churn | Was at index 16 | Moved to last column |

**Result:** 7,043 → 7,032 rows · 21 → 27 columns · 0 nulls · all numeric

---

## Feature Engineering Summary

| Feature | Type | Description | Correlation with Churn |
|---------|------|-------------|------------------------|
| AvgMonthlySpend | float | TotalCharges / tenure — historical average monthly spend | 0.192 |
| HasProtectionBundle | binary | 1 if customer holds 2+ of OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport | -0.178 |
| IsNewCustomer | binary | 1 if tenure ≤ 12 months | 0.320 |

**Rejected:** TotalServices (sum of 7 service columns) — correlation -0.070, weaker than several
constituent columns. Aggregation diluted signal rather than concentrating it.

---

## Business Interpretation

Four actionable high-risk profiles identified:

1. **Month-to-month customers in their first year** — highest-risk combination in the dataset.
   Incentivising an early contract upgrade from month-to-month to one-year dramatically reduces risk.

2. **Fiber optic customers with high monthly charges** — likely reflects perceived value mismatch.
   Proactive service quality check or loyalty discount for this segment.

3. **Electronic check payment customers** — proxy for low engagement.
   Migration to automatic payment correlates with churn rates dropping from 45% to 15–17%.

4. **Senior citizens without support services** — concentrated risk segment.
   Bundled support offer targeting seniors reduces both churn and service friction.

---

## Loading the Model

```python
import joblib
import pandas as pd

model = joblib.load('models/tuned_lgbm.joblib')

# Input must have the same 29 features used in training
# See src/predict.py for a complete example with sample input
y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= 0.35).astype(int)  # recommended threshold
```

---

## Workflow

| Phase | Notebook | Status |
|-------|----------|--------|
| Data understanding | 01_data_understanding.ipynb | ✅ Complete |
| Exploratory analysis | 02_eda.ipynb | ✅ Complete |
| Data cleaning | 03_data_cleaning.ipynb | ✅ Complete |
| Feature engineering | 04_feature_engineering.ipynb | ✅ Complete |
| Baseline modelling | 05_baseline_model.ipynb | ✅ Complete |
| Model improvement | 06_model_improvement.ipynb | ✅ Complete |
| Hyperparameter tuning | 07_tuning.ipynb | ✅ Complete |
| Final evaluation | 08_evaluation.ipynb | ✅ Complete |

---

## Evaluation Metric

**Primary:** ROC-AUC — measures the model's ability to rank churners above non-churners
across all thresholds. Chosen because accuracy is misleading at 73/27 class imbalance.

**Secondary:** Precision, Recall, F1 at threshold 0.35 — to quantify the churner
detection tradeoff in operational terms.

---

## Tools and Libraries

Python · pandas · numpy · matplotlib · seaborn · scikit-learn · LightGBM · Optuna · SHAP · joblib

---

## Author

Built in public as a portfolio project while transitioning into data science.

[LinkedIn](https://www.linkedin.com/in/osita-jerry) · [GitHub](https://github.com/ossydimma/telco-customer-churn-prediction)