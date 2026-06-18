# Customer Churn Prediction

## Business Problem

Customer retention is one of the most important challenges businesses face.
Acquiring new customers is often more expensive than retaining existing ones,
making churn prediction a valuable business problem.

This project builds a machine learning model to predict whether a customer is likely
to leave a telecoms company based on demographics, account information, and service usage.
The goal is not only a working predictive model but identifying the factors that drive
churn so retention efforts can be targeted where they matter most.

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| File | Rows | Description |
|------|------|-------------|
| Telco-Customer-Churn.csv | 7,043 | Customer demographics, account info, services subscribed, and churn label |

> **Note:** Raw data files are not tracked in this repository.
> Download the dataset from Kaggle and place the CSV in `data/raw/`.

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
│   └── 07_evaluation.ipynb
│
├── .gitignore
└── README.md
```

---

## Results

*To be updated after modelling.*

---

## Key Findings from EDA

- **Contract type** is the strongest categorical churn driver — month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
- **Fiber optic** customers churn at 41.9% — more than double the DSL rate of 19.0%
- **Electronic check** payment method has a 45.3% churn rate — the highest of all payment types
- **Senior citizens** churn at 41.7% vs 23.6% for non-seniors
- **Short-tenure customers are the highest-risk group** — 47.4% churn rate in the first 12 months, falling to 9.5% after 4 years
- **Churned customers pay more per month but less in total** — they leave before charges accumulate

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

| Feature | Type | Description |
|---------|------|--------------|
| AvgMonthlySpend | float | TotalCharges / tenure — historical average monthly spend |
| HasProtectionBundle | binary | 1 if customer holds 2+ protective/support services |
| IsNewCustomer | binary | 1 if tenure ≤ 12 months — captures the highest-risk window identified in EDA |

One feature (TotalServices — a sum of all 7 service columns) was built, tested against
target correlation, and rejected: aggregating diluted signal rather than concentrating it.

---

## Model Results

| Model | ROC-AUC | Precision (Churn) | Recall (Churn) | Notes |
|-------|---------|--------------------|------------------|-------|
| Logistic Regression (baseline) | 0.8348 | 0.64 | 0.53 | Unscaled features, default threshold |

*Table will grow as later notebooks add tuned and alternative models.*

---

## Workflow

| Phase | Notebook | Status |
|-------|----------|--------|
| Data understanding | 01_data_understanding.ipynb | ✅ Complete |
| Exploratory analysis | 02_eda.ipynb | ✅ Complete |
| Data cleaning | 03_data_cleaning.ipynb | ✅ Complete |
| Feature engineering | 04_feature_engineering.ipynb | ✅ Complete |
| Baseline modelling | 05_baseline_model.ipynb | ✅ Complete |
| Model improvement | 06_model_improvement.ipynb | ⏳ |
| Final evaluation | 07_evaluation.ipynb | ⏳ |

---

## Evaluation Metric

**Primary:** ROC-AUC — measures the model's ability to rank churners above non-churners
across all classification thresholds. Chosen because the dataset is imbalanced (~27% positive
class) and accuracy alone would be misleading.

**Secondary:** Precision, Recall, F1 — to evaluate the trade-off between catching actual
churners and false alarms.

---

## Tools and Libraries

Python · pandas · numpy · matplotlib · seaborn · scikit-learn · XGBoost · LightGBM · Optuna

---

## Author

Built in public as a portfolio project while transitioning into data science.
Follow the build on [LinkedIn](https://www.linkedin.com/in/osita-jerry)

GitHub: [github.com/ossydimma/customer-churn-prediction](https://github.com/ossydimma/telco-customer-churn-prediction)