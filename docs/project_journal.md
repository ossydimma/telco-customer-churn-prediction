# Project Journal

## 2026-06-10

Downloaded the Telco Customer Churn dataset from Kaggle.

Set up the full project structure, initialised a git repository, and committed
the skeleton before writing any analysis code.

Dataset shape: 7,043 rows × 21 columns.

telco.csv contains no null values according to `.isnull().sum()` — however this was
misleading. `TotalCharges` is stored as object despite being a billing amount.
Stripping whitespace and checking for blank entries revealed 11 rows with empty strings.
Converting with `pd.to_numeric(errors='coerce')` confirmed these become NaN.
Checking tenure for those 11 rows showed all values are 0 — new customers who had not
yet been billed. Decision: drop in cleaning. Imputing a billing figure for unbilled
customers would be fabricating data.

The target variable `Churn` has a 73% No / 27% Yes split. This is not extreme but
imbalanced enough that accuracy will be a useless metric. ROC-AUC will be the primary
measure, with precision and recall tracked to understand the false alarm vs missed
churner trade-off.

One column worth flagging early: `SeniorCitizen` is already encoded as int64 (0/1)
while every other binary column uses Yes/No strings. No encoding needed for this one.

Seven service columns use three values — Yes, No, and either "No internet service" or
"No phone service". These are effectively binary columns with extra labelling and will
be collapsed to 0/1 in cleaning.

Initial column classification documented in the notebook — binary Yes/No, binary 0/1,
3-value service columns, multi-class categorical, numeric stored as object, and
identifier to drop.

Next step: EDA — distributions for tenure, MonthlyCharges and TotalCharges, churn rate
by Contract, InternetService and PaymentMethod, and boxplots to check for outliers.


## 2026-06-12

Completed exploratory data analysis on the Telco Customer Churn dataset.

### Churn distribution
1,869 customers churned out of 7,043 — a 26.5% positive rate.
Class imbalance confirmed. Accuracy will not be a useful metric. ROC-AUC is the primary measure.

### Numeric features
Tenure is the clearest separator between churned and retained customers.
Churned customers have a median tenure of roughly 10 months; retained customers around 38 months.
Monthly charges are higher on average for churned customers.
Total charges are lower for churned customers despite the higher monthly rate — they leave before spend accumulates.
No outliers requiring removal were found in any numeric column.

### Contract type
Month-to-month customers churn at 42.7%.
One-year contracts: 11.3%. Two-year contracts: 2.8%.
Contract length is the single strongest categorical predictor identified in EDA.

### Internet service
Fiber optic customers churn at 41.9% — more than double the DSL rate of 19.0%.
Customers with no internet service churn at only 7.4%.
This may reflect pricing or service quality issues specific to the fiber product.

### Payment method
Electronic check customers churn at 45.3% — the highest of all payment methods.
Automatic payment methods (bank transfer, credit card) cluster between 15–17%.
Electronic check may be a proxy for lower engagement or less committed customers.

### Senior citizens
Senior citizens churn at 41.7% vs 23.6% for non-seniors.
A meaningful difference that will likely appear in feature importance.

### Tenure groups
0–12 months: 47.4% churn rate — the highest-risk window.
13–24 months: 28.7%. 25–48 months: 20.4%. 49–72 months: 9.5%.
The pattern is monotonically decreasing — the longer a customer stays, the less likely they are to leave.

### Correlation heatmap
Tenure and TotalCharges are strongly positively correlated (expected — longer customers accumulate more spend).
MonthlyCharges and TotalCharges are moderately correlated.
ChurnFlag shows the strongest negative correlation with tenure — confirming EDA findings.
Contract type and tenure are also negatively correlated — month-to-month customers have shorter tenures on average.

### Next step
Data cleaning (03_data_cleaning.ipynb) — drop the 11 blank TotalCharges rows, encode all binary columns, collapse three-value service columns to 0/1, encode remaining categoricals, drop customerID.

## 2026-06-14

Completed data cleaning on the Telco Customer Churn dataset.

### Rows removed
Dropped 11 rows where TotalCharges was a blank string. These were invisible to
`.isnull().sum()` — caught in notebook 01 by inspecting whitespace values directly.
All 11 had tenure = 0. Imputing a billing figure for customers who were never billed
would be fabricating data. Decision: drop. Dataset goes from 7,043 to 7,032 rows.

### Column removals
Dropped customerID — identifier column, carries no predictive signal.

### Binary encoding
Five columns were pure Yes/No strings: Partner, Dependents, PhoneService,
PaperlessBilling, Churn. Mapped Yes → 1, No → 0 using a loop with .map().
Churn distribution after encoding: 5,163 retained / 1,869 churned — consistent
with EDA figures.

### Service column collapsing
Seven service columns used three values: Yes, No, and either "No internet service"
or "No phone service". The distinction between No and No internet/phone service carries
no additional information for the model — both mean the customer does not have the feature.
Mapped Yes → 1, all other values → 0.

### Remaining categoricals
gender encoded as Male → 1, Female → 0. Near-even split: 3,549 male, 3,483 female.
InternetService, Contract, and PaymentMethod one-hot encoded using pd.get_dummies
with drop_first=False and dtype=int. Keeping all categories avoids hiding information
and keeps the encoding interpretable. This expands the dataset from 20 to 27 columns.

### Column ordering
Churn was at index 16 after encoding. Moved to last position so X / y splits
in the modelling notebook can use clean iloc slicing without specifying column names.

### Final state
7,032 rows · 27 columns · 0 missing values · all int64 or float64.
Saved to data/processed/telco_clean.csv.

### Next step
Feature engineering (04_feature_engineering.ipynb) — create interaction and ratio
features that capture what the raw columns can't express individually.

## 2026-06-17

Completed feature engineering on the Telco Customer Churn dataset.

### Features built and kept
AvgMonthlySpend = TotalCharges / tenure. Captures historical average spend, which can
differ from current MonthlyCharges if a customer's plan changed over time. Correlation
with Churn: 0.192 — on par with MonthlyCharges itself.

HasProtectionBundle: binary flag for customers holding 2 or more of OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport. Customers without the bundle churn at 32.9%
vs 16.8% for those with it — roughly double, even though the raw correlation coefficient
(-0.178) understates the gap since it only captures linear relationships.

IsNewCustomer: binary flag for tenure ≤ 12 months. The strongest engineered feature —
0.320 correlation, ranking 3rd in the entire dataset. Churn rate for new customers (47.7%)
vs everyone else (17.1%) confirms the early-tenure risk window found in EDA.

### Feature tested and rejected
TotalServices — sum of all 7 service columns into a single count. Correlation with Churn
was -0.070, weaker than several of its own constituent columns (OnlineSecurity alone: -0.171).
Services carry unequal predictive weight; summing them dilutes the strong signals with the
weak ones (StreamingTV, StreamingMovies barely correlate with churn at all). Decision: drop,
keep the 7 individual binary columns so the model can weigh each independently.

### Process note
Initially commented out the rejected feature's code instead of removing it. Corrected this —
dead code with no explanation reads as an accident, not a decision. The right pattern is to
delete the code and document the reasoning in a markdown cell or here in the journal, the
same way StorePromoLift was handled in the Rossmann project.

### Final state
7,032 rows · 30 columns · 0 missing values · all numeric.
Saved to data/processed/telco_features.csv.

### Next step
Baseline modelling (05_baseline_model.ipynb) — train/test split with stratification given
the 73/27 class imbalance, baseline classifier, establish ROC-AUC benchmark.