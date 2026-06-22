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

## 2026-06-18

Built the baseline model for the Telco Customer Churn dataset.

### Split strategy
80/20 train/test split, stratified on Churn. Stratification confirmed working —
train and test churn rates both landed at 26.58%, eliminating the risk of an
unrepresentative test set given the 73/27 class imbalance.

### Baseline result
Logistic Regression scored 0.8348 ROC-AUC on the test set. This is a strong, credible
number for this dataset — in line with typical published benchmarks — and becomes the
number every later model and tuning step must beat.

### Default threshold behavior
At the standard 0.5 cutoff: 80% accuracy, but only 53% recall on actual churners.
174 churners were missed out of 374 in the test set. This is expected with class
imbalance, not a model flaw — accuracy looks fine while quietly underperforming on
the class that actually matters. ROC-AUC was the right call as the primary metric
from day one.

### Convergence warning investigated
lbfgs failed to converge within 1000 iterations. Checked feature scales directly
rather than just increasing max_iter blindly: TotalCharges has a standard deviation
of ~2,276, while binary flags sit around 0.5. Three orders of magnitude apart on the
same loss surface. Scaling is the right fix, not more iterations — will apply
StandardScaler in the next notebook.

### Coefficient sanity check
Directions broadly agree with EDA and feature engineering findings. Contract_Month-to-month
and InternetService_Fiber optic carry the strongest churn-increasing coefficients;
Contract_Two year the strongest churn-decreasing one. IsNewCustomer holds a top-5 positive
coefficient, consistent with its 0.320 correlation found earlier.

Two coefficients (PhoneService, InternetService_No) showed unexpectedly large negative
magnitudes despite weak signal in earlier correlation checks — most likely a scale artifact
in unscaled logistic regression rather than genuine importance. Confirms that coefficient
magnitudes shouldn't be trusted at face value until features are scaled.

### Next step
Model improvement (06_model_improvement.ipynb) — scale features, test LightGBM and/or
Random Forest against the 0.8348 baseline, explore threshold tuning to improve churnerrecall.


## 2026-06-19

Completed model improvement notebook — tested scaling, Random Forest, and LightGBM
against the 0.8348 baseline.

### Results
All models clustered tightly. V1 (scaled LogReg): 0.8340. V3 (LightGBM default): 0.8344.
V2 (Random Forest): 0.8217. No model meaningfully beat the unscaled baseline by ROC-AUC.

This is a real finding, not a failure. The strongest predictors in this dataset —
contract type, tenure, payment method — are clean binary and encoded signals. Logistic
regression handles linear boundaries well, and this problem's decision boundary appears
to be largely linear at default settings. Tree models need tuning to find interactions
that aren't visible in the raw correlations.

### Scaling
Applying StandardScaler via Pipeline eliminated the ConvergenceWarning from notebook 05.
ROC-AUC barely moved (0.8348 → 0.8340) as expected — ROC-AUC is rank-based and
scale-invariant. But the pipeline is the correct object to save and deploy: scaler and
model travel together so inference-time data gets the same transformation training saw.

### class_weight='balanced' tested and rejected
Tried balanced weighting on both Random Forest and LightGBM. Both scores dropped.
At 73/27 imbalance, the models were already learning the minority class adequately.
Aggressive reweighting over-corrected and hurt overall ranking quality. Decision: revert.

### LightGBM feature importance observation
Split-count importance placed MonthlyCharges, TotalCharges, and AvgMonthlySpend as the
top three features — all measuring closely related aspects of customer spend. This is
partly a collinearity artifact and partly a known limitation of split-count importance,
which systematically undervalues binary features by offering fewer split points.
Contract_Month-to-month did not appear in the top 10 despite being the strongest
predictor throughout EDA and feature engineering. Importance metrics should be read
critically, not taken at face value.

### Threshold tuning
Tested thresholds from 0.30 to 0.50 on the scaled pipeline. Default 0.50 gives
precision 0.549, recall 0.535, F1 0.582. At 0.35: recall jumps to 0.717, F1 improves
to 0.620. Recommended threshold: 0.35. In a churn context, the cost of missing a
churner — lost recurring revenue, potential for churn to compound — exceeds the cost
of triggering an unnecessary retention offer.

### Next step
Hyperparameter tuning (07_tuning.ipynb) — Optuna search on LightGBM to test whether
deliberate tuning can beat the 0.8340 pipeline baseline.


## 2026-06-21

Completed hyperparameter tuning on LightGBM using Optuna with 5-fold
stratified cross-validation.

### Method
100 trials with StratifiedKFold (n_splits=5, shuffle=True, random_state=42) inside
the objective function. CV was added after the first tuning run — which evaluated
directly on the test set — to make the search more robust and prevent the optimiser
from finding params that happen to work on one specific test split.

### Result
Best CV score (training folds): 0.8495
Test set score (held-out, single evaluation): 0.8405

Tuned LightGBM beats the pipeline baseline by +0.0065 (0.8340 → 0.8405).
The CV score is higher than the test score — this is expected. CV averages
performance over 5 folds of training data; the test set is a different fixed
split. A gap of ~0.009 is normal and not a sign of overfitting.

### Stochastic variability
Optuna's score varied slightly across the three runs conducted during development
(0.8405, 0.8407, 0.8431 on test). This is expected — 100 trials sample a different
region of the parameter space each time. The consistent pattern was tuned LightGBM
landing between 0.840–0.843 across all runs, always above the 0.8340 baseline.
Final run locked in at 0.8405.

### Best params
n_estimators: 533, learning_rate: 0.0185, num_leaves: 56, max_depth: 9,
min_child_samples: 96, subsample: 0.596, colsample_bytree: 0.510,
reg_alpha: 9.618, reg_lambda: 0.212.

min_child_samples=96 (high) and num_leaves=56 (moderate) produce a more
regularised tree than defaults. This reduces over-splitting on continuous
features, which is consistent with the dataset's largely linear decision boundary.

### Feature importance
Continuous features still dominate split-count importance (TotalCharges, tenure,
MonthlyCharges, AvgMonthlySpend). The same caveat from notebook 06 applies —
split-count importance systematically undervalues binary features. Full SHAP
analysis in notebook 08 will give a more reliable view of what the model is
actually using.

### Next step
Final evaluation (08_evaluation.ipynb) — full classification report, confusion
matrix, threshold analysis, and SHAP feature importance on the tuned model.


## 2026-06-22

Completed final evaluation on the tuned LightGBM model and wrapped the project.

### Final model performance
ROC-AUC: 0.8404 on the held-out test set (1,407 customers, never seen during
training or tuning). At default threshold 0.5: recall 51.9%, F1 0.569.
At recommended threshold 0.35: recall 73.8%, F1 0.632 — catching 276 of 374
churners vs 194 at default.

### SHAP analysis
SHAP confirmed what EDA and feature engineering found throughout the project.
Contract type and tenure-related features (including IsNewCustomer) are the
dominant drivers. Split-count importance from notebooks 06 and 07 had
undervalued binary features — SHAP corrects this by measuring the actual
marginal contribution of each feature to each prediction rather than counting
tree splits. The two analyses are now consistent.

### Threshold decision
The 0.35 threshold is the recommended operating point. In a churn retention
context, the cost of missing a churner (lost recurring revenue) exceeds the
cost of a false alarm (one unnecessary retention offer). Moving from 0.5 to
0.35 catches 81 additional churners at the cost of 93 additional false alarms.
That tradeoff favours recall in virtually any realistic business valuation.

### Project wrap
predict.py added to src/ with full feature documentation and a sample input
representing a high-risk customer profile (month-to-month, fiber optic, new,
electronic check). requirements.txt added. README finalised with all sections
complete — business problem, results, model progression, data cleaning, feature
engineering, business interpretation, and loading instructions.

### Reflection
The project confirmed that on well-structured tabular business data, a simple
Logistic Regression baseline (0.8348) is genuinely competitive with a tuned
gradient boosting model (0.8404). The gap is real but modest. The more impactful
decisions were upstream — the threshold choice, the feature engineering, and the
decision not to impute TotalCharges for tenure=0 customers — rather than in
algorithm selection or tuning. That's a finding worth carrying forward.