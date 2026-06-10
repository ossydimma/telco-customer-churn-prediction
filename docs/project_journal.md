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