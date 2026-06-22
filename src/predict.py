"""
predict.py — Telco Customer Churn Prediction

Loads the tuned LightGBM model and returns churn probability and
binary prediction for a single customer record.

Expected features (29 total, in this order):
    gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    PaperlessBilling, MonthlyCharges, TotalCharges,
    InternetService_DSL, InternetService_Fiber optic, InternetService_No,
    Contract_Month-to-month, Contract_One year, Contract_Two year,
    PaymentMethod_Bank transfer (automatic), PaymentMethod_Credit card (automatic),
    PaymentMethod_Electronic check, PaymentMethod_Mailed check,
    AvgMonthlySpend, HasProtectionBundle, IsNewCustomer

Engineered features must be computed before calling predict():
    AvgMonthlySpend     = TotalCharges / tenure
    HasProtectionBundle = 1 if sum(OnlineSecurity, OnlineBackup,
                              DeviceProtection, TechSupport) >= 2 else 0
    IsNewCustomer       = 1 if tenure <= 12 else 0

Recommended classification threshold: 0.35
"""

import joblib
import pandas as pd

THRESHOLD = 0.35

MODEL_PATH = 'models/tuned_lgbm.joblib'

FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'AvgMonthlySpend', 'HasProtectionBundle', 'IsNewCustomer'
]

def predict(customer: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer : dict
        Dictionary with all 29 features as keys.

    Returns
    -------
    dict with keys:
        churn_probability : float  — model confidence (0–1)
        churn_prediction  : int    — 1 = likely to churn, 0 = likely to stay
        threshold_used    : float  — classification threshold applied
    """

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([customer])[FEATURES]
    proba = model.predict_proba(X)[:, 1][0]
    prediction = int(proba >= THRESHOLD)

    return {
        'churn_probability': round(float(proba), 4),
        'churn_prediction': prediction,
        'threshold_used': THRESHOLD
    }


if __name__ == '__main__':
    # Sample input — month-to-month fiber optic customer, 2 months tenure
    sample_customer = {
        'gender': 1,
        'SeniorCitizen': 0,
        'Partner': 0,
        'Dependents': 0,
        'tenure': 2,
        'PhoneService': 1,
        'MultipleLines': 0,
        'OnlineSecurity': 0,
        'OnlineBackup': 0,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'PaperlessBilling': 1,
        'MonthlyCharges': 70.70,
        'TotalCharges': 141.40,
        'InternetService_DSL': 0,
        'InternetService_Fiber optic': 1,
        'InternetService_No': 0,
        'Contract_Month-to-month': 1,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaymentMethod_Bank transfer (automatic)': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0,
        'AvgMonthlySpend': 70.70,
        'HasProtectionBundle': 0,
        'IsNewCustomer': 1
    }

    result = predict(sample_customer)
    print(f"Churn probability : {result['churn_probability']}")
    print(f"Churn prediction  : {'Will churn' if result['churn_prediction'] else 'Will stay'}")
    print(f"Threshold used    : {result['threshold_used']}")