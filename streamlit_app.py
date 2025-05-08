import streamlit as st
import pandas as pd
import joblib

# Load the trained model once
@st.cache_resource
def load_model(path):
    return joblib.load(path)

MODEL_PATH = 'models/best_model.pkl'
model = load_model(MODEL_PATH)

# Grab the exact feature names from the model
feature_names = list(model.feature_names_in_)

st.title("📶 Telecom Churn Predictor")
st.write("Fill in the customer details below and hit Predict to see their churn risk.")

# --- User inputs ---
gender = st.selectbox("Gender:", ['Female', 'Male'])
senior_citizen = st.checkbox("Senior Citizen")  # True → 1, False → 0
partner = st.selectbox("Has Partner?", ['No', 'Yes'])
dependents = st.selectbox("Has Dependents?", ['No', 'Yes'])
tenure = st.slider("Tenure (months):", 0, 72, 12)
phone_service = st.selectbox("Phone Service?", ['No', 'Yes'])
multiple_lines = st.selectbox("Multiple Lines?", ['No', 'Yes', 'No phone service'])
internet_service = st.selectbox("Internet Service:", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security?", ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox("Online Backup?", ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox("Device Protection?", ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox("Tech Support?", ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV?", ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies?", ['No', 'Yes', 'No internet service'])
contract = st.selectbox("Contract:", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing?", ['No', 'Yes'])
payment_method = st.selectbox(
    "Payment Method:",
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
)
monthly_charges = st.number_input("Monthly Charges ($):", min_value=0.0, value=70.0)
total_charges = st.number_input(
    "Total Charges ($):",
    min_value=0.0,
    value=monthly_charges * (tenure if tenure > 0 else 1)
)

if st.button("Predict churn risk"):
    # Initialize all features to zero
    row = {f: 0 for f in feature_names}

    # Fill numeric and binary features
    row['gender'] = 1 if gender == 'Male' else 0
    row['SeniorCitizen'] = 1 if senior_citizen else 0
    row['Partner'] = 1 if partner == 'Yes' else 0
    row['Dependents'] = 1 if dependents == 'Yes' else 0
    row['tenure'] = tenure
    row['PhoneService'] = 1 if phone_service == 'Yes' else 0
    row['PaperlessBilling'] = 1 if paperless_billing == 'Yes' else 0
    row['MonthlyCharges'] = monthly_charges
    row['avg_monthly_charge'] = total_charges / (tenure if tenure > 0 else 1)

    # One-hot multi-class features
    selections = {
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method
    }
    for prefix, choice in selections.items():
        key = f"{prefix}_{choice}"
        if key in row:
            row[key] = 1

    # Tenure group dummies (drop-first)
    if 12 < tenure <= 24 and 'tenure_group_13-24m' in row:
        row['tenure_group_13-24m'] = 1
    elif 24 < tenure <= 48 and 'tenure_group_25-48m' in row:
        row['tenure_group_25-48m'] = 1
    elif tenure > 48 and 'tenure_group_49-72m' in row:
        row['tenure_group_49-72m'] = 1

    # Build DataFrame in correct order
    feature_df = pd.DataFrame([row], columns=feature_names)

    # Predict and display
    prob = model.predict_proba(feature_df)[0, 1]
    st.success(f"Estimated churn probability: {prob:.1%}")
