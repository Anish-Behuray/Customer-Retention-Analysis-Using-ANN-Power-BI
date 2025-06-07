import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === Load model and preprocessing objects ===
model = load_model("C:/Users/ANISH/Desktop/telecom_project/ann_model.h5")
scaler = joblib.load("C:/Users/ANISH/Desktop/telecom_project/scaler.pkl")
expected_cols = joblib.load("C:/Users/ANISH/Desktop/telecom_project/input_columns.pkl")

# Columns to scale
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# === Streamlit Page Config ===
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("🔍 Customer Churn Prediction")
st.markdown("Fill in the details below to predict if the customer will churn.")

# === Input Form ===
with st.form("churn_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)

    with col2:
        tenure = st.number_input("Tenure (months)", min_value=0, step=1)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col3:
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.01)

    with col4:
        StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])



    submitted = st.form_submit_button("Predict Churn")


# === Prediction Logic ===
if submitted:
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_data])

    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

    # Replace and cast to int explicitly to avoid FutureWarning and dtype problems
    for col in yes_no_cols:
        input_df[col] = input_df[col].replace({
            'Yes': 1,
            'No': 0,
            'No phone service': 0,
            'No internet service': 0
        }).astype(int)

    input_df['gender'] = input_df['gender'].replace({'Female': 1, 'Male': 0}).astype(int)

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Add missing expected columns and set to 0 (as float)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0

    # Reorder columns exactly as expected
    input_df = input_df[expected_cols]

    # Ensure all columns are float type before scaling/model input
    input_df = input_df.astype(float)

    # Scale numeric columns
    NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

    # Convert to numpy array
    input_array = input_df.values

    # Predict
    prediction = model.predict(input_array)
    is_churn = int(prediction[0][0] > 0.5)

    # Display result
    st.subheader("🔎 Prediction Result")
    if is_churn:
        st.error("⚠️ This customer is likely to **churn**.")
    else:
        st.success("✅ This customer is **not likely** to churn.")
