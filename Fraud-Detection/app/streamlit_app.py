import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file containing credit card transactions.")

# ----------------------------
# Load Model and Scaler
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------------------
# File Upload
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# ----------------------------
# Prediction Logic
# ----------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Remove target column if present
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    required_columns = [
        "Time", "Amount",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7",
        "V8", "V9", "V10", "V11", "V12", "V13",
        "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns: {', '.join(missing_columns)}"
        )
        st.stop()

    # ----------------------------
    # Preprocessing
    # ----------------------------

    data["scaled_amount"] = scaler.transform(
        data[["Amount"]]
    )

    data["scaled_time"] = scaler.transform(
        data[["Time"]]
    )

    data = data.drop(
        columns=["Amount", "Time"]
    )

    expected_columns = [
        "scaled_amount",
        "scaled_time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7",
        "V8", "V9", "V10", "V11", "V12", "V13",
        "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28"
    ]

    data = data[expected_columns]

    # ----------------------------
    # Prediction
    # ----------------------------

    predictions = model.predict(data)
    
    probabilities = model.predict_proba(data)[:, 1]
    
    results = data.copy()
    
    results["Fraud_Probability"] = probabilities
    
    results["Prediction"] = [
        "Fraud" if pred == 1 else "Legitimate"
        for pred in predictions
        ]

    # ----------------------------
    # Metrics
    # ----------------------------

    total_transactions = len(results)

    fraud_count = (predictions == 1).sum()

    genuine_count = total_transactions - fraud_count

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraudulent",
        fraud_count
    )

    col3.metric(
        "Legitimate",
        genuine_count
    )

    # ----------------------------
    # Display Results
    # ----------------------------

    st.subheader("Prediction Results")

    st.dataframe(
        results[
            ["Fraud_Probability", "Prediction"]
        ].head(20)
    )

    fraud_transactions = results[
        results["Prediction"] == "Fraud"
    ]

    st.subheader("Detected Fraud Transactions")

    if len(fraud_transactions) > 0:
        st.dataframe(fraud_transactions)
    else:
        st.success("No fraudulent transactions detected.")

    # ----------------------------
    # Download Results
    # ----------------------------

    csv = results.to_csv(index=False)

    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )