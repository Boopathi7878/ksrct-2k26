import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# LOAD MODEL & SCALER
# ============================

model = joblib.load("E:\\Projects\\ML\\KSR\\data\\anomaly_model.pkl")
scaler = joblib.load("E:\\Projects\\ML\\KSR\\data\\scaler.pkl")

st.title("🛡️ Self Learning Cyberattack Detection Bot")

st.write("Upload network traffic CSV to detect anomalies")

# ============================
# FILE UPLOAD
# ============================

uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    # Remove label column if exists
    if "Attack Type" in df.columns:
        df = df.drop("Attack Type", axis=1)

    # ============================
    # PREPROCESS
    # ============================

    X_scaled = scaler.transform(df)

    # ============================
    # PREDICTION
    # ============================

    predictions = model.predict(X_scaled)

    # Convert output
    predictions = np.where(predictions == -1, 1, 0)

    # Risk Score
    scores = model.decision_function(X_scaled)
    risk_score = (scores - scores.min()) / (scores.max() - scores.min())

    df["Prediction"] = predictions
    df["Risk Score"] = risk_score

    # ============================
    # DISPLAY RESULTS
    # ============================

    st.subheader("Detection Results")
    st.write(df.head())

    attack_count = sum(predictions)
    normal_count = len(predictions) - attack_count

    st.metric("Detected Attacks", attack_count)
    st.metric("Normal Traffic", normal_count)

    st.bar_chart(df["Prediction"].value_counts())
