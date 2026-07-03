import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD MODEL FILES
# ======================
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# ======================
# TITLE
# ======================
st.title("🏥 Medical Cost Prediction")

st.write(
    "Predict annual medical costs based on patient health and insurance information."
)

st.markdown("---")

# ======================
# USER INPUT
# ======================
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 30)

    income = st.number_input(
        "Income",
        min_value=0.0,
        max_value=1000000.0,
        value=50000.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    chronic_count = st.number_input(
        "Chronic Disease Count",
        0,
        10,
        0
    )

    visits_last_year = st.number_input(
        "Visits Last Year",
        0,
        50,
        2
    )

with col2:
    medication_count = st.number_input(
        "Medication Count",
        0,
        20,
        1
    )

    risk_score = st.slider(
        "Risk Score",
        0.0,
        1.0,
        0.3
    )

    annual_premium = st.number_input(
        "Annual Premium",
        min_value=0.0,
        max_value=50000.0,
        value=2000.0
    )

    deductible = st.number_input(
        "Deductible",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )

    copay = st.number_input(
        "Copay",
        min_value=0.0,
        max_value=500.0,
        value=20.0
    )
    


# ======================
# CREATE INPUT DATAFRAME
# ======================
input_dict = {
    "age": age,
    "income": income,
    "bmi": bmi,
    "chronic_count": chronic_count,
    "visits_last_year": visits_last_year,
    "medication_count": medication_count,
    "risk_score": risk_score,
    "annual_premium": annual_premium,
    "deductible": deductible,
    "copay": copay,
}
input_df = pd.DataFrame([input_dict])

# ======================
# FEATURE ENGINEERING
# ======================

# utilization
input_df["visits_per_month"] = (
    input_df["visits_last_year"] / 12
)

input_df["utilization_score"] = (
    input_df["visits_last_year"] +
    (input_df["medication_count"] * 2)
)

# interaction
input_df["age_chronic_interaction"] = (
    input_df["age"] *
    input_df["chronic_count"]
)

# financial
input_df["premium_burden"] = (
    input_df["annual_premium"] /
    (input_df["income"] + 1)
)

input_df["deductible_to_income"] = (
    input_df["deductible"] /
    (input_df["income"] + 1)
)

input_df["out_of_pocket_risk"] = (
    input_df["deductible"] +
    input_df["copay"]
)

# bmi
input_df["is_obese"] = (
    input_df["bmi"] >= 30
).astype(int)

input_df["is_overweight"] = (
    input_df["bmi"] >= 25
).astype(int)

# polynomial
input_df["age_squared"] = (
    input_df["age"] ** 2
)

input_df["bmi_squared"] = (
    input_df["bmi"] ** 2
)

input_df["chronic_squared"] = (
    input_df["chronic_count"] ** 2
)

input_df["visits_squared"] = (
    input_df["visits_last_year"] ** 2
)

# log features
input_df["log_income"] = np.log1p(
    input_df["income"]
)

input_df["log_premium"] = np.log1p(
    input_df["annual_premium"]
)

input_df["log_visits"] = np.log1p(
    input_df["visits_last_year"]
)
# ======================
# ALIGN COLUMNS
# ======================
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# ======================
# SCALE FEATURES
# ======================

# ======================
# PREDICT
# ======================
if st.button("🔮 Predict Medical Cost"):

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    st.subheader("💰 Prediction Result")

    st.success(
        f"Estimated Annual Medical Cost: ${prediction:,.2f}"
    )