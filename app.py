# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Telecom Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn:")

# Create input fields for all features
account_weeks = st.number_input("Account Weeks (how long the account has been active)", min_value=0, value=100)
contract_renewal = st.checkbox("Contract Renewed Recently?")  # checkbox for binary input
data_plan = st.checkbox("Has Data Plan?")                     # checkbox for binary input
data_usage = st.number_input("Data Usage (GB/month)", min_value=0.0, value=0.0, format="%.2f")
cust_serv_calls = st.number_input("Customer Service Calls (in last month)", min_value=0, value=0)
day_mins = st.number_input("Daytime Minutes Used", min_value=0.0, value=0.0, format="%.1f")
day_calls = st.number_input("Daytime Calls Made", min_value=0, value=0)
monthly_charge = st.number_input("Monthly Charge (USD)", min_value=0.0, value=50.0, format="%.2f")
overage_fee = st.number_input("Recent Overage Fee (USD)", min_value=0.0, value=0.0, format="%.2f")
roam_mins = st.number_input("Roaming Minutes (last month)", min_value=0.0, value=0.0, format="%.1f")

# Convert checkbox booleans to 0/1 integers
contract_renewal = 1 if contract_renewal else 0
data_plan = 1 if data_plan else 0

# Arrange inputs into a single sample for prediction
input_features = np.array([[
    account_weeks, contract_renewal, data_plan, data_usage, 
    cust_serv_calls, day_mins, day_calls, monthly_charge, overage_fee, roam_mins
]])
# Apply the same scaling to the input as the training data
input_features_scaled = scaler.transform(input_features)

if st.button("Predict Churn"):
    # Get prediction probability for class 1 (churn)
    churn_prob = model.predict_proba(input_features_scaled)[0][1]
    churn_pred = model.predict(input_features_scaled)[0]
    # Display the results
    st.subheader("Prediction:")
    result_text = "This customer is likely to **churn**." if churn_pred == 1 else "This customer is **not likely to churn**."
    st.write(result_text)
    st.write(f"Estimated Churn Probability: **{churn_prob*100:.1f}%**")
