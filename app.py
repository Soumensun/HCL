# app.py
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import os

MODEL_PATH = "models/best_pipeline.joblib"
FEATURE_META = "models/features.json"

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Telecom Customer Churn — Quick Predict")

# Load model & metadata
@st.cache_resource
def load_model_and_meta():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = None
    if os.path.exists(FEATURE_META):
        with open(FEATURE_META, "r") as f:
            meta = json.load(f)
    return model, meta

model, meta = load_model_and_meta()

if model is None:
    st.warning("Model not found. Train the model first (run training.py).")
    st.stop()

st.markdown("Enter feature values below or upload a CSV for batch predictions.")

# If we have feature lists, build a simple form
feature_names = []
if meta:
    feature_names = meta.get("numeric_features", []) + meta.get("categorical_features", [])

use_csv = st.checkbox("Upload CSV for batch prediction", value=False)

if use_csv:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        try:
            preds = model.predict(df)
            out = pd.DataFrame({"prediction": preds})
            if hasattr(model, "predict_proba"):
                out["prob_churn"] = model.predict_proba(df)[:,1]
            st.write("Predictions (first 50 rows):")
            st.dataframe(out.head(50))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    if not feature_names:
        st.info("Model metadata (feature names) not available. Upload CSV for batch prediction instead.")
    else:
        st.write("Enter single-row values:")
        cols = st.columns(2)
        values = {}
        for i, name in enumerate(feature_names):
            container = cols[i % 2]
            # default numeric input; if categorical you can type
            if name in (meta.get("numeric_features") or []):
                values[name] = container.number_input(name, value=0.0)
            else:
                values[name] = container.text_input(name, value="")

        if st.button("Predict"):
            row = pd.DataFrame([values])
            try:
                pred = model.predict(row)[0]
                st.success(f"Churn prediction: {int(pred)}")
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(row)[:,1])
                    st.info(f"Probability of churn: {prob:.3f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
