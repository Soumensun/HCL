# ensembling.py
# small helpers if you want to load multiple models and compare predictions
import joblib
import numpy as np

def load_pipeline(path):
    return joblib.load(path)

def predict_proba_for_pipeline(pipeline, df):
    try:
        return pipeline.predict_proba(df)[:,1]
    except Exception:
        return pipeline.predict(df)
