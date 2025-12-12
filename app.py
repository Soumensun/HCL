"""app.py
Minimal FastAPI app to load saved model and serve a prediction endpoint.
Usage: uvicorn app:app --reload
POST /predict with JSON { "data": [[...]] } where inner list is feature vector matching preprocessing.
Note: For a production-ready API, you'd embed the preprocessing pipeline or apply it server-side.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

MODEL_PATH = 'models/best_model.pkl'

app = FastAPI(title='HCL Hackathon - Churn Predictor')

class PredictRequest(BaseModel):
    data: list  # list of feature vectors: [[f1,f2,...], [...]]

@app.on_event('startup')
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
        print('Failed to load model:', e)

@app.post('/predict')
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    arr = np.array(req.data)
    preds = model.predict(arr).tolist()
    probs = None
    try:
        probs = model.predict_proba(arr).tolist()
    except Exception:
        pass
    return {'predictions': preds, 'probabilities': probs}
