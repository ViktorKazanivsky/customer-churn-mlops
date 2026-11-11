from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("models/latest_model.joblib")

app = FastAPI(title="Customer Churn Prediction API")
model = None


class ChurnRequest(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    Contract: str = "Month-to-month"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 850.0


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    df = pd.DataFrame([payload.model_dump()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_label": "high" if probability >= 0.5 else "low",
    }