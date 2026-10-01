"""
Standalone inference module.

Can be used in two ways:

1. CLI mode — pass a JSON file or JSON string:
       python predict.py '{"tenure": 1, "MonthlyCharges": 29.85, ...}'
       python predict.py sample_input.json

2. REST API mode — launches a FastAPI server:
       uvicorn predict:app --host 0.0.0.0 --port 8000
"""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from config import MODEL_PATH


# ── Pydantic schema ───────────────────────────────────────────────────────────

class CustomerProfile(BaseModel):
    """Validates and documents a single customer JSON payload."""

    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str | float | int

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def coerce_total_charges(cls, v):
        """Accept string, int, or float — always store as string so the
        pipeline's TotalChargesFixer handles the conversion uniformly."""
        return str(v)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(path=None):
    path = path or MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No model found at {path}. Run `python train.py` first."
        )
    return joblib.load(path)


def _load_threshold():
    """Load the optimized decision threshold; fall back to 0.5."""
    threshold_path = Path(MODEL_PATH).parent / "optimal_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            return json.load(f).get("optimal_threshold", 0.5)
    return 0.5


def predict_single(model, profile: dict, threshold: float = None) -> dict:
    """Run inference on one customer profile and return prediction + probability."""
    if threshold is None:
        threshold = _load_threshold()

    # Drop customerID before feeding into the pipeline — it isn't a feature
    features = {k: v for k, v in profile.items() if k != "customerID"}
    df = pd.DataFrame([features])

    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "threshold_used": round(threshold, 4),
    }


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="Accepts a customer profile and returns churn prediction with probability.",
    version="1.0.0",
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(profile: CustomerProfile):
    """Accept a customer JSON payload, return prediction + probability."""
    try:
        model = _get_model()
        result = predict_single(model, profile.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py '<json_string>' | python predict.py input.json")
        sys.exit(1)

    arg = sys.argv[1]

    # Try parsing as inline JSON first; fall back to reading as a file path
    try:
        payload = json.loads(arg)
    except (json.JSONDecodeError, ValueError):
        with open(arg) as f:
            payload = json.load(f)

    model = load_model()
    result = predict_single(model, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
