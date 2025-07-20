import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/rf_telco_churn.pkl')
model = joblib.load(MODEL_PATH)

class CustomerFeatures(BaseModel):
    features: List[float]

@app.post('/predict')
def predict_churn(data: CustomerFeatures):
    # Expecting a list of features in the correct order
    X = pd.DataFrame([data.features])
    prediction = model.predict(X)[0]
    return {"churn_prediction": int(prediction)}

# To run: uvicorn src.api.predict_api:app --reload 