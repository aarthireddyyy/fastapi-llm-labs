# week1-fastapi-llm/day2-ml-model-serving/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

class Input(BaseModel):
    x: float

MODEL_PATH = "model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.get("/")
def root():
    return {"message": "Day 2 - model serving", "model_loaded": bool(model)}

@app.post("/predict")
def predict(inp: Input):
    if model is None:
        return {"error": "model not found. Run train_model.py to create model.pkl"}
    pred = model.predict([[inp.x]])[0]
    return {"x": inp.x, "pred": float(pred)}
