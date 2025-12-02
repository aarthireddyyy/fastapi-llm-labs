# app/main.py
from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse
from .ml import load_model, predict_from_features

app = FastAPI(title="FastAPI ML Day2")

@app.on_event("startup")
def startup_event():
    load_model("model.pkl")
    print("Model loaded.")

@app.get("/api/v1/health")
def health():
    return {"status": "ok"}

@app.get("/api/v1/predict")
def predict_info():
    """
    Helpful GET response for humans visiting the endpoint in a browser.
    Real predictions must be done via POST with JSON body like in the docs.
    """
    return {
        "detail": "This endpoint expects a POST with JSON body. Use POST /api/v1/predict with keys: sepal_length, sepal_width, petal_length, petal_width.",
        "example_request": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    }

@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        features = [req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]
        idx, label, probs = predict_from_features(features)
        return {"prediction": label, "class_index": idx, "probabilities": probs, "ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NOTE: no network calls or requests.* at module import time.
# Use the separate test script (tests/test_predict.py) after the server has started.
