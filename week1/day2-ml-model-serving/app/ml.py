# app/ml.py
import joblib
import numpy as np
from typing import Tuple

_MODEL = None
_TARGET_NAMES = None

def load_model(path="model.pkl"):
    global _MODEL, _TARGET_NAMES
    obj = joblib.load(path)
    _MODEL = obj["model"]
    _TARGET_NAMES = obj.get("target_names")
    return _MODEL

def predict_from_features(features: list) -> Tuple[int, str, list]:
    # features: [sepal_length, sepal_width, petal_length, petal_width]
    x = np.array(features).reshape(1, -1)
    idx = int(_MODEL.predict(x)[0])
    probs = None
    if hasattr(_MODEL, "predict_proba"):
        probs = _MODEL.predict_proba(x)[0].tolist()
    class_name = _TARGET_NAMES[idx] if _TARGET_NAMES is not None else str(idx)
    return idx, class_name, probs
