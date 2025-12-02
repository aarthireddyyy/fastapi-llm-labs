# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)

class PredictResponse(BaseModel):
    prediction: str
    class_index: int
    probabilities: Optional[List[float]]
    ok: bool = True
