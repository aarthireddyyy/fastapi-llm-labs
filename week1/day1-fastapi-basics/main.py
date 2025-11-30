# week1-fastapi-llm/day1-fastapi-basics/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    value: float

@app.get("/")
def read_root():
    return {"message": "Hello from Day 1 - FastAPI basics"}

@app.post("/predict")
def predict(item: Item):
    # toy prediction: multiply value
    return {"name": item.name, "pred": item.value * 2}
