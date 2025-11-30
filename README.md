# Day 1 - FastAPI basics

Run:
> uvicorn main:app --reload --port 8000

Endpoints:
GET / -> welcome
POST /predict -> sample body {"name":"x","value":1.0}

# Day 2 - ML model serving

1. Train:
> python train_model.py

2. Serve:
> uvicorn main:app --reload --port 8001

POST /predict body: {"x": 3}
