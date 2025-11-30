# week1-fastapi-llm/day2-ml-model-serving/train_model.py
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# toy dataset
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])  # y = 2*x

model = LinearRegression()
model.fit(X, y)

# Save model (this creates model.pkl)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved toy model to model.pkl")
