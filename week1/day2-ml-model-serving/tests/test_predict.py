# tests/test_predict.py
import requests
import time
import sys

# URL of your running server
URL = "http://127.0.0.1:8000/api/v1/predict"

payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Optional: short wait to allow server to finish startup
time.sleep(1)

try:
    r = requests.post(URL, json=payload, timeout=5)
except Exception as e:
    print("Request failed:", e)
    sys.exit(1)

print("Status code:", r.status_code)
try:
    print("JSON response:", r.json())
except ValueError:
    print("Response text:", r.text)
