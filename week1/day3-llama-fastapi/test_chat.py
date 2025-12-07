import requests

url = "http://127.0.0.1:8000/chat"

payload = {
    "message": "Give me 3 ideas for weekend coding projects.",
    "system_prompt": "You are a helpful assistant.",
    "max_tokens": 150,
    "temperature": 0.8,
    "model": "qwen2.5:1.5b"
}

resp = requests.post(url, json=payload)
print("Status:", resp.status_code)
print("Body:\n", resp.text)
