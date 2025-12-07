# Day 3 — LLM API with FastAPI + Ollama

## What this does

- Exposes a POST `/chat` endpoint via FastAPI.
- Takes JSON:
  - `message`
  - `system_prompt`
  - `max_tokens`
  - `temperature`
  - `model` (e.g. `qwen2.5:1.5b`)
- Forwards the request to `http://localhost:11434/api/generate` (Ollama).
- Returns the model's text response.

## Flow

Browser (index.html + JS)  
→ POST `http://127.0.0.1:8000/chat`  
→ `main.py` (`chat` endpoint)  
→ `services/ollama_service.py` (calls Ollama)  
→ Ollama generates tokens  
→ Response is sent back as plain text to the browser.

## How to run

```bash
# 1. Start Ollama desktop app and make sure a model is pulled:
ollama pull qwen2.5:1.5b

# 2. Start FastAPI backend
cd week1/day3-llama-fastapi
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Open frontend
# Open frontend/index.html in browser (or use Live Server in VS Code)
