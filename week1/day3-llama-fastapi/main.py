from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from schemas import ChatRequest
from services.ollama_service import stream_from_ollama
import json

app = FastAPI(title="LLM Local Proxy (Ollama) - Clean Text Stream")

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    payload = {
        "model": req.model,
        "prompt": f"<s>[INST]{req.system_prompt}\n{req.message}[/INST]</s>",
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    def event_stream():
        """
        Assemble bytes from Ollama, parse JSON-lines, extract `response` tokens,
        and yield them as raw bytes so the client receives plain text tokens.
        """
        buffer = b""
        for chunk in stream_from_ollama(payload):
            buffer += chunk
            # Common pattern: Ollama sends newline-separated JSON objects.
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    # If line isn't valid JSON, forward it as-is (safe fallback)
                    yield line + b"\n"
                    continue
                # The token text is in "response"
                token = obj.get("response")
                if token is not None:
                    # yield token bytes (client sees plain text)
                    yield token.encode("utf-8")
        # flush any remaining bytes in buffer
        if buffer.strip():
            try:
                obj = json.loads(buffer.decode("utf-8"))
                token = obj.get("response")
                if token:
                    yield token.encode("utf-8")
            except Exception:
                yield buffer

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")