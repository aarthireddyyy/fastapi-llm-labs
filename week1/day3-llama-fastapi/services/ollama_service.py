import json
from typing import Generator


def stream_from_ollama(payload: dict) -> Generator[bytes, None, None]:
    """
    Minimal stubbed stream generator for Ollama responses.
    Replace this with a real streaming HTTP request to your Ollama daemon
    (eg. `requests.post(..., stream=True)` or `httpx.AsyncClient.stream`) when ready.
    """
    # Simple demo: return a single JSON-line with a `response` token.
    example = {"response": "Hello from Ollama (stub)\n"}
    yield (json.dumps(example) + "\n").encode("utf-8")
