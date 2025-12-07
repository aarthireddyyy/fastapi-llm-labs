# services/ollama_service.py
import requests


def stream_from_ollama(payload: dict):
    """
    Call Ollama's /api/generate endpoint and stream back raw chunks.
    This matches what main.event_stream expects: bytes lines.
    """
    url = "http://localhost:11434/api/generate"

    # IMPORTANT: stream=True to get chunks as they come
    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                # main.py expects bytes and splits on b"\n"
                yield line + b"\n"
