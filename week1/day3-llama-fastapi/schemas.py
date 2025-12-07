from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str = "qwen2.5:1.5b"
    message: str
    system_prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.0