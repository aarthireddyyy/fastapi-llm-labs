from fastapi import FastAPI
from pydantic import BaseModel

from rag_core import get_rag_answer

app = FastAPI(
    title="PDF RAG API",
    description="Ask questions over your PDFs using a RAG pipeline",
    version="0.1.0",
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    answer = get_rag_answer(req.question)
    return AnswerResponse(answer=answer)
