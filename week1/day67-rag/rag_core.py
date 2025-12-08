from pathlib import Path
from typing import Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


def _load_pdfs(folder_path: Path):
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path.resolve()}")

    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


def _split_docs(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def _build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def _build_components() -> Tuple:
    docs = _load_pdfs(DATA_DIR)
    chunks = _split_docs(docs)
    vectorstore = _build_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="qwen2.5:1.5b")
    return retriever, llm


# Build once at import time
print("Initializing RAG components for API...")
RETRIEVER, LLM = _build_components()
print("RAG components ready.")


def get_rag_answer(question: str) -> str:
    docs = RETRIEVER.get_relevant_documents(question)
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question: {question}

If the answer is not in the context, say you don't know.
Answer:
"""

    answer = LLM.invoke(prompt)
    return str(answer)
