from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document  # if this errors, remove type hints


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


def load_pdfs(folder_path: Path) -> List[Document]:
    """
    Load all PDFs from the folder and return a list of Document objects.
    """
    all_docs = []
    pdf_files = list(folder_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path.resolve()}")

    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")
    return all_docs

def split_docs(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_vectorstore(docs, persist_dir: Path) -> Chroma:
    """
    Create embeddings and a Chroma vector store from documents.
    """
    print("Creating embeddings and building Chroma vector store...")

    embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print(f"Vector store built and persisted to {persist_dir.resolve()}")
    return vectorstore


def build_rag_components():
    """
    Prepare vectorstore, retriever, and LLM (Ollama).
    If a persisted Chroma DB already exists, you could load it instead.
    """
    # 1. Load PDFs
    docs = load_pdfs(DATA_DIR)

    # 2. Split into chunks
    chunks = split_docs(docs)

    # 3. Build vectorstore
    vectorstore = build_vectorstore(chunks, CHROMA_DIR)

    # 4. Build retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. LLM
    llm = Ollama(model="qwen2.5:1.5b")

    return retriever, llm

def format_context(docs) -> str:
    """
    Join retrieved docs into a single context string.
    """
    contents = [d.page_content for d in docs]
    return "\n\n---\n\n".join(contents)


def main():
    print("🔹 Building RAG pipeline...")
    print(f"PDF folder: {DATA_DIR.resolve()}")
    print(f"Chroma DB folder: {CHROMA_DIR.resolve()}")

    retriever, llm = build_rag_components()
    print("✅ RAG components ready.")
    print("Ask questions about your PDFs. Type 'q' to quit.\n")

    while True:
        question = input("❓ Your question: ").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("Exiting. Bye!")
            break

        if not question:
            continue

        # 1. Retrieve docs
        docs = retriever.get_relevant_documents(question)
        context = format_context(docs)

        # 2. Build prompt
        prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question: {question}

If the answer is not in the context, say you don't know.
Answer:
"""

        # 3. Call LLM
        print("\n💭 Thinking...")
        answer = llm.invoke(prompt)
        print("\n🧠 Answer:")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
