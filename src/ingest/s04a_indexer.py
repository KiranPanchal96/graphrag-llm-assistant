"""
s04_indexer.py
Builds and saves FAISS index using LangChain-compatible chunks and embeddings.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def build_faiss_index(
    chunks: list[Document],
    model_name: str = "all-MiniLM-L6-v2",
    save_path: str = "embeddings/faiss",
) -> None:
    """
    Build FAISS index from document chunks using HuggingFace embeddings.
    """
    # embedder = HuggingFaceEmbeddings(model_name=model_name)
    # vectorstore = FAISS.from_documents(chunks, embedder)

    embedder = HuggingFaceEmbeddings(model_name="/app/my_local_model")
    vectorstore = FAISS.from_documents(chunks, embedder)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved to: {save_path}")


# -------------------------
# Development / Test Block
# -------------------------
if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.ingest.s01_loader import load_documents
    from src.ingest.s02_preprocessor import preprocess_documents

    print("\n📥 Loading documents...")
    docs: list[Document] = load_documents("data/raw")

    print("✂️ Preprocessing into chunks...")
    chunks: list[Document] = preprocess_documents(docs)

    print("📦 Building FAISS index...")
    build_faiss_index(chunks)

    print("\n🎉 Done.")
