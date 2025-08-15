"""
s05a_retriever.py
Loads FAISS vector store and returns a LangChain retriever.
"""

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings


def get_retriever(
    index_path: str = "embeddings/faiss",
    model_name: str = "all-MiniLM-L6-v2",
    local_model_path: Optional[str] = "my_local_model",
) -> BaseRetriever:
    """
    Load a FAISS index from disk and return a LangChain retriever.

    Args:
        index_path: Path to the FAISS index folder.
        model_name: Name of the HuggingFace embedding model.
        local_model_path: Optional local model path override.

    Returns:
        BaseRetriever: LangChain retriever instance.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=local_model_path or model_name)
    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,  # Required after LangChain 0.2+
    )
    return vectorstore.as_retriever()


if __name__ == "__main__":
    retriever = get_retriever()
    print("✅ Retriever loaded successfully.")
