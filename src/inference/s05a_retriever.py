"""
s05_retriever.py
Loads FAISS vector store and returns a LangChain retriever.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever(index_path="embeddings/faiss", model_name="all-MiniLM-L6-v2"):

    local_model_path = "my_local_model"
    embedding_model = HuggingFaceEmbeddings(model_name=local_model_path)
    vectorstore = FAISS.load_local(
        index_path, 
        embedding_model, 
        allow_dangerous_deserialization=True  # ✅ required after LangChain 0.2+
    )
    return vectorstore.as_retriever()

if __name__ == "__main__":
    retriever = get_retriever()
    print("✅ Retriever loaded successfully.")
