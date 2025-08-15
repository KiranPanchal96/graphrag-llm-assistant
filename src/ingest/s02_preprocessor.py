"""
s02_preprocessor.py
Splits raw documents into smaller text chunks using LangChain's text splitter.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

def preprocess_documents(documents, chunk_size=512, chunk_overlap=50):
    """
    Chunk documents into smaller pieces for embedding.
    
    Args:
        documents (List[Document]): List of LangChain Document objects.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: Chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# -------------------------
# Development / Test Block
# -------------------------
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    from src.ingest.s01_loader import load_documents  # Adjust if you're testing differently

    print("\n🔄 Loading documents from data/raw...")
    docs = load_documents("data/raw")

    print(f"✅ Loaded {len(docs)} document(s). Now chunking...")

    chunks = preprocess_documents(docs, chunk_size=512, chunk_overlap=50)
    print(f"📦 Created {len(chunks)} chunk(s).\n")

    # Preview a few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:500])  # First 500 characters
        print()
