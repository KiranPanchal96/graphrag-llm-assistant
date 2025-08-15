"""
s03_embedder.py
Encodes text chunks using Sentence-BERT.
"""

from sentence_transformers import SentenceTransformer

def get_embedder(model_name="all-MiniLM-L6-v2"):
    """
    Load and return the Sentence-BERT embedding model.
    """
    return SentenceTransformer(model_name)

def embed_chunks(chunks, embedder):
    """
    Generate embeddings for each text chunk using the given embedder.
    
    Args:
        chunks (List[Document]): LangChain-style documents with page_content.
        embedder (SentenceTransformer): The embedding model.

    Returns:
        List[np.ndarray]: List of embedding vectors.
    """
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    return embeddings


# -------------------------
# Development / Test Block
# -------------------------
if __name__ == "__main__":
    import sys
    import os

    # Add src to path so we can import modules cleanly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    from src.ingest.s01_loader import load_documents
    from src.ingest.s02_preprocessor import preprocess_documents

    print("\n🔄 Loading and preprocessing documents...")
    docs = load_documents("data/raw")
    chunks = preprocess_documents(docs)

    print(f"✅ Loaded {len(docs)} documents and created {len(chunks)} chunks.\n")

    print("⚙️  Loading Sentence-BERT model...")
    embedder = get_embedder()

    print("🔗 Generating embeddings...")
    vectors = embed_chunks(chunks, embedder)
    print(f"✅ Generated {len(vectors)} embedding vectors.\n")

    # Preview first vector shape
    import numpy as np
    print(f"🔍 First vector shape: {np.array(vectors[0]).shape}")
