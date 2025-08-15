"""
s01_loader.py
Recursively loads documents from various formats (PDF, HTML, DOCX) in the data/raw directory.
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)
import os

def load_documents(data_dir="data/raw"):
    """
    Recursively load and return all supported documents from the given directory.
    Supports PDF, HTML, and DOCX formats.
    """
    documents = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)

            # Choose loader based on file type
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".html"):
                loader = UnstructuredHTMLLoader(path)
            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                continue  # Skip unsupported files

            try:
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return documents

# ------------------------------
# Development/Test Block
# ------------------------------
if __name__ == "__main__":
    docs = load_documents()
    print(f"\n✅ Loaded {len(docs)} documents.\n")

    # Print a preview of first few chunks
    for i, doc in enumerate(docs[:3]):
        print(f"--- Document {i+1} ---")
        print(doc.page_content[:500])  # Show first 500 chars
        print()
