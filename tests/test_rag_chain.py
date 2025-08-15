import os
import sys
import pytest

# Adjust import to ensure src is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.s06a_rag_chain import run_rag

def test_run_rag_basic():
    query = "What is the capital of France?"
    result = run_rag(query)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert any("paris" in doc.page_content.lower() for doc in result["sources"]) or "paris" in result["answer"].lower()

def test_run_rag_with_ground_truth():
    query = "What is the capital of France?"
    ground_truth = "The capital of France is Paris."
    result = run_rag(query, ground_truth=ground_truth)
    # You can be as strict as you want here:
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    # Optional: check ground_truth appeared in metadata (if you expose it)
    # Or just check if the answer is relevant
    assert "paris" in result["answer"].lower()

if __name__ == "__main__":
    pytest.main([__file__])
