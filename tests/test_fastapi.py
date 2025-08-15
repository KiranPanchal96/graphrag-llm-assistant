import pytest
from fastapi.testclient import TestClient

from src.inference.s10a_fastapi import app

client = TestClient(app)


def test_rag_endpoint_basic() -> None:
    response = client.post("/rag", json={"query": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_rag_endpoint_with_ground_truth() -> None:
    query = "What is the capital of France?"
    ground_truth = "The capital of France is Paris."
    response = client.post("/rag", json={"query": query, "ground_truth": ground_truth})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    # Optionally check if answer mentions Paris (depends on your RAG pipeline)
    assert "paris" in data["answer"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
