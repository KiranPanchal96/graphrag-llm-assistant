import pytest

from src.inference.s06a_rag_chain import run_rag


def test_run_rag_basic() -> None:
    query = "What is the capital of France?"
    result = run_rag(query)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert (
        any("paris" in doc.page_content.lower() for doc in result["sources"])
        or "paris" in result["answer"].lower()
    )


def test_run_rag_with_ground_truth() -> None:
    query = "What is the capital of France?"
    ground_truth = "The capital of France is Paris."
    result = run_rag(query, ground_truth=ground_truth)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert "paris" in result["answer"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
