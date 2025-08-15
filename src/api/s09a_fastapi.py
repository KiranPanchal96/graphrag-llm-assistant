from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.s06a_rag_chain import run_rag  # Import your pipeline

app = FastAPI()


# Pydantic model for request body
class RAGRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    ground_truth: Optional[str] = Field(None, max_length=1000)


# Health check endpoint
@app.get("/")
def root() -> dict[str, str]:
    return {"status": "Life strategy RAG FastAPI server is running!"}


# Main RAG endpoint
@app.post("/rag")
def rag_endpoint(request: RAGRequest) -> dict[str, Any]:
    # Pass empty string when ground_truth is None to satisfy run_rag's str type
    result = run_rag(request.query, ground_truth=request.ground_truth or "")
    return {
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result["sources"]],
    }
