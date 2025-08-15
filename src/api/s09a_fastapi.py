import os
import sys
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# Setup environment and import RAG pipeline
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.s06a_rag_chain import run_rag  # Import your pipeline

app = FastAPI()

# Pydantic model for request body
class RAGRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    ground_truth: Optional[str] = Field(None, max_length=1000)

# Health check endpoint
@app.get("/")
def root():
    return {"status": "Life strategy RAG FastAPI server is running!"}

# Main RAG endpoint
@app.post("/rag")
def rag_endpoint(request: RAGRequest):
    # Note: Make sure run_rag expects `question`, not `query`
    result = run_rag(request.query, ground_truth=request.ground_truth)
    return {
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result["sources"]]
    }
