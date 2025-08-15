"""
s07b_graph_pipeline.py
Runs the Graph-RAG pipeline via CLI using LCEL-based chain.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure project root is in path
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inference.s06b_nwkx_graph_chain import run_graph_rag


def run_graph_rag_pipeline(query: str) -> dict:
    return run_graph_rag(query)  # Returns dict with "answer" and "sources"


# -------------------------
# Dev CLI Block
# -------------------------
if __name__ == "__main__":
    print("🌐 Life Strategy Graph-RAG CLI\n")
    while True:
        user_query = input("❓ Ask a question (or type 'exit'): ")
        if user_query.strip().lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        result = run_graph_rag_pipeline(user_query)

        print(f"\n💬 Answer:\n{result['answer']}\n")
        print("📚 Source Nodes:")
        for i, doc in enumerate(result["sources"], 1):
            snippet = doc.page_content.strip().split("\n")[0][:100]
            print(f"{i}. {snippet} — {doc.metadata}")
