"""
s07c_neo4j_graph_pipeline.py
Runs the Neo4j Graph-RAG pipeline via CLI using the LLM+Cypher QA chain.
Just prints the answer and Cypher query used.
"""

import os
import sys
from dotenv import load_dotenv

# Set up the environment and path
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inference.s06c_neo4j_graph_chain import run_graph_qa

def run_graph_rag_pipeline(query: str) -> dict:
    """Convenience wrapper for Neo4j Graph QA pipeline."""
    return run_graph_qa(query)

def main():
    print("🌐 Neo4j Graph-RAG CLI\n")
    while True:
        user_query = input("❓ Ask a question (or type 'exit'): ")
        if user_query.strip().lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        try:
            result = run_graph_rag_pipeline(user_query)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        print(f"\n💬 Answer:\n{result.get('result')}\n")
        print("📥 Cypher Query Used:")
        intermediate = result.get("intermediate_steps")
        if isinstance(intermediate, dict) and "cypher_query" in intermediate:
            print(intermediate["cypher_query"])
        elif isinstance(intermediate, list) and len(intermediate) > 0:
            print(intermediate[0].get("cypher_query", intermediate[0]))
        else:
            print(intermediate)
        print()

if __name__ == "__main__":
    main()
