"""
s03c_neo4j_ingest.py
Parses and inserts documents into Neo4j using LangChain's LLMGraphTransformer.
"""

import os
import sys
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph  # ✅ updated to latest package
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# Setup
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingest.s01_loader import load_documents
from src.ingest.s02_preprocessor import preprocess_documents

# -----------------------------
# Config
# -----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 5
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # can override with gpt-4 if needed

# -----------------------------
# Connect to Neo4j
# -----------------------------
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL", "neo4j://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
    database=os.getenv("NEO4J_DATABASE", "neo4j")
)

# -----------------------------
# Load LLM + Graph Transformer
# -----------------------------
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
graph_transformer = LLMGraphTransformer(llm=llm)

# -----------------------------
# Ingest Function
# -----------------------------
def load_and_ingest_to_neo4j():
    print("📄 Loading and chunking documents...")
    raw_docs = load_documents()
    chunks = preprocess_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"🧩 Total Chunks: {len(chunks)}")

    all_graph_docs = []

    print("🔍 Extracting knowledge graph in batches...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"⏳ Processing batch {i + 1} to {i + len(batch)}...")
        try:
            graph_docs = graph_transformer.convert_to_graph_documents(batch)
            all_graph_docs.extend(graph_docs)
        except Exception as e:
            print(f"⚠️ Failed to process batch {i}-{i + len(batch)}: {e}")

    print(f"🔗 Total nodes/edges extracted: {len(all_graph_docs)}")
    print("🚀 Ingesting into Neo4j...")
    graph.add_graph_documents(all_graph_docs)
    print("✅ Done! Neo4j graph updated.")


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    load_and_ingest_to_neo4j()
