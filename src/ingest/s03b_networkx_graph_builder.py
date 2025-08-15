"""
s03b_graph_builder.py
Builds a NetworkX-based knowledge graph from preprocessed document chunks.
Nodes represent concepts or passages, and edges represent relationships or co-occurrence.
"""

import os
from uuid import uuid4

import networkx as nx

# LangChain
from langchain_core.documents import Document

from src.ingest.s01_loader import load_documents
from src.ingest.s02_preprocessor import preprocess_documents


def extract_entities_and_relationships(text: str) -> list[tuple[str, str]]:
    """
    Dummy rule-based entity linking. Replace with NLP or LLM-based method.
    Returns list of (entity, related_entity) pairs based on short lines.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    entities = [line for line in lines if len(line.split()) < 10]

    edges: list[tuple[str, str]] = []
    for i, source in enumerate(entities):
        for target in entities[i + 1 :]:
            edges.append((source, target))
    return edges


def build_knowledge_graph(documents: list[Document]) -> nx.Graph:
    """Constructs a NetworkX knowledge graph from document chunks."""
    G = nx.Graph()

    for doc in documents:
        doc_id = str(uuid4())
        text = doc.page_content.strip()
        metadata = doc.metadata

        node_id = f"doc_{doc_id}"
        G.add_node(node_id, text=text, metadata=metadata)

        # Extract co-occurrence relationships
        pairs = extract_entities_and_relationships(text)
        for e1, e2 in pairs:
            G.add_node(e1, text=e1)
            G.add_node(e2, text=e2)
            G.add_edge(node_id, e1)
            G.add_edge(e1, e2)

    return G


def save_graph(graph: nx.Graph, output_path: str = "data/graph/life_graph.gml") -> None:
    """Save the NetworkX graph to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_gml(graph, output_path)
    print(f"✅ Graph saved to {output_path}")


def main() -> None:
    """Load documents, build a graph, and save it to disk."""
    print("📚 Loading raw documents...")
    raw_docs = load_documents()

    print("🧩 Preprocessing (chunking)...")
    chunks = preprocess_documents(raw_docs, chunk_size=800, chunk_overlap=100)

    print(f"🔗 Building graph from {len(chunks)} chunks...")
    graph = build_knowledge_graph(chunks)

    print("💾 Saving graph to disk...")
    save_graph(graph)


if __name__ == "__main__":
    main()
