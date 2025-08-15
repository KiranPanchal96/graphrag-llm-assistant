"""
s05b_nwkx_graph_retriever.py
Builds and returns a NetworkX-based graph retriever using LangChain-compatible interface.
This is an alternative to vector retrieval for structured, relationship-aware search.
"""

import os
import sys
import networkx as nx
from typing import List
from pydantic import Field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# -------------------------
# Graph-Based Retriever
# -------------------------

class GraphKeywordRetriever(BaseRetriever):
    """
    A basic keyword-matching retriever over a NetworkX graph.
    Each node in the graph is assumed to have:
        - node['text']: a string of content
        - node['metadata']: optional dictionary for provenance or type
    """
    graph: nx.Graph = Field(repr=False)
    top_k: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_lower = query.lower()
        results = []

        for node_id, data in self.graph.nodes(data=True):
            text = data.get("text", "")
            if query_lower in text.lower():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": data.get("metadata", {}),
                        "node": node_id
                    }
                )
                results.append(doc)

        return results[:self.top_k]

# -------------------------
# Loader + Entry
# -------------------------

def load_graph(graph_path: str = "data/graph/life_graph.gml") -> nx.Graph:
    """
    Loads a graph from a GraphML or GML file.
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found at path: {graph_path}")

    if graph_path.endswith(".gml"):
        return nx.read_gml(graph_path)
    elif graph_path.endswith(".graphml"):
        return nx.read_graphml(graph_path)
    else:
        raise ValueError("Unsupported graph format. Use .gml or .graphml")


def get_graph_retriever(graph_path="data/graph/life_graph.gml", top_k=3) -> GraphKeywordRetriever:
    graph = load_graph(graph_path)
    return GraphKeywordRetriever(graph=graph, top_k=top_k)

# -------------------------
# Dev / Test Block
# -------------------------

if __name__ == "__main__":
    retriever = get_graph_retriever()
    query = "ROI"
    docs = retriever.invoke(query)

    print(f"\n🔎 Query: {query}")
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Result {i}:\n{doc.page_content}")
        print(f"🔗 Metadata: {doc.metadata}")
