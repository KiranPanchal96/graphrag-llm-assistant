"""
s05b_nwkx_graph_retriever.py
Builds and returns a NetworkX-based graph retriever using LangChain-compatible interface.
This is an alternative to vector retrieval for structured, relationship-aware search.
"""

import os

import networkx as nx
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class GraphKeywordRetriever(BaseRetriever):
    """
    A basic keyword-matching retriever over a NetworkX graph.
    Each node in the graph is assumed to have:
        - node['text']: a string of content
        - node['metadata']: optional dictionary for provenance or type
    """

    graph: nx.Graph = Field(repr=False)
    top_k: int = 3

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """
        Return up to `top_k` documents whose node text contains the query string.

        Args:
            query: Search string to match against node text.

        Returns:
            A list of LangChain Document objects.
        """
        query_lower = query.lower()
        results: list[Document] = []

        for node_id, data in self.graph.nodes(data=True):
            text = data.get("text", "")
            if query_lower in text.lower():
                doc = Document(
                    page_content=text,
                    metadata={"source": data.get("metadata", {}), "node": node_id},
                )
                results.append(doc)

        return results[: self.top_k]


def load_graph(graph_path: str = "data/graph/life_graph.gml") -> nx.Graph:
    """
    Loads a graph from a GraphML or GML file.

    Args:
        graph_path: Path to the .gml or .graphml file.

    Returns:
        The loaded NetworkX Graph.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the format is unsupported.
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found at path: {graph_path}")

    if graph_path.endswith(".gml"):
        return nx.read_gml(graph_path)
    elif graph_path.endswith(".graphml"):
        return nx.read_graphml(graph_path)
    else:
        raise ValueError("Unsupported graph format. Use .gml or .graphml")


def get_graph_retriever(
    graph_path: str = "data/graph/life_graph.gml", top_k: int = 3
) -> GraphKeywordRetriever:
    """
    Load a NetworkX graph from the given path and return a GraphKeywordRetriever.

    Args:
        graph_path: Path to a .gml or .graphml graph file.
        top_k: Maximum number of documents to retrieve.

    Returns:
        GraphKeywordRetriever instance.
    """
    graph = load_graph(graph_path)
    return GraphKeywordRetriever(graph=graph, top_k=top_k)


if __name__ == "__main__":
    retriever = get_graph_retriever()
    query = "ROI"
    docs = retriever.invoke(query)

    print(f"\n🔎 Query: {query}")
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Result {i}:\n{doc.page_content}")
        print(f"🔗 Metadata: {doc.metadata}")
