"""
s05c_neo4j_graph_retriever.py
Provides a Neo4j keyword retriever returning LangChain Document objects.
Now matches on BOTH node labels and property values.
"""

import os
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

class Neo4jKeywordRetriever:
    """
    Keyword-matching retriever over a Neo4j knowledge graph.
    Returns a list of Document objects matching the keyword.
    Matches BOTH node labels and property values (case-insensitive).
    """
    def __init__(self, graph: Neo4jGraph, top_k: int = 3):
        self.graph = graph
        self.top_k = top_k

    def retrieve(self, query: str):
        cypher = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE toLower(label) CONTAINS toLower('{query}'))
           OR any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower('{query}'))
        OPTIONAL MATCH (n)-[r]->(m)
        OPTIONAL MATCH (p)-[r2]->(n)
        WITH n, labels(n) AS labels,
             collect(DISTINCT {{type: type(r), direction: 'out', target_label: labels(m), target_id: m.id}}) AS out_rels,
             collect(DISTINCT {{type: type(r2), direction: 'in', source_label: labels(p), source_id: p.id}}) AS in_rels
        RETURN n, labels, out_rels, in_rels
        LIMIT {self.top_k}
        """
        results = self.graph.query(cypher)
        docs = []
        for record in results:
            node = record['n']
            labels = record.get('labels', [])
            out_rels = [rel for rel in record.get('out_rels', []) if rel.get('type') is not None]
            in_rels = [rel for rel in record.get('in_rels', []) if rel.get('type') is not None]
            content = "\n".join([f"{k}: {v}" for k, v in node.items()])
            metadata = {
                "labels": labels,
                "properties": node,
                "outgoing_relationships": out_rels,
                "incoming_relationships": in_rels
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

def get_neo4j_graph():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE")
    )

# Dev/test
if __name__ == "__main__":
    graph = get_neo4j_graph()
    retriever = Neo4jKeywordRetriever(graph=graph, top_k=5)
    query = "Activity"
    docs = retriever.retrieve(query)
    print(f"\n🔎 Query: {query}")
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Result {i}:\n{doc.page_content}")
        print(f"🔗 Labels: {doc.metadata.get('labels')}")
        print(f"🔗 Outgoing relationships:")
        for rel in doc.metadata.get("outgoing_relationships", []):
            print(f"    → ({rel.get('type')}) {rel.get('target_label')} id={rel.get('target_id')}")
        print(f"🔗 Incoming relationships:")
        for rel in doc.metadata.get("incoming_relationships", []):
            print(f"    ← ({rel.get('type')}) {rel.get('source_label')} id={rel.get('source_id')}")
