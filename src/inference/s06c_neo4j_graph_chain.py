"""
s06c_neo4j_graph_chain.py
Runs QA over your Neo4j knowledge graph using LLM + Cypher.
With runtime Cypher patch for label quoting (handles spaces in labels).
"""

import json
import os
import re
from typing import Any

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI


def get_neo4j_graph() -> Neo4jGraph:
    """
    Create and return a Neo4jGraph instance using environment variables.
    """
    return Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),  # optional if using default
    )


graph: Neo4jGraph = get_neo4j_graph()


# --- PATCH Cypher for labels with spaces ---
def quote_labels_with_spaces(cypher: str) -> str:
    """
    Quote Neo4j labels that contain spaces.

    Example:
        (:Life framework) -> (:`Life framework`)
    """
    return re.sub(r":([A-Za-z0-9_]+(?: [A-Za-z0-9_]+)+)", r":`\1`", cypher)


# Patch the Neo4jGraph's query method
orig_query = graph.query


def patched_query(cypher: str, *args: Any, **kwargs: Any) -> Any:
    """
    Patch wrapper around Neo4jGraph.query to handle labels with spaces.
    """
    cypher = quote_labels_with_spaces(cypher)
    return orig_query(cypher, *args, **kwargs)


graph.query = patched_query
# --- END PATCH ---

# Print and write graph schema (optional)
schema_str: str = graph.get_schema
print("📘 Graph Schema:\n")
print(schema_str)
os.makedirs("data/graph", exist_ok=True)
with open("data/graph/neo4j_graph.txt", "w", encoding="utf-8") as f:
    f.write(schema_str)

# Load examples from JSON (for better Cypher generation)
with open("data/cypher/examples.json", encoding="utf-8") as f:
    examples: list[dict[str, Any]] = json.load(f)

# LLM and Cypher prompt
llm = ChatOpenAI(temperature=0, model="gpt-4")
cypher_prompt = PromptTemplate.from_template(
    template=(
        "Given the following Neo4j schema:\n"
        "{schema}\n\n"
        "Generate a Cypher query that answers the user's question.\n"
        "- Only return valid Cypher.\n"
        "- No explanations or natural language.\n"
    )
).partial(schema=schema_str)

# GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_generation_prompt=cypher_prompt,
    examples=examples,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)


def run_graph_qa(question: str) -> dict[str, Any]:
    """
    Run a Cypher-based QA query over the Neo4j graph.

    Args:
        question (str): Natural language question.

    Returns:
        Dict[str, Any]: Result containing 'result' and optionally 'intermediate_steps'.
    """
    return chain.invoke({"query": question})


# Dev/test
if __name__ == "__main__":
    test_question = "What activities improve mood?"
    print(f"\n🔍 Test Question: {test_question}\n")
    answer = run_graph_qa(test_question)
    print("\n💬 Answer:")
    print(answer["result"])
    print("\n📥 Cypher Query Used:")
    intermediate = answer.get("intermediate_steps")
    if isinstance(intermediate, dict) and "cypher_query" in intermediate:
        print(intermediate["cypher_query"])
    elif isinstance(intermediate, list) and len(intermediate) > 0:
        print(intermediate[0].get("cypher_query", intermediate[0]))
    else:
        print(intermediate)
    print()
