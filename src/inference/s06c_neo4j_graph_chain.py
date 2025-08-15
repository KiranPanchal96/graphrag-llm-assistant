"""
s06c_neo4j_graph_chain.py
Runs QA over your Neo4j knowledge graph using LLM + Cypher.
With runtime Cypher patch for label quoting (handles spaces in labels).
"""

import os
import sys
import json
import re
from dotenv import load_dotenv

from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate

# Setup
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def get_neo4j_graph():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE")  # optional if using default
    )

graph = get_neo4j_graph()

# --- PATCH Cypher for labels with spaces ---
def quote_labels_with_spaces(cypher: str) -> str:
    # Match :Label with spaces, not already quoted
    # (:Life framework) -> (:`Life framework`)
    # Works for multi-word labels
    return re.sub(r':([A-Za-z0-9_]+(?: [A-Za-z0-9_]+)+)', r':`\1`', cypher)

# Patch the Neo4jGraph's query method
orig_query = graph.query
def patched_query(cypher: str, *args, **kwargs):
    cypher = quote_labels_with_spaces(cypher)
    return orig_query(cypher, *args, **kwargs)
graph.query = patched_query
# --- END PATCH ---

# Print and write graph schema (optional)
schema_str = graph.get_schema
print("📘 Graph Schema:\n")
print(schema_str)
os.makedirs("data/graph", exist_ok=True)
with open("data/graph/neo4j_graph.txt", "w") as f:
    f.write(schema_str)

# Load examples from JSON (for better Cypher generation)
with open("data/cypher/examples.json", "r") as f:
    examples = json.load(f)

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

def run_graph_qa(question: str):
    """Returns dict with 'result' (LLM answer) and 'intermediate_steps' (Cypher/query info)."""
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
