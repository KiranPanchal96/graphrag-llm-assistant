"""
s06b_graph_chain.py
Standardized Graph-RAG pipeline using LangChain Expression Language (LCEL)
with few-shot examples, chain-of-thought reasoning, and source attribution.
"""

import os
import sys
import json
from dotenv import load_dotenv

from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as DeprecatedChatOpenAI

# Load environment and project path
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.s05b_nwkx_graph_retriever import get_graph_retriever

# -------------------------
# Load Few-Shot Examples
# -------------------------
EXAMPLES_PATH = "data/prompts/few_shot_examples.json"
with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
    few_shot_examples = json.load(f)

# -------------------------
# Prompt Templates
# -------------------------
example_prompt = PromptTemplate.from_template("""
Example:
Context:
{context}

Question:
{question}

Answer:
{answer}
""")

suffix_template = PromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Let's think step by step:
1. Identify the key elements in the context.
2. Reason through what is most relevant to the question.
3. Provide a clear and accurate answer.

Answer:
""")

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    suffix=suffix_template.template,
    input_variables=["context", "question"]
)

# -------------------------
# Graph Retriever
# -------------------------
retriever = get_graph_retriever()

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

try:
    llm = ChatOpenAI(temperature=0)
except Exception:
    llm = DeprecatedChatOpenAI(temperature=0)

# -------------------------
# Graph RAG LCEL Chain (safe version)
# -------------------------

def add_answer_to_input(inputs: dict) -> dict:
    answer = (few_shot_prompt | llm | StrOutputParser()).invoke({
        "context": format_docs(inputs["documents"]),
        "question": inputs["question"]
    })
    return {
        "answer": answer,
        "sources": inputs["documents"]
    }

llm_chain = RunnableLambda(add_answer_to_input)

# -------------------------
# Pipeline Entry Point
# -------------------------
def run_graph_rag(question: str) -> dict:
    context_docs = retriever.invoke(question)

    result = llm_chain.invoke({
        "question": question,
        "documents": context_docs
    })

    return result  # { "answer": ..., "sources": [...] }

# -------------------------
# Dev Test Block
# -------------------------
if __name__ == "__main__":
    # test_query = "What are the highest ROI practices mentioned in the document?"
    test_query = "ROI" # issue with long query
    result = run_graph_rag(test_query)

    print(f"\n💬 Answer:\n{result['answer']}\n")
    print("📚 Sources:")
    for i, doc in enumerate(result["sources"], 1):
        snippet = doc.page_content.strip().split("\n")[0][:100]
        print(f"{i}. {snippet} — {doc.metadata}")
