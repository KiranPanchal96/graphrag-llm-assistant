import os
import sys
import json
from dotenv import load_dotenv

from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as DeprecatedChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# Ollama import (added!)
from langchain_community.llms import Ollama

# Load environment variables and project path
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.s05a_retriever import get_retriever
from src.prompts.prompt_loader import get_example_prompt, get_suffix_prompt

PROMPT_SOURCE = os.getenv("PROMPT_SOURCE", "langfuse").lower()

# --- Langfuse Setup (conditional) ---
if PROMPT_SOURCE == "langfuse":
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_API_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_API_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    langfuse_handler = CallbackHandler()
else:
    langfuse = None
    langfuse_handler = None

# --- Load Few-Shot Examples ---
EXAMPLES_PATH = "data/prompts/few_shot_examples.json"
with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
    few_shot_examples = json.load(f)

# --- Use prompt loader helpers ---
example_prompt = get_example_prompt(langfuse)
suffix_prompt_string = get_suffix_prompt(langfuse)

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    suffix=suffix_prompt_string,
    input_variables=["context", "question"]
)

# --- Build RAG Chain with LCEL ---
retriever = get_retriever()
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ---- LLM Selection Logic (OpenAI vs Ollama) ----
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"  # Set USE_OLLAMA=1 to use local LLM via Ollama

USE_DOCKER = os.getenv("USE_DOCKER", "0") == "1"

if USE_DOCKER:
    OLLAMA_URL = "http://host.docker.internal:11434"
else:
    OLLAMA_URL = "http://localhost:11434"

if USE_OLLAMA:
    # You can specify the model name via env (defaults to llama3)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    llm = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    print(f"🔄 Using Ollama LLM: {OLLAMA_MODEL}")
else:
    try:
        llm = ChatOpenAI(temperature=0)
        print("🟢 Using OpenAI LLM")
    except Exception:
        llm = DeprecatedChatOpenAI(temperature=0)
        print("🟡 Using Deprecated OpenAI LLM")

rag_chain = (
    RunnableMap({
        "documents": retriever,
        "question": RunnablePassthrough()
    })
    | RunnableMap({
        "context": lambda x: format_docs(x["documents"]),
        "question": lambda x: x["question"],
        "documents": lambda x: x["documents"]
    })
    | RunnableMap({
        "answer": few_shot_prompt | llm | StrOutputParser(),
        "sources": lambda x: x["documents"]
    })
)

# --- Utility for Pipeline Use ---
def run_rag(question: str, ground_truth: str = None) -> dict:
    """
    Run the RAG chain with tracing and optional ground_truth for eval.
    """
    context_docs = retriever.invoke(question)
    context_text = format_docs(context_docs)

    config = {
        "run_name": "main_rag_chain",
        "inputs": {"query": question},
        "outputs": {},  # Filled in by the callback after LLM run
        "metadata": {"ground_truth": ground_truth or "No ground truth provided."}
    }
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]

    # Run the RAG chain and capture result
    result = (
        RunnableMap({
            "context": lambda _: context_text,
            "question": lambda _: question,
            "documents": lambda _: context_docs
        })
        | RunnableMap({
            "answer": few_shot_prompt | llm | StrOutputParser(),
            "sources": lambda x: x["documents"]
        })
    ).invoke(
        {},
        config=config
    )
    return result

# --- Dev/Test Block ---
if __name__ == "__main__":
    test_query = "What are the highest ROI practices mentioned in the document?"
    ground_truth = "The highest ROI practices mentioned in the document are reading books and AI personal projects"
    result = run_rag(test_query, ground_truth=ground_truth)
    print(f"\n💬 Answer:\n{result['answer']}\n")
    print("📚 Sources:")
    for i, doc in enumerate(result['sources'], 1):
        snippet = doc.page_content.strip().split("\n")[0][:100]
        print(f"{i}. {snippet}...")

