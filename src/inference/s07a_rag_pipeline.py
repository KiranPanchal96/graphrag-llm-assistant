"""
s07_rag_pipeline.py
Runs the RAG pipeline via CLI using LCEL-based chain.
"""

from src.inference.s06a_rag_chain import run_rag


def run_rag_pipeline(query: str) -> dict:
    return run_rag(query)  # ✅ returns dict with "answer" and "sources"


if __name__ == "__main__":
    print("🧠 Life Strategy RAG CLI\n")
    while True:
        user_query = input("❓ Ask a question (or type 'exit'): ")
        if user_query.strip().lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        result = run_rag_pipeline(user_query)
        print(f"\n💬 Answer:\n{result['answer']}\n")
        print("📚 Sources:")
        for i, doc in enumerate(result["sources"], 1):
            snippet = doc.page_content.strip().split("\n")[0][:100]
            print(f"{i}. {snippet}...")
