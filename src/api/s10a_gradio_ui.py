"""
s09a_gradio_ui.py
Gradio-based interactive UI for the Life Strategy RAG pipeline.
Uses HTTP to talk to FastAPI backend.
"""

import os
import sys
import gradio as gr
import requests

# ---------------------------
# Endpoint Config
# ---------------------------
API_HOST = os.getenv("FASTAPI_HOST", "localhost")
API_PORT = os.getenv("FASTAPI_PORT", "8000")
API_HEALTH_URL = f"http://{API_HOST}:{API_PORT}/"
API_RAG_URL = f"http://{API_HOST}:{API_PORT}/rag"

# ---------------------------
# FastAPI Health Check
# ---------------------------
def check_fastapi():
    try:
        resp = requests.get(API_HEALTH_URL, timeout=3)
        if resp.status_code == 200:
            print(f"✅ FastAPI is running at {API_HEALTH_URL}")
            return True
        else:
            print(f"⚠️  FastAPI responded with status {resp.status_code} at {API_HEALTH_URL}")
            return False
    except Exception as e:
        print(f"❌ FastAPI not running at {API_HEALTH_URL}. Please start your FastAPI server!\nError: {e}")
        sys.exit(1)

# ---------------------------
# RAG Query Wrapper
# ---------------------------
def query_rag(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    data = {"query": question}
    try:
        resp = requests.post(API_RAG_URL, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        answer = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])
        if not sources:
            sources_output = "No source documents were retrieved."
        else:
            sources_output = "\n\n".join(
                f"**Source {i+1}:**\n> {src[:400]}..." for i, src in enumerate(sources)
            )
        return answer, sources_output
    except Exception as e:
        return f"❌ Error contacting FastAPI: {e}", ""

# ---------------------------
# UI Definition
# ---------------------------
examples = [
    "What mindset is encouraged by the document’s core philosophy?",
    "How does the document suggest making impactful decisions?",
    "What role does peace play in the overall life strategy?",
    "What are the highest ROI practices mentioned in the document?",
    "How are time and energy treated in this framework?",
]

with gr.Blocks(title="Life Strategy RAG UI") as demo:
    gr.Markdown("""
    # 🧠 Life Strategy QA (RAG)
    Ask a question grounded in your life strategy documents. The system uses retrieval + reasoning with source attribution.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Ask your question",
                placeholder="e.g., What values guide decision-making in this strategy?",
                lines=1
            )
            ask_button = gr.Button("🔍 Search")

        with gr.Column(scale=1):
            example_box = gr.Dropdown(choices=examples, label="🔎 Try an Example")
            use_example_btn = gr.Button("Use Example")

    with gr.Row():
        with gr.Column():
            answer_output = gr.Textbox(label="💬 Answer", lines=5, interactive=False)
            sources_output = gr.Markdown(label="📚 Sources")

    # Bind events
    ask_button.click(fn=query_rag, inputs=question_input, outputs=[answer_output, sources_output])
    use_example_btn.click(fn=lambda x: x, inputs=example_box, outputs=question_input)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    check_fastapi()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
