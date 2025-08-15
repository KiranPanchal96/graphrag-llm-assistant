"""
s09b_gradio_graph_ui.py
Gradio-based interactive UI for the Life Strategy Graph-RAG pipeline.
Uses structured graph retrieval, reasoning, and source attribution.
"""

import gradio as gr

from src.inference.s06b_nwkx_graph_chain import (  # ✅ changed to use full output
    run_graph_rag,
)


# ---------------------------
# Graph-RAG Query Wrapper
# ---------------------------
def query_graph_rag(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    result = run_graph_rag(question)
    answer = result.get("answer", "No answer generated.")

    sources = result.get("sources", [])
    if not sources:
        sources_output = "No source nodes were retrieved."
    else:
        sources_output = "\n\n".join(
            f"**Source {i+1}:** `Node: {doc.metadata.get('node', 'N/A')}`\n> {doc.page_content.strip()[:400]}..."
            for i, doc in enumerate(sources)
        )

    return answer, sources_output


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

with gr.Blocks(title="Life Strategy Graph-RAG UI") as demo:
    gr.Markdown(
        """
    # 🧠 Life Strategy QA (Graph-RAG)
    Ask a question grounded in your life strategy documents. This system uses a knowledge graph for structured retrieval and answers with source context.
    """
    )

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Ask your question",
                placeholder="e.g., What values guide decision-making in this strategy?",
                lines=1,
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
    ask_button.click(
        fn=query_graph_rag,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )
    use_example_btn.click(fn=lambda x: x, inputs=example_box, outputs=question_input)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, show_error=True)
