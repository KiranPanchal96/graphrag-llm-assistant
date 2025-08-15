"""
s09c_gradio_neo4j_graph_ui.py
Gradio-based interactive UI for the Life Strategy Neo4j Graph-RAG pipeline.
Uses Cypher-powered retrieval, reasoning, and source attribution.
"""

import os
import sys
import gradio as gr
from dotenv import load_dotenv

# Setup environment and import Graph-RAG pipeline
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.s06c_neo4j_graph_chain import run_graph_qa

# ---------------------------
# Graph-RAG Query Wrapper
# ---------------------------
def query_neo4j_graph_rag(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = run_graph_qa(question)
    except Exception as e:
        return f"❌ Error: {e}", ""

    answer = result.get("result", "No answer generated.")

    # Show the Cypher query used for transparency/debug
    intermediate = result.get("intermediate_steps")
    if isinstance(intermediate, dict) and "cypher_query" in intermediate:
        cypher_query = intermediate["cypher_query"]
    elif isinstance(intermediate, list) and len(intermediate) > 0:
        cypher_query = intermediate[0].get("cypher_query", str(intermediate[0]))
    else:
        cypher_query = str(intermediate)

    cypher_display = f"**Cypher Query Used:**\n```\n{cypher_query}\n```" if cypher_query else "No Cypher query available."

    # Show the full context nodes retrieved (if available)
    context_nodes = []
    def extract_context_nodes(intermediate):
        if isinstance(intermediate, dict) and "full_context" in intermediate:
            return intermediate["full_context"]
        elif isinstance(intermediate, list):
            for step in intermediate:
                if isinstance(step, dict) and "full_context" in step:
                    return step["full_context"]
        return []

    for i, entry in enumerate(extract_context_nodes(intermediate), 1):
        # Try to find any values in each entry that look like dicts or strings (id or node dict)
        for k, v in entry.items():
            if isinstance(v, dict) and "id" in v:
                context_nodes.append(f"**Node {i}:** `{v['id']}` ({k})")
            else:
                context_nodes.append(f"**Node {i}:** `{v}` ({k})")

    if context_nodes:
        context_display = "\n".join(context_nodes)
    else:
        context_display = "No source nodes retrieved."

    sources_output = cypher_display + "\n\n" + context_display

    return answer, sources_output

# ---------------------------
# UI Definition
# ---------------------------
examples = [
    "What activities improve mood?",
    "Which routines include reading-related activities?",
    "What are the highest ROI practices?",
    "What goals are supported by current activities?",
    "What items are considered essential?",
]

with gr.Blocks(title="Life Strategy Neo4j Graph-RAG UI") as demo:
    gr.Markdown("""
    # 🧠 Life Strategy QA (Neo4j Graph-RAG)
    Ask a question grounded in your life strategy knowledge graph. This system uses Neo4j for structured retrieval and answers with full query transparency.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Ask your question",
                placeholder="e.g., What activities improve mood?",
                lines=1
            )
            ask_button = gr.Button("🔍 Search")

        with gr.Column(scale=1):
            example_box = gr.Dropdown(choices=examples, label="🔎 Try an Example")
            use_example_btn = gr.Button("Use Example")

    with gr.Row():
        with gr.Column():
            answer_output = gr.Textbox(label="💬 Answer", lines=5, interactive=False)
            sources_output = gr.Markdown(label="📚 Cypher & Source Nodes")

    # Bind events
    ask_button.click(fn=query_neo4j_graph_rag, inputs=question_input, outputs=[answer_output, sources_output])
    use_example_btn.click(fn=lambda x: x, inputs=example_box, outputs=question_input)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, show_error=True)
