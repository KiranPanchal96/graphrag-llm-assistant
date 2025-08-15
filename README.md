# 📚 graphrag-llm-assistant

A side project implementing **RAG (Retrieval-Augmented Generation)** and **GraphRAG** pipelines for answering questions using both **vector search** (FAISS + embeddings) and **graph-based reasoning** (NetworkX / Neo4j).

The project integrates **LangChain**, **FastAPI**, **Gradio UI**, and **Langfuse** for prompt management and evaluation.

---

## 🚀 Features

- **RAG pipeline** using FAISS vector store & HuggingFace embeddings  
- **GraphRAG pipeline** using NetworkX / Neo4j graph reasoning  
- **LLM-based evaluation** (QAEvalChain style) + BLEU, ROUGE-L, BERTScore  
- **Prompt management** with Langfuse  
- **Multiple LLM options** — OpenAI API or local Ollama  
- **FastAPI backend** for programmatic access  
- **Gradio UI** for interactive queries  
- **Dockerized deployment**  
- **Pre-commit hooks** for linting, formatting, and type checking  

---

## 🛠️ Tech Stack

- **LangChain** – Orchestration of retrieval + generation  
- **HuggingFace** – Embedding models  
- **FAISS** – Vector store for semantic search  
- **NetworkX / Neo4j** – Graph reasoning  
- **FastAPI** – REST API backend  
- **Gradio** – Interactive web UI  
- **Langfuse** – Prompt and evaluation tracking  
- **Ollama** – Local LLM hosting (optional)  
- **Docker** – Containerised deployment  
- **pre-commit** – Linting, formatting, type checking  

---

## 📂 Folder Structure
graphrag-llm-assistant/
│
├── data/ # Prompt examples, evaluation data (stripped in public repo)
├── embeddings/ # FAISS vector store and embedding model files
├── outputs/ # Generated evaluation results and logs
│
├── src/ # Source code
│ ├── api/ # FastAPI backend & Gradio UI
│ ├── evaluators/ # Evaluation scripts for RAG & GraphRAG
│ ├── inference/ # RAG & GraphRAG pipelines
│ ├── prompts/ # Prompt templates & loaders
│ └── init.py
│
├── tests/ # Unit and integration tests
│
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── docker-compose.yml # Docker Compose setup
├── Dockerfile # Docker image build file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── LICENSE # License file (MIT)
└── .gitignore # Git ignore rules

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
git clone https://github.com/KiranPanchal96/graphrag-llm-assistant.git

cd graphrag-llm-assistant

### 2️⃣ Create & Activate a Virtual Environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

### 3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

---

## ⚙️ Environment Variables

Create a .env file in the project root with the following variables:

<pre>
  OPENAI_API_KEY=your_openai_api_key
  LANGFUSE_PUBLIC_API_KEY=your_langfuse_public_key
  LANGFUSE_SECRET_API_KEY=your_langfuse_secret_key
  LANGFUSE_HOST=https://cloud.langfuse.com
  FASTAPI_HOST=localhost
  FASTAPI_PORT=8000
  USE_OLLAMA=0
  OLLAMA_MODEL=llama3 
</pre>

---

## 🏃 Running the Project
### 1️⃣ Start the FastAPI Backend
uvicorn src.api.s09a_fastapi:app --reload --host 0.0.0.0 --port 8000

### 2️⃣ Launch the Gradio UI
python src/api/s10a_gradio_ui.py
Then visit http://localhost:7860 in your browser.

---

## 📊 Evaluation
Run evaluation scripts to assess model performance:

RAG Evaluation:
python src/evaluators/s08a_evaluator.py

GraphRAG Evaluation:
python src/evaluators/s08b_nwkx_graph_evaluator.py

Results are saved in outputs/eval_results/ as timestamped JSON files.

---

## 🐳 Docker Deployment
Build the image:
docker build -t graphrag-assistant .

Run with Docker Compose:
docker-compose up --build

This will start both the FastAPI backend and Gradio UI.

---

## 🧪 Running Tests
pytest tests/ --maxfail=1 --disable-warnings -q

---

## 📜 License
This project is licensed under the MIT License. See LICENSE for details.
