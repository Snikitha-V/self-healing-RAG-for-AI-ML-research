<div align="center">

# 🔁 Self-Healing RAG for AI/ML Research

### *An intelligent RAG pipeline that detects its own failures and recovers — autonomously*

<br>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cyclical-FF6F00?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.1:8b-000000?logo=ollama&logoColor=white)](https://ollama.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-7B1FA2?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> **✨ What if your RAG system could check its own homework — and fix its mistakes before you ever saw them?**

**Self-Healing RAG** is exactly that: a retrieval-augmented generation pipeline with a built-in **critic-retry loop**. It retrieves documents, generates an answer, then has a *critic agent* evaluate that answer for hallucinations. If the answer is grounded in the sources, it's returned. If not — the system reformulates the query and retries, up to 2 times, before gracefully falling back.

</div>

<br>

## 🧠 Architecture

The pipeline is a **LangGraph cycle**, not a linear chain:

```
                    ┌───────────┐
                    │  RETRIEVE │
                    └─────┬─────┘
                          │
                          ▼
                    ┌───────────┐
                    │ GENERATE  │
                    └─────┬─────┘
                          │
                          ▼
                    ┌───────────┐     grounded     ┌──────────┐
                    │  CRITIQUE │ ──────────────►  │ FINALIZE │ ──► ✅ Answer
                    └─────┬─────┘                  └──────────┘
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     ▼
      ┌───────────────┐    ┌──────────────┐
      │  REWRITE QUERY │    │   FALLBACK   │
      │   (retry + 1)  │    │   (exhausted │
      └───────┬───────┘    │  or no docs)  │
              │            └──────────────┘
              │  ▲
              └──┘ (up to 2 cycles)
```

**The critic never sleeps.** Every answer is checked before delivery.

<br>

## 🛠️ Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Native support for cyclical workflows with explicit state management |
| **LLM** | [llama3.1:8b](https://ollama.ai) via [Ollama](https://ollama.ai) | Fully open-source, runs locally, no API costs, no data leaving your machine |
| **Embeddings** | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via [Sentence-Transformers](https://sbert.net) | Lightweight (384-dim), fast, surprisingly effective for semantic search |
| **Vector Store** | [Qdrant](https://qdrant.tech) (local persistent mode) | RocksDB-backed persistence, no Docker needed, cosine similarity out of the box |
| **API** | [FastAPI](https://fastapi.tiangolo.com) | Auto-docs, Pydantic validation, async-ready |

<br>

## 📦 Project Structure

```
self-healing-rag/
├── AGENTS.md               ← Project context & conventions
├── ARCHITECTURE.md         ← System design decisions
├── PROMPTS.md              ← All LLM prompts (generator, critic, rewriter)
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── graph/
│   │   ├── state.py        ← RAGState TypedDict
│   │   ├── nodes.py        ← retrieve, generate, critique, rewrite, finalize
│   │   ├── edges.py        ← Conditional routing logic
│   │   └── graph.py        ← LangGraph graph assembly
│   ├── retriever/
│   │   ├── embedder.py     ← Sentence-Transformers embedding singleton
│   │   └── vectorstore.py  ← Qdrant client, search, & ingest
│   ├── critic/
│   │   └── critic.py       ← Grounding checker module (swappable)
│   ├── ingest/
│   │   └── ingest.py       ← Document chunking & Qdrant indexing
│   └── api/
│       └── main.py         ← FastAPI endpoints
└── tests/
    ├── conftest.py         ← Pytest config
    └── test_graph.py       ← 5 tests: routing logic + graph compilation
```

<br>

## 🚀 Quick Start

### Prerequisites

1. **Install [Ollama](https://ollama.ai)**
2. **Pull the LLM:**
   ```bash
   ollama pull llama3.1:8b
   ```

### Setup

```bash
# Clone & enter
git clone https://github.com/Snikitha-V/self-healing-RAG-for-AI-ML-research.git
cd self-healing-RAG-for-AI-ML-research

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env    # defaults are fine for local setup
```

### Ingest Documents

```bash
python -m src.ingest.ingest path/to/your/document.txt
```

The chunker splits text into 500-char segments with 50-char overlap, embeds them with `all-MiniLM-L6-v2`, and indexes them in Qdrant.

### Launch the API

```bash
uvicorn src.api.main:app --reload
```

Then visit **http://localhost:8000/docs** for the interactive Swagger UI.

### Ask a Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}'
```

### Run Tests

```bash
pytest tests/ -v
```

<br>

## 🔄 The Self-Healing Loop Explained

| Step | What Happens |
|---|---|
| **1. Retrieve** | Embed the query → cosine search in Qdrant → return top 4 docs |
| **2. Generate** | Prompt `llama3.1:8b` with the docs as context → produce an answer |
| **3. Critique** | Prompt the same LLM (in JSON mode) to check: *"Is every claim in this answer supported by the documents?"* |
| **4a. Grounded ✅** | Answer is safe → pass to `final_answer` → done |
| **4b. Hallucinated 🔄** | Rewrite the query using critic feedback → re-retrieve → re-generate → re-critique (up to 2 cycles) |
| **4c. Exhausted / No Docs ❌** | Return a graceful fallback message |

> **Why "self-healing"?** Because the pipeline autonomously detects hallucinations, reformulates its search, and retries — all without human intervention.

<br>

## ⚙️ Configuration

All settings live in environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3.1:8b` | Model name for generation & critique |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `QDRANT_PATH` | `./qdrant_db` | Path for Qdrant's persistent storage |

<br>

## 🧪 Testing Philosophy

The routing logic is fully unit-tested without needing Ollama or Qdrant running:

- **`test_routes_grounded_to_finalize`** — happy path
- **`test_routes_insufficient_to_finalize`** — graceful degradation
- **`test_routes_hallucinated_with_retries_to_rewrite`** — recovery cycle
- **`test_routes_hallucinated_exhausted_to_finalize`** — max retries exhausted
- **`test_graph_builds`** — compilation smoke test

<br>

## 🗺️ Roadmap

- [ ] **PDF ingestion** — native support for research papers
- [ ] **Reranker node** — cross-encoder reranking between retrieve & generate
- [ ] **Streaming responses** — token-by-token streaming via SSE
- [ ] **Multiple LLM backends** — swap between Ollama, vLLM, or cloud APIs via config
- [ ] **Evaluation harness** — automated benchmarking on KILT / QAMPARI

<br>

---

<div align="center">

Made with ❤️ for robust, trustworthy AI research

</div>
