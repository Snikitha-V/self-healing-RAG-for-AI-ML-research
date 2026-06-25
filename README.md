<div align="center">

# 🔁 Self-Healing RAG for AI/ML Research

### *A self-healing RAG pipeline over **10,000+ ArXiv papers** with a critic agent that detects hallucinated citations and auto-reformulates queries*

<br>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cyclical-FF6F00?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.1:8b-000000?logo=ollama&logoColor=white)](https://ollama.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-7B1FA2?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![ArXiv](https://img.shields.io/badge/ArXiv-10K_Papers-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> **✨ "What if your RAG system could check its own homework — and fix its mistakes before you ever saw them?"**

**Self-Healing RAG** is a retrieval-augmented generation pipeline with a built-in **critic-retry loop**,
indexed over **10,000+ ArXiv AI/ML papers (2023–2024)** on LLMs, RAG, and hallucination.
It retrieves papers, generates an answer, then a *critic agent* evaluates that answer for hallucinations.
If grounded — it's returned. If not — the system reformulates the query and retries, up to 2 times,
before gracefully falling back.

</div>

<br>

## 🧠 The Meta Angle

```
┌──────────────────────────────────────────────────────────┐
│  This pipeline reads papers about hallucination          │
│  while checking its own answers for hallucinations.      │
│                                                          │
│  If it cites a non-existent method or confuses authors,  │
│  its built-in critic catches it and retries.             │
│                                                          │
│  The system that studies RAG failure modes               │
│  is itself resilient to RAG failure modes.               │
└──────────────────────────────────────────────────────────┘
```

Testing a hallucination detector on papers **about** hallucination is not just a demo —
it's a genuinely hard evaluation. Answers contain verifiable facts (authors, methods,
benchmark scores) that the critic must check against the source abstracts.

<br>

## 📚 Dataset — ArXiv AI/ML Papers

| Property | Value |
|---|---|
| **Source** | [ArXiv API](https://arxiv.org) — live research data, no auth needed |
| **Categories** | `cs.AI` (Artificial Intelligence), `cs.CL` (Computation & Language) |
| **Topics** | LLMs, RAG, hallucination, large language models, retrieval augmented generation |
| **Date range** | January 2023 – December 2024 |
| **Scale** | 10,000+ papers (configurable via `--max`) |
| **Storage** | 1 document = 1 abstract, with full metadata (title, authors, categories, arxiv_id, pdf_url) |

Each paper abstract is a dense, fact-packed paragraph. When the pipeline answers a question
about one of these papers, the critic can verify every claim: *"Did paper X really report 92% accuracy?
Was author Y actually a co-author?"*

<br>

## 🧬 Architecture

```
                    ┌───────────┐
                    │  RETRIEVE │  ← cosine search over 10K ArXiv paper embeddings
                    └─────┬─────┘
                          │
                          ▼
                    ┌───────────┐
                    │ GENERATE  │  ← llama3.1:8b with paper abstracts as context
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

The critic never sleeps. Every answer is checked before delivery.

<br>

## 🛠️ Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Native support for cyclical workflows with explicit state management |
| **LLM** | [llama3.1:8b](https://ollama.ai) via [Ollama](https://ollama.ai) | Fully open-source, runs locally, no API costs |
| **Embeddings** | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via [Sentence-Transformers](https://sbert.net) | 384-dim, fast, effective for semantic search |
| **Vector Store** | [Qdrant](https://qdrant.tech) (local persistent mode) | RocksDB-backed, no Docker needed, cosine similarity |
| **Data Source** | [ArXiv API](https://info.arxiv.org/help/api/index.html) | 10,000+ live AI/ML papers, no authentication required |
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
├── README.md
├── src/
│   ├── graph/
│   │   ├── state.py        ← RAGState TypedDict
│   │   ├── nodes.py        ← retrieve, generate, critique, rewrite, finalize
│   │   ├── edges.py        ← Conditional routing logic
│   │   └── graph.py        ← LangGraph graph assembly
│   ├── retriever/
│   │   ├── embedder.py     ← Sentence-Transformers embedding singleton
│   │   └── vectorstore.py  ← Qdrant client, search & ingest
│   ├── critic/
│   │   └── critic.py       ← Grounding checker module (swappable)
│   ├── ingest/
│   │   └── ingest.py       ← ArXiv paper fetching & document chunking
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
git clone https://github.com/Snikitha-V/self-healing-RAG-for-AI-ML-research.git
cd self-healing-RAG-for-AI-ML-research

pip install -r requirements.txt
cp .env.example .env    # defaults are fine for local setup
```

### Ingest 10,000+ ArXiv Papers

```bash
python -m src.ingest.ingest --arxiv
```

This fetches papers from ArXiv by the query:
> `(cat:cs.AI OR cat:cs.CL) AND (LLM OR "retrieval augmented generation" OR hallucination OR "large language model" OR RAG)`

Each paper's abstract is embedded with `all-MiniLM-L6-v2` and indexed in Qdrant (local persistence).

For a smaller test batch:
```bash
python -m src.ingest.ingest --arxiv --max 500
```

### Launch the API

```bash
uvicorn src.api.main:app --reload
```

Interactive docs at **http://localhost:8000/docs** 🎯

### Ask a Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What methods does the RAG paper by Lewis et al. propose?"}'
```

**Try tricking it:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who authored the 2024 paper on self-healing RAG systems?"}'
```
This paper doesn't exist → the critic detects hallucination → rewrite → fallback message.

### Run Tests

```bash
pytest tests/ -v
```

<br>

## 🔄 The Self-Healing Loop Explained

| Step | What Happens |
|---|---|
| **1. Retrieve** | Embed the query → cosine search across 10K ArXiv paper embeddings in Qdrant → return top 4 abstracts |
| **2. Generate** | Prompt `llama3.1:8b` with the paper abstracts as context → produce an answer |
| **3. Critique** | Prompt the same LLM (in JSON mode) to check: *"Is every claim in this answer supported by the papers? Any invented authors, methods, or benchmarks?"* |
| **4a. Grounded ✅** | Answer is safe → pass to `final_answer` → done |
| **4b. Hallucinated 🔄** | Rewrite the query using critic feedback → re-retrieve → re-generate → re-critique (up to 2 cycles) |
| **4c. Exhausted / No Docs ❌** | Return graceful fallback: *"I couldn't find sufficient information..."* |

> **Why "self-healing"?** Because the pipeline autonomously detects hallucinations, reformulates its search, and retries — all without human intervention.

<br>

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3.1:8b` | Model for generation & critique |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model |
| `QDRANT_PATH` | `./qdrant_db` | Qdrant persistent storage path |
| `ARXIV_MAX_RESULTS` | `10000` | Max ArXiv papers to fetch |

<br>

## 🧪 Testing

The routing logic is fully unit-tested without Ollama or Qdrant running:

| Test | Scenario |
|---|---|
| `test_routes_grounded_to_finalize` | Happy path — grounded answer returned immediately |
| `test_routes_insufficient_to_finalize` | Graceful degradation when docs don't contain info |
| `test_routes_hallucinated_with_retries_to_rewrite` | Recovery cycle triggers query rewrite |
| `test_routes_hallucinated_exhausted_to_finalize` | Max retries hit → safe fallback |
| `test_graph_builds` | Compilation smoke test |

<br>

## 📄 Prompts

### Generator
> *"You are a precise assistant. Answer the question using ONLY the provided context.
> If the context doesn't contain enough information, say 'I don't have enough information.'"*

### Critic
> *"You are a hallucination detector. Given a question, an answer, and source documents,
> determine if the answer is fully grounded in the documents.
> Respond ONLY with valid JSON."*

### Query Rewriter
> *"Rewrite the query to be more specific and likely to find relevant information.
> Return only the rewritten query, nothing else."*

<br>

## 🗺️ Roadmap

- [ ] **PDF ingestion** — full paper content beyond abstracts
- [ ] **Reranker node** — cross-encoder reranking between retrieve & generate
- [ ] **Streaming** — token-by-token via SSE
- [ ] **Multi-LLM backend** — pluggable Ollama / vLLM / cloud APIs
- [ ] **Benchmark suite** — evaluate on KILT / QAMPARI

<br>

---

<div align="center">

> **"The system that studies RAG failure modes is itself resilient to RAG failure modes."**

<br>

Made with ❤️ for robust, trustworthy AI research

</div>
