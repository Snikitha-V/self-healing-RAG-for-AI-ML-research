# Self-Healing RAG — Project Context

## What This Is
A Retrieval-Augmented Generation pipeline with a critic-retry loop built using LangGraph,
indexed over **10,000+ ArXiv AI/ML papers (2023–2024)** on LLMs, RAG, and hallucination.
The system retrieves papers → generates an answer → a critic agent evaluates hallucination risk →
retries with a reformulated query if rejected.

## Stack
- **Orchestration**: LangGraph (stateful, cyclical graph)
- **LLM**: llama3.1:8b via Ollama (open-source)
- **Embeddings**: all-MiniLM-L6-v2 via sentence-transformers (open-source)
- **Vector Store**: Qdrant (local persistent mode)
- **Framework**: Python 3.11+
- **API Layer**: FastAPI
- **Config**: python-dotenv

## Project Structure
self-healing-rag/
├── AGENTS.md               ← you are here
├── ARCHITECTURE.md         ← system design decisions
├── PROMPTS.md              ← all LLM prompts live here
├── .env.example
├── requirements.txt
├── src/
│   ├── graph/
│   │   ├── state.py        ← RAGState TypedDict
│   │   ├── nodes.py        ← retrieve, generate, critique, rewrite nodes
│   │   ├── edges.py        ← conditional routing logic
│   │   └── graph.py        ← LangGraph graph assembly
│   ├── retriever/
│   │   ├── vectorstore.py  ← Qdrant setup & querying
│   │   └── embedder.py     ← embedding logic (sentence-transformers)
│   ├── critic/
│   │   └── critic.py       ← grounding checker module doc
│   ├── ingest/
│   │   └── ingest.py       ← document chunking & indexing
│   └── api/
│       └── main.py         ← FastAPI endpoints
└── tests/

## Key Conventions
- All LangGraph node functions are pure: (state: RAGState) -> dict
- Nodes return ONLY the keys they modify — never the full state
- Max retry loop = 2 (controlled by state["retry_count"])
- Critic returns: { "verdict": "grounded" | "hallucinated" | "insufficient" }
- Never import from graph.py inside nodes.py (avoid circular imports)

## Environment Variables Required
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBED_MODEL_NAME=all-MiniLM-L6-v2
QDRANT_PATH=./qdrant_db

## Prerequisites
1. Install Ollama from https://ollama.ai
2. Run: ollama pull llama3.1:8b

## Commands
pip install -r requirements.txt
python -m src.ingest.ingest --arxiv  # index 10k ArXiv papers
uvicorn src.api.main:app --reload    # start API
pytest tests/                        # run tests