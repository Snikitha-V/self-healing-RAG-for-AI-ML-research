# Architecture Decisions

## Graph Shape
The LangGraph graph is a CYCLE, not a chain:

  [retrieve] → [generate] → [critique]
                    ↑              |
                    |    grounded  → END (return answer)
              [rewrite_query] ←   |
                                  | hallucinated/insufficient
                                  ↓
                             (retry or fallback)

## RAGState Schema
```python
class RAGState(TypedDict):
    question: str
    query: str                  # may be rewritten
    documents: list[Document]
    answer: str
    verdict: str                # grounded | hallucinated | insufficient
    verdict_reason: str
    retry_count: int
    final_answer: str
```

## Dataset — ArXiv AI/ML Papers (2023–2024)

| Property | Value |
|---|---|
| **Source** | ArXiv API via `arxiv` Python client |
| **Categories** | `cs.AI` (Artificial Intelligence), `cs.CL` (Computation & Language) |
| **Topics** | LLMs, retrieval augmented generation, hallucination, large language models, RAG |
| **Date range** | January 2023 – December 2024 |
| **Scale** | 10,000+ papers (configurable) |
| **Document format** | Each paper = 1 document with abstract as `page_content` |
| **Metadata** | title, authors, categories, published date, arxiv_id, pdf_url |

Each paper abstract is stored as a single document. This keeps chunks clean and
fact-dense — ideal for meaningful hallucination detection (wrong author, invented
benchmark, misattributed method).

## Retry Logic
- Max retries: 2
- On hallucination → rewrite query and re-retrieve
- On insufficient docs → return graceful fallback string
- On grounded → pass answer to final_answer and END

## Critic Design
The critic receives (question, answer, documents) and checks:
1. Is every factual claim in the answer supported by a document?
2. Are there any invented entities, authors, dates, or benchmarks?
Returns a JSON verdict — never prose.

## Why LangGraph over LangChain Chains?
- Chains are linear; this workflow is cyclical
- LangGraph gives explicit state management and conditional edges
- Easier to add new nodes (e.g. a "reranker" node) without refactoring
