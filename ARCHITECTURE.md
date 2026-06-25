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
    retry_count: int
    final_answer: str
```

## Retry Logic
- Max retries: 2
- On hallucination → rewrite query and re-retrieve
- On insufficient docs → return graceful fallback string
- On grounded → pass answer to final_answer and END

## Critic Design
The critic receives (question, answer, documents) and checks:
1. Is every factual claim in the answer supported by a document?
2. Are there any invented entities, dates, or quotes?
Returns a JSON verdict — never prose.

## Why LangGraph over LangChain Chains?
- Chains are linear; this workflow is cyclical
- LangGraph gives explicit state management and conditional edges
- Easier to add new nodes (e.g. a "reranker" node) without refactoring