from typing import TypedDict, List


class Document(TypedDict):
    page_content: str
    metadata: dict


class RAGState(TypedDict):
    question: str
    query: str
    documents: List[Document]
    answer: str
    verdict: str
    verdict_reason: str
    retry_count: int
    final_answer: str
