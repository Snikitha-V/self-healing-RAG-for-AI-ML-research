import asyncio
import functools

from fastapi import FastAPI
from pydantic import BaseModel

from ..graph.graph import build_graph

app = FastAPI(title="Self-Healing RAG")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    retry_count: int


_graph_app = None


def get_graph():
    global _graph_app
    if _graph_app is None:
        _graph_app = build_graph()
    return _graph_app


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    graph = get_graph()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(
            graph.invoke,
            {
                "question": request.question,
                "query": request.question,
                "documents": [],
                "answer": "",
                "verdict": "",
                "verdict_reason": "",
                "retry_count": 0,
                "final_answer": "",
            },
        ),
    )
    return QueryResponse(answer=result["final_answer"], retry_count=result["retry_count"])


@app.get("/health")
async def health():
    return {"status": "ok"}
