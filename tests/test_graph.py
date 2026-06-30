from src.graph.state import RAGState
from src.graph.edges import route_after_critique


def test_routes_grounded_to_finalize():
    state: RAGState = {
        "question": "test",
        "query": "test",
        "documents": [],
        "answer": "answer",
        "verdict": "grounded",
        "verdict_reason": "",
        "retry_count": 0,
        "final_answer": "",
    }
    assert route_after_critique(state) == "finalize"


def test_routes_insufficient_to_rewrite():
    state: RAGState = {
        "question": "test",
        "query": "test",
        "documents": [],
        "answer": "answer",
        "verdict": "insufficient",
        "verdict_reason": "",
        "retry_count": 0,
        "final_answer": "",
    }
    assert route_after_critique(state) == "rewrite_query"


def test_routes_hallucinated_with_retries_to_rewrite():
    state: RAGState = {
        "question": "test",
        "query": "test",
        "documents": [],
        "answer": "answer",
        "verdict": "hallucinated",
        "verdict_reason": "",
        "retry_count": 0,
        "final_answer": "",
    }
    assert route_after_critique(state) == "rewrite_query"


def test_routes_hallucinated_exhausted_to_finalize():
    state: RAGState = {
        "question": "test",
        "query": "test",
        "documents": [],
        "answer": "answer",
        "verdict": "hallucinated",
        "verdict_reason": "",
        "retry_count": 2,
        "final_answer": "",
    }
    assert route_after_critique(state) == "finalize"


def test_graph_builds():
    from src.graph.graph import build_graph

    graph = build_graph()
    assert graph is not None
