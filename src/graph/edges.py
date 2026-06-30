from .state import RAGState


def route_after_critique(state: RAGState) -> str:
    if state["verdict"] == "grounded":
        return "finalize"
    if state["retry_count"] >= 2:
        return "finalize"
    return "rewrite_query"
