from langgraph.graph import END, StateGraph

from .edges import route_after_critique
from .nodes import critique, finalize, generate, retrieve, rewrite_query
from .state import RAGState


def build_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("critique", critique)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "grounded": "finalize",
            "rewrite_query": "rewrite_query",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("finalize", END)

    return workflow.compile()
