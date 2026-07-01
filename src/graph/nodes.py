import os

import ollama

from ..critic.critic import check_grounding
from ..retriever.vectorstore import similarity_search
from .state import RAGState

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")


def retrieve(state: RAGState) -> dict:
    query = state["query"] or state["question"]
    docs = similarity_search(query, k=4)
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    if not state.get("documents"):
        return {"answer": "No relevant documents were found to answer the question. Try rephrasing."}

    docs_text = "\n\n".join(
        f"[Doc {i+1}] {d['page_content']}"
        for i, d in enumerate(state["documents"])
    )
    system_prompt = (
        "You are a precise assistant. Answer the question using ONLY the "
        "provided context. If the context doesn't contain enough information, "
        "say 'I don't have enough information.'"
    )
    user_prompt = f"Context:\n{docs_text}\n\nQuestion: {state['question']}"

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0},
        )
        return {"answer": response["message"]["content"]}
    except Exception as e:
        return {"answer": f"Generation failed due to LLM error: {e}"}


def critique(state: RAGState) -> dict:
    return check_grounding(
        state["question"],
        state["answer"],
        state.get("documents", []),
    )


def rewrite_query(state: RAGState) -> dict:
    system_prompt = "You are a search query optimizer."
    user_prompt = (
        "The following query failed to retrieve useful documents:\n"
        f"Original query: {state['query'] or state['question']}\n"
        f"Critic feedback: {state['verdict_reason']}\n\n"
        "Rewrite the query to be more specific and likely to find relevant "
        "information.\nReturn only the rewritten query, nothing else."
    )

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0},
        )
        new_query = response["message"]["content"].strip()
    except Exception as e:
        new_query = state.get("query") or state["question"]

    return {"query": new_query, "retry_count": state["retry_count"] + 1}


def finalize(state: RAGState) -> dict:
    if state["verdict"] == "grounded":
        return {"final_answer": state["answer"]}
    return {
        "final_answer": (
            "I couldn't find sufficient information to answer your question. "
            "Please try rephrasing or providing more details."
        )
    }
