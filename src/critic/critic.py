"""
Hallucination detection module.

This is a **volatile** component — the criticism strategy (prompts, LLM,
parsing, or even the entire approach) is likely to change. All critic
logic lives here, behind the stable check_grounding() interface so that
graph nodes never need to change when the critic evolves.
"""

import json
import os

import ollama

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

CRITIC_SYSTEM_PROMPT = (
    "You are a hallucination detector. Given a question, an answer, and "
    "source documents, determine if the answer is fully grounded in the "
    "documents.\nRespond ONLY with valid JSON."
)

CRITIC_USER_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Documents: {documents}\n\n"
    'Respond with JSON: {{"verdict": "grounded" | "hallucinated" | '
    '"insufficient", "reason": "..."}}'
)


def _format_documents(documents: list[dict]) -> str:
    return "\n\n".join(
        f"[Doc {i+1}] {d['page_content']}"
        for i, d in enumerate(documents)
    )


def check_grounding(question: str, answer: str, documents: list[dict]) -> dict:
    """
    Evaluate whether *answer* is grounded in the provided *documents*.

    Returns
    -------
    dict with keys:
        "verdict"       — "grounded" | "hallucinated" | "insufficient"
        "verdict_reason" — human-readable explanation
    """
    if not documents:
        return {"verdict": "insufficient", "verdict_reason": "No documents retrieved"}

    docs_text = _format_documents(documents)
    user_prompt = CRITIC_USER_TEMPLATE.format(
        question=question, answer=answer, documents=docs_text
    )

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0},
        )
        content = response["message"]["content"]
    except Exception as e:
        return {"verdict": "insufficient", "verdict_reason": f"Critic LLM error: {e}"}

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"verdict": "insufficient", "reason": "Failed to parse critic JSON"}

    return {
        "verdict": result.get("verdict", "insufficient"),
        "verdict_reason": result.get("reason", ""),
    }
