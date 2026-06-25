import os

from sentence_transformers import SentenceTransformer

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return embedder.encode(texts, convert_to_numpy=True).tolist()
