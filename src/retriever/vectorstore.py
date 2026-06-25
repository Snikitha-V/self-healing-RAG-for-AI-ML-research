import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embedder import embed_texts, get_embedder

COLLECTION_NAME = "documents"

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        path = os.getenv("QDRANT_PATH", "./qdrant_db")
        _client = QdrantClient(path=path)
    return _client


def ensure_collection(dim: int = 384):
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def add_documents(docs: list[dict]):
    client = get_client()
    embedder = get_embedder()
    dim = embedder.get_sentence_embedding_dimension()
    ensure_collection(dim)

    texts = [d["page_content"] for d in docs]
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=idx,
            vector=vec,
            payload={"text": d["page_content"], "metadata": d.get("metadata", {})},
        )
        for idx, (d, vec) in enumerate(zip(docs, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def similarity_search(query: str, k: int = 4) -> list[dict]:
    client = get_client()
    embedder = get_embedder()
    dim = embedder.get_sentence_embedding_dimension()
    ensure_collection(dim)

    query_vector = embed_texts([query])[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k,
    )
    return [
        {"page_content": r.payload["text"], "metadata": r.payload.get("metadata", {})}
        for r in results
    ]
