import hashlib
import os
import uuid

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


def close_client():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def ensure_collection(dim: int = 384):
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _make_id(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text.encode("utf-8")))


def _existing_ids(texts: list[str]) -> set[str]:
    client = get_client()
    ids = [_make_id(t) for t in texts]
    existing = set()
    for id_ in ids:
        try:
            result = client.retrieve(
                collection_name=COLLECTION_NAME, ids=[id_]
            )
            if result:
                existing.add(id_)
        except Exception:
            pass
    return existing


def add_documents(docs: list[dict]):
    client = get_client()
    embedder = get_embedder()
    dim = embedder.get_embedding_dimension()
    ensure_collection(dim)

    texts = [d["page_content"] for d in docs]
    vectors = embed_texts(texts)

    existing = _existing_ids(texts)

    points = [
        PointStruct(
            id=_make_id(d["page_content"]),
            vector=vec,
            payload={"text": d["page_content"], "metadata": d.get("metadata", {})},
        )
        for d, vec in zip(docs, vectors)
        if _make_id(d["page_content"]) not in existing
    ]

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Added {len(points)} new documents")
    else:
        print("No new documents to add (all already exist)")


def similarity_search(query: str, k: int = 4) -> list[dict]:
    client = get_client()
    embedder = get_embedder()
    dim = embedder.get_embedding_dimension()
    ensure_collection(dim)

    query_vector = embed_texts([query])[0]
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    return [
        {"page_content": r.payload["text"], "metadata": r.payload.get("metadata", {})}
        for r in response.points
    ]
