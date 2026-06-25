import argparse
import os

from ..retriever.embedder import get_embedder
from ..retriever.vectorstore import add_documents, ensure_collection


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    docs = [
        {"page_content": chunk, "metadata": {"source": filepath, "chunk": i}}
        for i, chunk in enumerate(chunks)
    ]
    add_documents(docs)
    print(f"Ingested {len(docs)} chunks from {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("files", nargs="+", help="Text files to ingest")
    args = parser.parse_args()

    embedder = get_embedder()
    dim = embedder.get_sentence_embedding_dimension()
    ensure_collection(dim)

    for f in args.files:
        if os.path.isfile(f):
            ingest_file(f)
        else:
            print(f"Skipping {f}: not a file")
