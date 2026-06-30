import argparse
import os
import time

import arxiv
from tqdm import tqdm

from ..retriever.embedder import get_embedder
from ..retriever.vectorstore import add_documents, ensure_collection

ARXIV_QUERY = (
    "(cat:cs.AI OR cat:cs.CL) AND "
    "(LLM OR \"retrieval augmented generation\" OR hallucination "
    "OR \"large language model\" OR RAG) AND "
    "submittedDate:[202301010000 TO 202412312359]"
)


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


def fetch_arxiv_papers(max_results: int = 10000) -> list[dict]:
    print(f"Fetching up to {max_results} ArXiv papers (AI/ML, 2023-2024)...")
    print(f"Query: {ARXIV_QUERY}")

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)

    search = arxiv.Search(
        query=ARXIV_QUERY,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in tqdm(client.results(search), desc="Fetching papers"):
        papers.append({
            "page_content": result.summary,
            "metadata": {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "published": str(result.published.date()),
                "arxiv_id": result.entry_id.split("/")[-1],
                "pdf_url": str(result.pdf_url),
            },
        })

    print(f"Fetched {len(papers)} papers")
    return papers


def ingest_arxiv(max_results: int = 10000):
    papers = fetch_arxiv_papers(max_results)

    if not papers:
        print("No papers fetched — check query or network")
        return

    print("Indexing papers into Qdrant...")
    add_documents(papers)
    print(f"Ingested {len(papers)} ArXiv papers into vector store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("files", nargs="*", help="Text files to ingest")
    parser.add_argument(
        "--arxiv",
        action="store_true",
        help="Fetch and index ArXiv AI/ML papers (2023-2024)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=10000,
        help="Max ArXiv papers to fetch (default: 10000)",
    )
    args = parser.parse_args()

    embedder = get_embedder()
    dim = embedder.get_embedding_dimension()
    ensure_collection(dim)

    if args.arxiv:
        ingest_arxiv(max_results=args.max)
    elif args.files:
        for f in args.files:
            if os.path.isfile(f):
                ingest_file(f)
            else:
                print(f"Skipping {f}: not a file")
    else:
        parser.print_help()
