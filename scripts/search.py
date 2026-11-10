"""Semantic search from the command line, to try out retrieval by hand.

Usage (from the project root, venv active, after scripts/index.py):
    python scripts/search.py "What should an SBOM contain?"
    python scripts/search.py "software classification rule 11" --category eu_guidance eu_regulation
    python scripts/search.py "cybersecurity testing" --doc-id fda_cybersecurity_2025 --k 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings  # noqa: E402
from retrieval.embedder import Embedder  # noqa: E402
from retrieval.vector_store import VectorStore, make_filter  # noqa: E402


def one_or_many(values: list[str] | None) -> str | list[str] | None:
    """argparse gives a list; make_filter wants a single value or a list."""
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, help="number of results (default TOP_K)")
    parser.add_argument("--category", nargs="+", help="only these categories")
    parser.add_argument("--doc-id", nargs="+", help="only these documents")
    parser.add_argument("--chunk-size", type=int, help="which collection to search")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=logging.WARNING)
    chunk_size = args.chunk_size or settings.chunk_size

    store = VectorStore(settings.chroma_dir, f"chunks_{chunk_size}", settings.embedding_model)
    if store.count() == 0:
        print(f"Collection chunks_{chunk_size} is empty. Run: python scripts/index.py")
        sys.exit(1)

    # Model loading is a one-off startup cost (an API loads it once), so it is
    # timed separately from the per-question retrieval latency.
    t0 = time.perf_counter()
    embedder = Embedder(
        settings.embedding_model,
        device=settings.embedding_device,
        query_prefix=settings.query_prefix,
    )
    t1 = time.perf_counter()
    query_vector = embedder.embed_query(args.query)
    t2 = time.perf_counter()
    where = make_filter(category=one_or_many(args.category), doc_id=one_or_many(args.doc_id))
    results = store.search(query_vector, k=args.k or settings.top_k, where=where)
    t3 = time.perf_counter()

    print(f'\nQuery: "{args.query}"   filter: {where or "none"}   collection: chunks_{chunk_size}\n')
    if not results:
        print("No results (check the filter values).")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. score {r.score:.3f} | {r.title[:70]} | p.{r.page} | {r.chunk_id}")
        print(textwrap.indent(textwrap.fill(r.text[:400] + ("..." if len(r.text) > 400 else ""), 100), "   "))
        print()
    print(
        f"Timing: model load {t1 - t0:.2f}s (one-off) | embed query {(t2 - t1) * 1000:.0f} ms | "
        f"vector search {(t3 - t2) * 1000:.0f} ms | retrieval total {(t3 - t1) * 1000:.0f} ms"
    )


if __name__ == "__main__":
    main()
