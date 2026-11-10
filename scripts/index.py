"""Embed the chunks and store them in ChromaDB (re-embeds only changed documents).

Usage (from the project root, venv active, after scripts/ingest.py):
    python scripts/index.py                    # chunk size from .env (256)
    python scripts/index.py --chunk-size 510   # index the large-chunk variant
    python scripts/index.py --rebuild          # wipe and rebuild (after changing
                                               # the cleaner, chunker or model)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings  # noqa: E402
from retrieval.embedder import Embedder  # noqa: E402
from retrieval.indexer import load_chunks, sync_index  # noqa: E402
from retrieval.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger("index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB.")
    parser.add_argument("--chunk-size", type=int, help="which chunks_<size>.jsonl to index")
    parser.add_argument("--rebuild", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    chunk_size = args.chunk_size or settings.chunk_size
    chunks_path = settings.processed_dir / f"chunks_{chunk_size}.jsonl"
    if not chunks_path.exists():
        logger.error("%s not found. Run: python scripts/ingest.py --chunk-size %d", chunks_path, chunk_size)
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    # One collection per chunk size, so M4 can compare them side by side.
    store = VectorStore(
        settings.chroma_dir, f"chunks_{chunk_size}", settings.embedding_model, reset=args.rebuild
    )
    embedder = Embedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        query_prefix=settings.query_prefix,
    )

    start = time.perf_counter()
    report = sync_index(chunks, store, embedder)
    elapsed = time.perf_counter() - start

    logger.info(
        "Done in %.1fs: embedded %d chunks | added/updated %s | unchanged %s | removed %s | "
        "collection now holds %d chunks",
        elapsed,
        report.chunks_embedded,
        report.added or "-",
        report.skipped or "-",
        report.removed or "-",
        store.count(),
    )


if __name__ == "__main__":
    main()
