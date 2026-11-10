"""Ingest all PDFs listed in the manifest into a JSONL file of chunks.

Usage (from the project root, venv active):
    python scripts/ingest.py                    # settings from .env
    python scripts/ingest.py --chunk-size 510 --overlap 64  # override for an experiment
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

# Make src/ importable when running this file directly. Once the project is
# an installable package (M9, pyproject.toml), this line can go.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from config import get_settings  # noqa: E402
from ingest.chunker import chunk_pages  # noqa: E402
from ingest.cleaner import clean_pages  # noqa: E402
from ingest.loader import file_sha256, load_manifest, load_pdf  # noqa: E402
from ingest.models import Chunk  # noqa: E402

logger = logging.getLogger("ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into chunks.")
    parser.add_argument("--chunk-size", type=int, help="override CHUNK_SIZE")
    parser.add_argument("--overlap", type=int, help="override CHUNK_OVERLAP")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # The Hugging Face client logs every HTTP request at INFO; that's noise here.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = args.overlap if args.overlap is not None else settings.chunk_overlap

    manifest = load_manifest(settings.manifest_path)

    # Warn about PDFs that are on disk but not in the manifest (easy to forget).
    on_disk = {p.name for p in settings.raw_dir.glob("*.pdf")}
    for name in sorted(on_disk - manifest.keys()):
        logger.warning("Skipping %s: not listed in %s", name, settings.manifest_path.name)

    # Load the tokenizer once: loading it per document would be slow.
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

    start = time.perf_counter()
    all_chunks: list[Chunk] = []
    for filename, row in manifest.items():
        path = settings.raw_dir / filename
        try:
            pages = load_pdf(path, row["doc_id"], row["title"], row["category"])
        except (FileNotFoundError, ValueError) as e:
            # One broken file shouldn't stop the whole run.
            logger.error("Skipping %s: %s", filename, e)
            continue

        pages = clean_pages(pages, has_line_numbers=row.get("strip_line_numbers") == "true")
        chunks = chunk_pages(pages, tokenizer, chunk_size, overlap, file_sha256(path))
        logger.info("%s: %d pages -> %d chunks", row["doc_id"], len(pages), len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks produced. Check %s and the manifest.", settings.raw_dir)
        sys.exit(1)

    # One file per chunk size, so M4 can compare 256 vs 512 side by side.
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / f"chunks_{chunk_size}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            # ensure_ascii=False keeps "§" and "–" readable in the file.
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    sizes = [c.n_tokens for c in all_chunks]
    logger.info(
        "Wrote %d chunks to %s in %.1fs (tokens: mean %.0f, median %.0f, max %d)",
        len(all_chunks),
        out_path,
        time.perf_counter() - start,
        statistics.mean(sizes),
        statistics.median(sizes),
        max(sizes),
    )


if __name__ == "__main__":
    main()
