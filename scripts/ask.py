"""Ask a question and get a grounded, cited answer from the local LLM.

Usage (from the project root, venv active, Ollama running, after scripts/index.py):
    python scripts/ask.py "What should an SBOM contain?"
    python scripts/ask.py "How is software classified?" --category eu_guidance eu_regulation
    python scripts/ask.py "What is a PCCP?" --k 3 --show-sources
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings  # noqa: E402
from generation.llm import LLMError, OllamaClient  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from retrieval.embedder import Embedder  # noqa: E402
from retrieval.vector_store import VectorStore, make_filter  # noqa: E402


def one_or_many(values: list[str] | None) -> str | list[str] | None:
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the indexed documents a question.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, help="number of chunks to retrieve (default TOP_K)")
    parser.add_argument("--category", nargs="+", help="only these categories")
    parser.add_argument("--doc-id", nargs="+", help="only these documents")
    parser.add_argument("--show-sources", action="store_true", help="print all retrieved chunks")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    store = VectorStore(
        settings.chroma_dir, f"chunks_{settings.chunk_size}", settings.embedding_model
    )
    if store.count() == 0:
        sys.exit(f"Collection chunks_{settings.chunk_size} is empty. Run: python scripts/index.py")

    embedder = Embedder(
        settings.embedding_model,
        device=settings.embedding_device,
        query_prefix=settings.query_prefix,
    )
    llm = OllamaClient(
        settings.ollama_url,
        settings.llm_model,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        timeout=settings.llm_timeout,
    )
    pipeline = RagPipeline(
        embedder, store, llm, top_k=settings.top_k, min_score=settings.min_score
    )

    where = make_filter(category=one_or_many(args.category), doc_id=one_or_many(args.doc_id))
    try:
        result = pipeline.answer(args.question, top_k=args.k, where=where)
    except LLMError as e:  # includes LLMUnavailableError
        sys.exit(f"LLM error: {e}")

    print(f"\nQ: {result.question}\n")
    print(textwrap.fill(result.answer, 100))
    if result.refused:
        print(f"\n(refused: {result.refusal_reason})")
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  {c.label()}")
    for w in result.warnings:
        print(f"\nWarning: {w}")

    if args.show_sources:
        print("\nRetrieved chunks:")
        for n, r in enumerate(result.sources, start=1):
            print(f"  [{n}] score {r.score:.3f} | {r.title[:60]} | p.{r.page}")

    t = result.timings_ms
    print(
        f"\nTiming: retrieval {t.get('retrieval', 0):.0f} ms | "
        f"generation {t.get('generation', 0):.0f} ms | total {t.get('total', 0):.0f} ms"
    )


if __name__ == "__main__":
    main()
