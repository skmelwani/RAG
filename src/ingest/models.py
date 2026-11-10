"""Data types shared by the ingestion pipeline: a PDF page and a text chunk."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Page:
    """One page of a PDF document.

    frozen=True makes instances read-only. Each pipeline step returns new
    objects instead of changing old ones, so a bug in one step can't silently
    corrupt data that another step is still using.
    """

    doc_id: str  # short stable id, e.g. "fda_cybersecurity_2023"
    title: str  # human-readable title, shown in citations
    category: str  # e.g. "fda_guidance", "eu_mdr" (used for filters later)
    page: int  # 1-based, so it matches the page number in a PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of text that will be embedded and retrieved as one unit."""

    chunk_id: str  # deterministic: "{doc_id}-p{page}-c{chunk_index}"
    doc_id: str
    title: str
    category: str
    page: int
    chunk_index: int  # position of this chunk within its page
    text: str
    n_tokens: int  # measured with the embedding model's tokenizer
    source_hash: str  # SHA-256 of the PDF, to detect changed files on re-ingest

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict, ready for json.dumps (one JSONL line)."""
        return asdict(self)
