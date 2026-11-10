"""Keep the vector store in sync with the chunks file, re-embedding only what changed."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ingest.models import Chunk

from .embedder import DocumentEmbedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """What a sync did, so the script can print it and tests can check it."""

    added: list[str] = field(default_factory=list)  # new or changed documents
    skipped: list[str] = field(default_factory=list)  # unchanged documents
    removed: list[str] = field(default_factory=list)  # no longer in the chunks file
    chunks_embedded: int = 0


def load_chunks(path: Path) -> list[Chunk]:
    """Read the JSONL file written by scripts/ingest.py."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def sync_index(chunks: list[Chunk], store: VectorStore, embedder: DocumentEmbedder) -> IndexReport:
    """Make the store match `chunks`, embedding only new or changed documents.

    A document counts as changed when its PDF hash differs from the stored one.
    Limitation: the hash covers the PDF only. If you change the cleaner or the
    chunker, run a full rebuild, because unchanged PDFs would be skipped.
    """
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    indexed = store.document_hashes()
    report = IndexReport()

    for doc_id, doc_chunks in by_doc.items():
        new_hash = doc_chunks[0].source_hash
        if indexed.get(doc_id) == new_hash:
            report.skipped.append(doc_id)
            continue
        if doc_id in indexed:
            # Changed file: delete ALL old chunks first. Upsert alone would
            # leave stale chunks behind if the new version has fewer chunks.
            store.delete_document(doc_id)
        embeddings = embedder.embed_documents([c.text for c in doc_chunks])
        store.add_chunks(doc_chunks, embeddings)
        report.added.append(doc_id)
        report.chunks_embedded += len(doc_chunks)
        logger.info("Indexed %s: %d chunks", doc_id, len(doc_chunks))

    # Documents removed from the manifest shouldn't keep showing up in answers.
    for doc_id in indexed.keys() - by_doc.keys():
        store.delete_document(doc_id)
        report.removed.append(doc_id)
        logger.info("Removed %s from the index", doc_id)

    return report
