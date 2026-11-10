"""Store chunk vectors in ChromaDB and search them by cosine similarity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ingest.models import Chunk

logger = logging.getLogger(__name__)

# Chroma limits how many items one call may add; stay well below it.
_ADD_BATCH_SIZE = 1000


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    doc_id: str
    title: str
    category: str
    page: int
    score: float  # cosine similarity: higher = more similar (max 1.0)


def make_filter(
    category: str | list[str] | None = None, doc_id: str | list[str] | None = None
) -> dict[str, Any] | None:
    """Build a Chroma `where` filter, e.g. "only the EU MDR".

    Chroma syntax: {"category": "eu_regulation"} for one value,
    {"category": {"$in": [...]}} for several, and {"$and": [...]} to combine
    fields (Chroma rejects a dict with two fields without $and).
    """
    conditions = []
    for field, value in (("category", category), ("doc_id", doc_id)):
        if value is None:
            continue
        if isinstance(value, list):
            conditions.append({field: {"$in": value}})
        else:
            conditions.append({field: value})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class VectorStore:
    """One Chroma collection holding the chunks of one chunk size."""

    def __init__(
        self, path: Path, collection_name: str, embedding_model: str, reset: bool = False
    ) -> None:
        # PersistentClient saves to disk (data/chroma/): no server needed,
        # and the index survives restarts. Telemetry off: we run offline.
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        if reset and collection_name in [c.name for c in self.client.list_collections()]:
            logger.info("Deleting collection %s for a full rebuild", collection_name)
            self.client.delete_collection(collection_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Cosine distance for the HNSW index (Chroma's default is L2).
            configuration={"hnsw": {"space": "cosine"}},
            # Remember which model made the vectors (see the check below).
            metadata={"embedding_model": embedding_model},
            # We always pass our own vectors. Without this, Chroma attaches a
            # default model and would silently embed with it if we forgot.
            embedding_function=None,
        )

        # Vectors from different models can't be compared. Fail loudly
        # instead of returning nonsense results.
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise ValueError(
                f"Collection {collection_name!r} was built with {stored_model!r}, "
                f"not {embedding_model!r}. Rebuild it (scripts/index.py --rebuild)."
            )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert or overwrite chunks. upsert + deterministic ids = no duplicates."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start : start + _ADD_BATCH_SIZE]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _ADD_BATCH_SIZE].tolist(),
                documents=[c.text for c in batch],
                # Metadata is what filters and citations use later.
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "category": c.category,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "n_tokens": c.n_tokens,
                        "source_hash": c.source_hash,
                    }
                    for c in batch
                ],
            )

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of one document (before re-adding a changed file)."""
        self.collection.delete(where={"doc_id": doc_id})

    def document_hashes(self) -> dict[str, str]:
        """Return {doc_id: source_hash} for the documents already indexed."""
        records = self.collection.get(include=["metadatas"])
        return {m["doc_id"]: m["source_hash"] for m in records["metadatas"] or []}

    def search(
        self, query_embedding: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Return the k chunks closest to the query vector, best first."""
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns one list per query; we sent one query, so take [0].
        return [
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                doc_id=meta["doc_id"],
                title=meta["title"],
                category=meta["category"],
                page=int(meta["page"]),
                # Chroma gives cosine DISTANCE (1 - similarity, lower = closer).
                score=1.0 - distance,
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
