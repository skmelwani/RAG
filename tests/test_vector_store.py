"""Tests for the vector store and indexer, using a tiny fake embedder.

The fake embedder makes vectors from keyword counts, so tests are instant,
need no model download, and we know exactly which chunk should match.
"""

from __future__ import annotations

import numpy as np
import pytest

from ingest.models import Chunk
from retrieval.indexer import sync_index
from retrieval.vector_store import VectorStore, make_filter

VOCAB = ["cybersecurity", "software", "classification", "clinical"]


class FakeEmbedder:
    """Vector = counts of each VOCAB word, normalised to length 1."""

    def __init__(self) -> None:
        self.calls = 0  # lets tests check whether anything was (re-)embedded

    def _vec(self, text: str) -> np.ndarray:
        words = text.lower().split()
        v = np.array([words.count(w) for w in VOCAB], dtype=np.float32) + 0.01
        return v / np.linalg.norm(v)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_chunk(doc_id: str, idx: int, text: str, category: str = "fda_guidance", h: str = "h1") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-p1-c{idx}",
        doc_id=doc_id,
        title=f"Title {doc_id}",
        category=category,
        page=1,
        chunk_index=idx,
        text=text,
        n_tokens=len(text.split()),
        source_hash=h,
    )


@pytest.fixture
def store(tmp_path):
    # tmp_path: a fresh empty folder per test, deleted afterwards by pytest.
    return VectorStore(tmp_path / "chroma", "test_chunks", "fake-model")


@pytest.fixture
def chunks():
    return [
        make_chunk("fda", 0, "cybersecurity cybersecurity controls"),
        make_chunk("fda", 1, "software documentation software"),
        make_chunk("mdr", 0, "classification classification rules", category="eu_regulation"),
        make_chunk("mdr", 1, "clinical evaluation clinical", category="eu_regulation"),
    ]


def test_search_returns_most_similar_first(store, chunks):
    embedder = FakeEmbedder()
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    results = store.search(embedder.embed_query("cybersecurity"), k=3)
    assert results[0].chunk_id == "fda-p1-c0"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)


def test_search_result_carries_citation_metadata(store, chunks):
    embedder = FakeEmbedder()
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    top = store.search(embedder.embed_query("clinical"), k=1)[0]
    assert (top.doc_id, top.title, top.page, top.category) == ("mdr", "Title mdr", 1, "eu_regulation")
    assert top.text == "clinical evaluation clinical"


def test_category_filter(store, chunks):
    embedder = FakeEmbedder()
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    # "cybersecurity" best matches an FDA chunk, but the filter allows only the MDR.
    results = store.search(embedder.embed_query("cybersecurity"), k=4, where=make_filter("eu_regulation"))
    assert {r.category for r in results} == {"eu_regulation"}


def test_upsert_does_not_duplicate(store, chunks):
    embedder = FakeEmbedder()
    vectors = embedder.embed_documents([c.text for c in chunks])
    store.add_chunks(chunks, vectors)
    store.add_chunks(chunks, vectors)
    assert store.count() == len(chunks)


def test_search_on_empty_store_returns_nothing(store):
    assert store.search(FakeEmbedder().embed_query("software"), k=5) == []


def test_mismatched_lengths_raise(store, chunks):
    with pytest.raises(ValueError):
        store.add_chunks(chunks, FakeEmbedder().embed_documents(["only one"]))


def test_wrong_embedding_model_is_rejected(tmp_path):
    VectorStore(tmp_path / "chroma", "test_chunks", "model-a")
    with pytest.raises(ValueError, match="model-a"):
        VectorStore(tmp_path / "chroma", "test_chunks", "model-b")


def test_reset_empties_collection(tmp_path, chunks):
    first = VectorStore(tmp_path / "chroma", "test_chunks", "fake-model")
    first.add_chunks(chunks, FakeEmbedder().embed_documents([c.text for c in chunks]))
    rebuilt = VectorStore(tmp_path / "chroma", "test_chunks", "fake-model", reset=True)
    assert rebuilt.count() == 0


def test_sync_skips_unchanged_documents(store, chunks):
    embedder = FakeEmbedder()
    sync_index(chunks, store, embedder)
    calls_after_first_sync = embedder.calls
    report = sync_index(chunks, store, embedder)
    assert sorted(report.skipped) == ["fda", "mdr"]
    assert report.chunks_embedded == 0
    assert embedder.calls == calls_after_first_sync  # nothing re-embedded


def test_sync_replaces_changed_document_without_leftovers(store, chunks):
    embedder = FakeEmbedder()
    sync_index(chunks, store, embedder)
    # New version of "fda" (different hash) with only ONE chunk instead of two.
    changed = [make_chunk("fda", 0, "software only now", h="h2")] + chunks[2:]
    report = sync_index(changed, store, embedder)
    assert report.added == ["fda"] and report.skipped == ["mdr"]
    assert store.count() == 3  # old fda-p1-c1 must be gone
    assert store.document_hashes()["fda"] == "h2"


def test_sync_removes_deleted_documents(store, chunks):
    embedder = FakeEmbedder()
    sync_index(chunks, store, embedder)
    report = sync_index(chunks[:2], store, embedder)  # "mdr" left the manifest
    assert report.removed == ["mdr"]
    assert set(store.document_hashes()) == {"fda"}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, None),
        ({"category": "eu_regulation"}, {"category": "eu_regulation"}),
        ({"category": ["a", "b"]}, {"category": {"$in": ["a", "b"]}}),
        (
            {"category": "a", "doc_id": "d"},
            {"$and": [{"category": "a"}, {"doc_id": "d"}]},
        ),
    ],
)
def test_make_filter(kwargs, expected):
    assert make_filter(**kwargs) == expected
