"""Tests for the RAG pipeline with a fake embedder, store and LLM (no models, no Ollama)."""

from __future__ import annotations

import numpy as np
import pytest

from generation.llm import LLMUnavailableError
from generation.prompt import REFUSAL
from rag.pipeline import RagPipeline, is_refusal, parse_citations
from retrieval.vector_store import SearchResult


def result(n: int, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk_id=f"doc-p{n}-c0",
        text=f"Chunk {n} text.",
        doc_id="doc",
        title="Cyber Guidance",
        category="fda_guidance",
        page=n * 10,
        score=score,
    )


class FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_k: int | None = None
        self.last_where = None

    def search(self, query_embedding, k, where=None):
        self.last_k, self.last_where = k, where
        return self.results[:k]


class FakeLLM:
    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


def make(results, reply="", min_score=0.5, error=None):
    store, llm = FakeStore(results), FakeLLM(reply, error)
    return RagPipeline(FakeEmbedder(), store, llm, top_k=3, min_score=min_score), store, llm


THREE = [result(1, 0.9), result(2, 0.8), result(3, 0.7)]


# ---------- answers and citations ----------


def test_answer_maps_citation_numbers_to_title_and_page():
    pipe, _, llm = make(THREE, "SBOMs list components [2]. They are required [1][2].")
    out = pipe.answer("What is an SBOM?")
    assert not out.refused
    assert [c.number for c in out.citations] == [2, 1]  # order of first use, no duplicates
    assert (out.citations[0].page, out.citations[1].page) == (20, 10)
    assert out.citations[0].label() == "[2] Cyber Guidance, p. 20"
    assert out.sources == THREE
    assert len(llm.calls) == 1


def test_invalid_citation_numbers_are_removed_and_warned():
    pipe, _, _ = make(THREE, "Claim A [1][7]. Claim B [9]. Claim C [2, 8].")
    out = pipe.answer("q")
    assert [c.number for c in out.citations] == [1, 2]
    assert out.answer == "Claim A [1]. Claim B. Claim C [2]."
    assert any("[7, 9, 8]" in w for w in out.warnings)


def test_answer_without_citations_is_flagged():
    pipe, _, _ = make(THREE, "SBOMs are useful.")
    out = pipe.answer("q")
    assert not out.refused
    assert out.citations == []
    assert any("no citations" in w for w in out.warnings)


def test_timings_are_recorded():
    pipe, _, _ = make(THREE, "Answer [1].")
    t = pipe.answer("q").timings_ms
    assert {"retrieval", "generation", "total"} <= set(t)
    assert t["total"] == pytest.approx(t["retrieval"] + t["generation"])


def test_top_k_and_filter_are_passed_to_store():
    pipe, store, _ = make(THREE, "Answer [1].")
    pipe.answer("q", top_k=2, where={"category": "eu_mdr"})
    assert store.last_k == 2
    assert store.last_where == {"category": "eu_mdr"}


def test_prompt_contains_retrieved_chunks():
    pipe, _, llm = make(THREE, "Answer [1].")
    pipe.answer("What is an SBOM?")
    user = llm.calls[0][1]["content"]
    assert "Chunk 1 text." in user and "Chunk 3 text." in user
    assert "Question: What is an SBOM?" in user


# ---------- refusals ----------


def test_empty_index_refuses_without_calling_llm():
    pipe, _, llm = make([], "should not be used")
    out = pipe.answer("q")
    assert out.refused and out.refusal_reason == "no_results"
    assert out.answer == REFUSAL
    assert llm.calls == []


def test_low_score_refuses_without_calling_llm():
    pipe, _, llm = make([result(1, 0.4)], "should not be used", min_score=0.5)
    out = pipe.answer("What is the capital of France?")
    assert out.refused and out.refusal_reason == "low_score"
    assert out.sources  # still returned, for the UI / evaluation
    assert llm.calls == []


def test_llm_refusal_is_detected():
    pipe, _, llm = make(THREE, "I could not find this in the documents.")
    out = pipe.answer("q")
    assert out.refused and out.refusal_reason == "llm"
    assert out.citations == []
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        REFUSAL,
        "i could not find this in the documents",
        "Sorry. I could NOT find this in the documents!",
    ],
)
def test_is_refusal_tolerates_case_and_punctuation(text):
    assert is_refusal(text)


def test_normal_answer_is_not_a_refusal():
    assert not is_refusal("The documents describe SBOM contents [1].")


# ---------- bad input and errors ----------


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_empty_question_raises(question):
    pipe, _, _ = make(THREE)
    with pytest.raises(ValueError, match="empty"):
        pipe.answer(question)


def test_invalid_top_k_raises():
    pipe, _, _ = make(THREE)
    with pytest.raises(ValueError, match="top_k"):
        pipe.answer("q", top_k=-1)


def test_llm_unavailable_propagates():
    pipe, _, _ = make(THREE, error=LLMUnavailableError("down"))
    with pytest.raises(LLMUnavailableError):
        pipe.answer("q")


# ---------- parse_citations ----------


@pytest.mark.parametrize(
    "text, valid, invalid",
    [
        ("No citations.", [], []),
        ("A [1]. B [2][3].", [1, 2, 3], []),
        ("A [1, 3]. B [3].", [1, 3], []),
        ("A [0]. B [4]. C [2].", [2], [0, 4]),
        ("See [a] and [1.5] and [].", [], []),  # not citations
    ],
)
def test_parse_citations(text, valid, invalid):
    assert parse_citations(text, n_sources=3) == (valid, invalid)
