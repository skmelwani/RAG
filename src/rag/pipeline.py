"""The RAG pipeline: retrieve -> (refuse early?) -> prompt -> generate -> cite.

Every component is passed in (dependency injection) and typed as a Protocol,
so tests can use a fake embedder, store and LLM: fast, offline, deterministic.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from generation.llm import Message
from generation.prompt import REFUSAL, build_messages
from retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

# [1], [2][3] and [1, 2] are all valid citation styles a model might produce.
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

RefusalReason = Literal["no_results", "low_score", "llm"]


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> np.ndarray: ...


class Searcher(Protocol):
    def search(
        self, query_embedding: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...


class ChatModel(Protocol):
    def chat(self, messages: list[Message]) -> str: ...


@dataclass(frozen=True)
class Citation:
    """One source the answer actually cites, e.g. [2] -> (title, page)."""

    number: int  # the [n] used in the answer text
    title: str
    page: int
    doc_id: str
    chunk_id: str

    def label(self) -> str:
        return f"[{self.number}] {self.title}, p. {self.page}"


@dataclass
class RagAnswer:
    question: str
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None = None
    citations: list[Citation] = field(default_factory=list)  # sources the answer cites
    sources: list[SearchResult] = field(default_factory=list)  # everything retrieved
    warnings: list[str] = field(default_factory=list)  # e.g. "answer has no citations"
    timings_ms: dict[str, float] = field(default_factory=dict)


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) source numbers cited in `text`, in order of first use.

    A number is valid if a source with that number was actually in the prompt.
    """
    valid: list[int] = []
    invalid: list[int] = []
    for match in _CITATION_PATTERN.finditer(text):
        for part in match.group(1).split(","):
            n = int(part)
            target = valid if 1 <= n <= n_sources else invalid
            if n not in target:
                target.append(n)
    return valid, invalid


def is_refusal(text: str) -> bool:
    """True if the model gave the refusal sentence (tolerant of case/punctuation)."""
    normalise = lambda s: re.sub(r"[^a-z ]", "", s.lower())  # noqa: E731
    return normalise(REFUSAL).strip() in normalise(text)


class RagPipeline:
    def __init__(
        self,
        embedder: QueryEmbedder,
        store: Searcher,
        llm: ChatModel,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def answer(
        self, question: str, top_k: int | None = None, where: dict[str, Any] | None = None
    ) -> RagAnswer:
        """Answer `question` from the indexed documents, with citations.

        Raises:
            ValueError: empty question or top_k < 1.
            generation.llm.LLMError / LLMUnavailableError: the LLM call failed.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        k = top_k or self.top_k
        if k < 1:
            raise ValueError(f"top_k must be >= 1, got {k}")

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        results = self.store.search(self.embedder.embed_query(question), k=k, where=where)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        # Refuse BEFORE the LLM when retrieval found nothing useful:
        # faster, and a model that is never called cannot hallucinate.
        if not results:
            return self._refusal(question, "no_results", results, timings)
        if results[0].score < self.min_score:
            logger.info("Best score %.3f < min_score %.3f: refusing", results[0].score, self.min_score)
            return self._refusal(question, "low_score", results, timings)

        t1 = time.perf_counter()
        text = self.llm.chat(build_messages(question, results))
        timings["generation"] = (time.perf_counter() - t1) * 1000
        timings["total"] = timings["retrieval"] + timings["generation"]

        if is_refusal(text):
            return self._refusal(question, "llm", results, timings)

        valid, invalid = parse_citations(text, len(results))
        warnings: list[str] = []
        if invalid:
            # The model cited a source that was never shown to it: a sign of hallucination.
            warnings.append(f"invalid citation numbers removed: {invalid}")
            text = self._remove_invalid(text, len(results))
        if not valid:
            warnings.append("answer has no citations (may not be grounded)")
        for w in warnings:
            logger.warning("%s | question: %s", w, question)

        citations = [
            Citation(
                number=n,
                title=results[n - 1].title,
                page=results[n - 1].page,
                doc_id=results[n - 1].doc_id,
                chunk_id=results[n - 1].chunk_id,
            )
            for n in valid
        ]
        return RagAnswer(
            question=question,
            answer=text,
            refused=False,
            citations=citations,
            sources=results,
            warnings=warnings,
            timings_ms=timings,
        )

    @staticmethod
    def _remove_invalid(text: str, n_sources: int) -> str:
        """Drop out-of-range numbers from citation brackets; drop empty brackets."""

        def fix(match: re.Match[str]) -> str:
            kept = [p.strip() for p in match.group(1).split(",") if 1 <= int(p) <= n_sources]
            return f"[{', '.join(kept)}]" if kept else ""

        return re.sub(r" +([.,;:])", r"\1", _CITATION_PATTERN.sub(fix, text)).strip()

    @staticmethod
    def _refusal(
        question: str,
        reason: RefusalReason,
        results: list[SearchResult],
        timings: dict[str, float],
    ) -> RagAnswer:
        timings.setdefault("total", sum(timings.values()))
        return RagAnswer(
            question=question,
            answer=REFUSAL,
            refused=True,
            refusal_reason=reason,
            sources=results,  # still returned, so a UI/eval can show what was found
            timings_ms=timings,
        )
