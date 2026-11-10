"""Tests for the chunker. Run with: pytest"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from ingest.chunker import chunk_page, count_tokens
from ingest.models import Page

CHUNK_SIZE = 64
OVERLAP = 16


@pytest.fixture(scope="module")
def tokenizer():
    # scope="module": load once for all tests in this file, not once per test.
    # Uses the local Hugging Face cache, so it works offline after the first run.
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def make_page(text: str, page: int = 3) -> Page:
    return Page(doc_id="doc", title="Test Doc", category="test", page=page, text=text)


def long_text(n_sentences: int = 40) -> str:
    return " ".join(f"Sentence {i} explains a different safety control." for i in range(n_sentences))


def test_no_chunk_exceeds_chunk_size(tokenizer):
    chunks = chunk_page(make_page(long_text()), tokenizer, CHUNK_SIZE, OVERLAP, "h")
    assert len(chunks) > 1
    assert all(c.n_tokens <= CHUNK_SIZE for c in chunks)


def test_consecutive_chunks_overlap(tokenizer):
    chunks = chunk_page(make_page(long_text()), tokenizer, CHUNK_SIZE, OVERLAP, "h")
    for prev, nxt in zip(chunks, chunks[1:]):
        # The next chunk starts with a sentence the previous chunk ended with,
        # and the previous chunk's last sentence is carried over too.
        first_sentence = nxt.text.split(". ")[0]
        last_sentence = prev.text.split(". ")[-1]
        assert first_sentence in prev.text
        assert last_sentence in nxt.text


def test_zero_overlap_means_no_shared_text(tokenizer):
    chunks = chunk_page(make_page(long_text()), tokenizer, CHUNK_SIZE, 0, "h")
    assert chunks[0].text.split(". ")[-1] not in chunks[1].text


def test_no_text_is_lost(tokenizer):
    text = long_text()
    chunks = chunk_page(make_page(text), tokenizer, CHUNK_SIZE, OVERLAP, "h")
    joined = " ".join(c.text for c in chunks)
    for i in range(40):
        assert f"Sentence {i} " in joined


@pytest.mark.parametrize("overlap", [CHUNK_SIZE, CHUNK_SIZE + 1, -1])
def test_invalid_overlap_raises(tokenizer, overlap):
    with pytest.raises(ValueError):
        chunk_page(make_page("Some text."), tokenizer, CHUNK_SIZE, overlap, "h")


def test_invalid_chunk_size_raises(tokenizer):
    with pytest.raises(ValueError):
        chunk_page(make_page("Some text."), tokenizer, 0, 0, "h")


def test_chunk_size_must_leave_room_for_special_tokens(tokenizer):
    # bge-small reads 512 tokens including [CLS] and [SEP], so 510 is the max.
    chunk_page(make_page("Some text."), tokenizer, 510, 40, "h")  # fine
    with pytest.raises(ValueError, match="at most 510"):
        chunk_page(make_page("Some text."), tokenizer, 512, 40, "h")


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_page_gives_no_chunks(tokenizer, text):
    assert chunk_page(make_page(text), tokenizer, CHUNK_SIZE, OVERLAP, "h") == []


def test_chunk_ids_are_deterministic(tokenizer):
    page = make_page(long_text())
    first = [c.chunk_id for c in chunk_page(page, tokenizer, CHUNK_SIZE, OVERLAP, "h")]
    second = [c.chunk_id for c in chunk_page(page, tokenizer, CHUNK_SIZE, OVERLAP, "h")]
    assert first == second
    assert first[0] == "doc-p3-c0"
    assert len(set(first)) == len(first)  # ids are unique


def test_metadata_is_copied_from_page(tokenizer):
    chunk = chunk_page(make_page("A short page.", page=7), tokenizer, CHUNK_SIZE, OVERLAP, "abc")[0]
    assert (chunk.doc_id, chunk.title, chunk.category, chunk.page) == ("doc", "Test Doc", "test", 7)
    assert chunk.source_hash == "abc"
    assert chunk.chunk_index == 0


def test_huge_sentence_is_hard_split_and_keeps_case(tokenizer):
    # One "sentence" far longer than chunk_size, with no punctuation to split on.
    text = " ".join(["FDA"] * 300)
    chunks = chunk_page(make_page(text), tokenizer, CHUNK_SIZE, OVERLAP, "h")
    assert len(chunks) > 1
    assert all(c.n_tokens <= CHUNK_SIZE for c in chunks)
    # Slicing the original string (not decoding tokens) keeps uppercase.
    assert all(c.text.startswith("FDA") for c in chunks)


def test_paragraph_boundaries_preferred(tokenizer):
    para_a = "First paragraph about risk management."
    para_b = "Second paragraph about software validation."
    chunks = chunk_page(make_page(f"{para_a}\n\n{para_b}"), tokenizer, CHUNK_SIZE, OVERLAP, "h")
    assert len(chunks) == 1  # both fit together
    assert para_a in chunks[0].text and para_b in chunks[0].text


def test_count_tokens_excludes_special_tokens(tokenizer):
    # "hello" is one token; with [CLS]/[SEP] it would be 3.
    assert count_tokens("hello", tokenizer) == 1
