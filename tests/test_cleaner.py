"""Tests for the cleaner rules found while inspecting the real PDFs."""

from __future__ import annotations

from ingest.cleaner import clean_pages, normalize_text, strip_line_numbers
from ingest.models import Page


def make_pages(texts: list[str]) -> list[Page]:
    return [Page("doc", "T", "c", i, t) for i, t in enumerate(texts, start=1)]


def test_ligatures_and_hyphenation_fixed():
    assert normalize_text("the ﬁrst docu-\nment") == "the first document"


def test_lines_joined_but_paragraphs_kept():
    assert normalize_text("line one\nline two\n\nnext para") == "line one line two\n\nnext para"


def test_dot_leaders_removed():
    assert normalize_text("Introduction ........... 1") == "Introduction 1"


def test_repeated_headers_and_page_numbers_removed():
    pages = make_pages(
        [f"Contains Nonbinding Recommendations\nUnique content {w}.\n{i}" for i, w in
         enumerate(["alpha", "beta", "gamma", "delta"], start=1)]
    )
    cleaned = clean_pages(pages)
    assert [p.text for p in cleaned] == [
        "Unique content alpha.", "Unique content beta.", "Unique content gamma.", "Unique content delta."
    ]


def test_page_numbers_kept_and_empty_pages_dropped():
    pages = make_pages(["Header\nReal text one.", "Header", "Header\nReal text three."])
    cleaned = clean_pages(pages)
    assert [p.page for p in cleaned] == [1, 3]


def test_strip_line_numbers():
    assert strip_line_numbers("should be1236\nreflective of 1237") == "should be\nreflective of"


def test_line_numbers_only_stripped_when_asked():
    # Different words per page, so the header detector doesn't remove them.
    pages = make_pages(["See Article 115", "Annex VIII rule 11", "Recital 19 and 20"])
    assert [p.text for p in clean_pages(pages)] == [
        "See Article 115", "Annex VIII rule 11", "Recital 19 and 20"
    ]
