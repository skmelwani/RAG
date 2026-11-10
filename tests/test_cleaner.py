"""Tests for the cleaner rules found while inspecting the real PDFs."""

from __future__ import annotations

from ingest.cleaner import clean_pages, is_toc_page, normalize_text, strip_line_numbers
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


DOTTED_TOC = """Table of Contents
I. Introduction ........................ 1
II. Scope .............................. 2
III. Background ........................ 3
IV. Recommendations .................... 5
V. Premarket Submissions .............. 9
Appendix 1 ........................... 38"""

# MDCG style: no dot leaders, just page numbers at line ends.
PLAIN_TOC = """Contents
1. Scope and purpose of this document 4
2. Definitions 5
3. Qualification criteria 7
4. Classification rules 15
5. IVD software 21"""

BODY = """Manufacturers should provide an SBOM that includes commercial,
open-source and off-the-shelf software components. The SBOM should
be machine readable and kept up to date throughout the device
lifecycle, as described in Section V.A.4 of this guidance.
Article 115 sets out how delegated acts are adopted."""


def test_toc_pages_detected_with_and_without_dot_leaders():
    assert is_toc_page(DOTTED_TOC)
    assert is_toc_page(PLAIN_TOC)


def test_body_page_is_not_toc():
    assert not is_toc_page(BODY)


def test_short_page_is_not_toc():
    assert not is_toc_page("Overview\nSection 1 2\nSection 2 3")


def test_margin_line_numbers_look_like_toc_until_stripped():
    # Draft guidances have a line number at the end of every line.
    numbered = "\n".join(f"{line} {n}" for n, line in enumerate(BODY.splitlines(), start=21))
    assert is_toc_page(numbered)  # why stripping must come first
    assert not is_toc_page(strip_line_numbers(numbered))


def test_clean_pages_drops_toc_but_keeps_page_numbers():
    pages = make_pages([DOTTED_TOC, BODY, BODY.replace("SBOM", "label")])
    assert [p.page for p in clean_pages(pages)] == [2, 3]
