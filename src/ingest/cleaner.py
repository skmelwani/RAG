"""Clean raw PDF text: fix unicode, rejoin broken lines, remove headers/footers.

All functions are pure (input in, new output out), which makes them easy to test.
These rules are a starting point: tune them after looking at real extracted text.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import replace

from .models import Page

logger = logging.getLogger(__name__)

# "cyber-\nsecurity" -> "cybersecurity". Trade-off: a real hyphen at a line
# end ("pre-\nmarket") also gets joined. That's rare enough to accept.
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
# A blank line (possibly containing spaces) separates paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_MULTI_SPACE = re.compile(r"[ \t]+")
# Control characters PDFs sometimes contain (keeps \t and \n).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DIGITS = re.compile(r"\d+")
# Table-of-contents leaders: "Introduction ........ 1". Pure noise for retrieval.
_DOT_LEADER = re.compile(r"(?:\.\s?){4,}")
# FDA draft guidances print line numbers in the margin, which pypdf glues onto
# the end of each line ("should be1236"). Only used when the manifest says so,
# because on other documents it would eat real numbers ("Article 115").
_TRAILING_LINE_NUMBER = re.compile(r"\s*\d{1,4}\s*$")
# A table-of-contents line ends in a page number ("Introduction 4") or has a
# dot leader ("Introduction ........ 4").
_TOC_LINE = re.compile(r"\s\d{1,3}\s*$|\.{4,}|(?:\. ){4,}")

# Measured on our PDFs: TOC pages score 0.76-0.95, every other page <= 0.36.
TOC_LINE_RATIO = 0.6

# Headers and footers are short. Longer lines are real content even if repeated.
_MAX_HEADER_LEN = 100


def _line_key(line: str) -> str:
    """Normalise a line for comparison: "Page 3 of 40" and "Page 4 of 40" match."""
    return _DIGITS.sub("#", line.strip().lower())


def find_repeated_lines(pages: list[Page], threshold: float = 0.5, min_pages: int = 3) -> set[str]:
    """Find header/footer lines: short lines that appear on most pages of ONE document.

    Args:
        pages: pages of a single document (don't mix documents).
        threshold: a line counts as repeated if it's on more than this share of pages.
        min_pages: with fewer pages, repetition means nothing, so return nothing.
    """
    if len(pages) < min_pages:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        # A set per page: a line repeated 5 times on ONE page still counts once.
        keys = {
            _line_key(line)
            for line in page.text.splitlines()
            if line.strip() and len(line.strip()) <= _MAX_HEADER_LEN
        }
        counts.update(keys)

    cutoff = threshold * len(pages)
    return {key for key, count in counts.items() if count > cutoff}


def remove_lines(text: str, lines_to_remove: set[str]) -> str:
    """Drop every line whose normalised form is in lines_to_remove."""
    return "\n".join(line for line in text.splitlines() if _line_key(line) not in lines_to_remove)


def strip_line_numbers(text: str) -> str:
    """Remove margin line numbers from the end of every line (draft guidances only)."""
    return "\n".join(_TRAILING_LINE_NUMBER.sub("", line) for line in text.splitlines())


def is_toc_page(text: str, threshold: float = TOC_LINE_RATIO, min_lines: int = 5) -> bool:
    """True if most lines look like table-of-contents entries.

    Must run on RAW text (lines intact, dots present) and AFTER margin line
    numbers are stripped: otherwise every line of a draft guidance "ends in a
    number" and every page looks like a TOC.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < min_lines:
        # Tiny pages (title slides) give unreliable ratios.
        return False
    hits = sum(1 for line in lines if _TOC_LINE.search(line))
    return hits / len(lines) >= threshold


def normalize_text(text: str) -> str:
    """Fix unicode and whitespace while keeping paragraph boundaries ("\\n\\n").

    PDF text has a hard line break at the end of every visual line. We join
    lines inside a paragraph into one line, but keep blank lines between
    paragraphs, because the chunker prefers to split there.
    """
    # NFKC turns ligatures ("ﬁ" -> "fi") and odd spaces into plain characters.
    # Without it, "ﬁle" and "file" would be different words for BM25 later.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _DOT_LEADER.sub(" ", text)
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)

    paragraphs = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        joined = " ".join(line.strip() for line in paragraph.splitlines())
        joined = _MULTI_SPACE.sub(" ", joined).strip()
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def clean_pages(pages: list[Page], has_line_numbers: bool = False) -> list[Page]:
    """Clean all pages of ONE document. Pages that end up empty are dropped.

    Page numbers are untouched, so citations still point to the right page.

    Args:
        has_line_numbers: strip margin line numbers first (set per document
            in the manifest). This runs before header detection, otherwise a
            numbered header would look different on every page.
    """
    if has_line_numbers:
        pages = [replace(page, text=strip_line_numbers(page.text)) for page in pages]

    # Drop table-of-contents pages: they mention every section title, so they
    # match many queries and would become useless citations ("see p.4").
    toc_pages = [page.page for page in pages if is_toc_page(page.text)]
    if toc_pages:
        logger.info("%s: skipping table-of-contents pages %s", pages[0].doc_id, toc_pages)
        pages = [page for page in pages if page.page not in toc_pages]

    repeated = find_repeated_lines(pages)
    cleaned: list[Page] = []
    for page in pages:
        text = normalize_text(remove_lines(page.text, repeated))
        if text:
            # replace() copies the frozen Page with only `text` changed.
            cleaned.append(replace(page, text=text))
    return cleaned
