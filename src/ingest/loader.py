"""Read PDFs into a list of raw pages. No cleaning or chunking happens here."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .models import Page

logger = logging.getLogger(__name__)

# A real text page has hundreds of characters. Fewer than this usually means
# a scanned image, a blank page or a figure-only page.
MIN_CHARS_PER_PAGE = 50

MANIFEST_COLUMNS = {"filename", "doc_id", "title", "category"}


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """Hash a file's bytes in 1 MB blocks, so large files never fill RAM.

    The same file always gives the same hash, and any change gives a new one.
    That lets re-ingestion skip files that haven't changed.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Read data/manifest.csv and return {filename: row}.

    Keeping titles and categories in a manifest (not in code or filenames)
    means adding a document never requires a code change.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return {row["filename"]: row for row in reader}


def load_pdf(path: Path, doc_id: str, title: str, category: str) -> list[Page]:
    """Extract the text of every page of a PDF.

    Empty pages are kept (with a warning) so page numbers stay correct;
    the cleaner drops them later.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is not a readable PDF or is encrypted.
    """
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(path)
    except PdfReadError as e:
        # Re-raise with the filename, otherwise you can't tell which PDF broke.
        raise ValueError(f"Could not read PDF {path.name}: {e}") from e

    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted: {path.name}")

    pages: list[Page] = []
    # start=1 so page numbers match what a user sees in their PDF viewer.
    for number, pdf_page in enumerate(reader.pages, start=1):
        try:
            # extract_text() can return None, hence `or ""`.
            text = pdf_page.extract_text() or ""
        except Exception as e:  # one bad page shouldn't kill the whole document
            logger.warning("%s p.%d: text extraction failed: %s", path.name, number, e)
            text = ""

        if len(text.strip()) < MIN_CHARS_PER_PAGE:
            logger.warning(
                "%s p.%d: only %d characters (scan or blank page?)",
                path.name,
                number,
                len(text.strip()),
            )

        pages.append(Page(doc_id=doc_id, title=title, category=category, page=number, text=text))

    logger.info("Loaded %s: %d pages", path.name, len(pages))
    return pages
