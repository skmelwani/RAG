"""Tests for the PDF loader and manifest reader.

PDFs are generated inside each test with pypdf, so the tests don't depend on
the real documents in data/raw/ (which are gitignored and absent in CI).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pypdf import PageObject, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ingest.loader import MIN_CHARS_PER_PAGE, file_sha256, load_manifest, load_pdf

LONG_TEXT = "Manufacturers should document the software architecture of the device."  # > 50 chars
assert len(LONG_TEXT) >= MIN_CHARS_PER_PAGE


def _add_text_page(writer: PdfWriter, text: str) -> None:
    """Add a page that draws `text` in Helvetica, so extract_text() can read it back."""
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)


def make_pdf(path: Path, pages: list[str | None], password: str | None = None) -> Path:
    """Write a PDF with one page per entry. None = a blank page (like a scan or figure)."""
    writer = PdfWriter()
    for text in pages:
        if text is None:
            writer.add_blank_page(width=612, height=792)
        else:
            _add_text_page(writer, text)
    if password:
        writer.encrypt(password)
    with path.open("wb") as f:
        writer.write(f)
    return path


def load(path: Path):
    return load_pdf(path, doc_id="doc", title="Test Doc", category="test")


# ---------- load_pdf ----------


def test_loads_one_page_object_per_pdf_page(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", [LONG_TEXT, "Second page talks about cybersecurity controls in detail."])
    pages = load(pdf)
    assert len(pages) == 2
    assert "software architecture" in pages[0].text
    assert "cybersecurity" in pages[1].text


def test_page_numbers_are_one_based(tmp_path):
    # Citations must match the page number a user sees in their PDF viewer.
    pdf = make_pdf(tmp_path / "a.pdf", [LONG_TEXT, LONG_TEXT, LONG_TEXT])
    assert [p.page for p in load(pdf)] == [1, 2, 3]


def test_metadata_is_attached_to_every_page(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", [LONG_TEXT, LONG_TEXT])
    for p in load(pdf):
        assert (p.doc_id, p.title, p.category) == ("doc", "Test Doc", "test")


def test_blank_page_is_kept_and_warned_about(tmp_path, caplog):
    # A blank page must stay in the list, or every later page number shifts by one.
    pdf = make_pdf(tmp_path / "a.pdf", [LONG_TEXT, None, LONG_TEXT])
    with caplog.at_level(logging.WARNING, logger="ingest.loader"):
        pages = load(pdf)
    assert len(pages) == 3
    assert pages[1].text.strip() == ""
    assert pages[2].page == 3
    assert "p.2" in caplog.text and "scan or blank" in caplog.text


def test_failing_page_extraction_does_not_kill_document(tmp_path, monkeypatch, caplog):
    pdf = make_pdf(tmp_path / "a.pdf", [LONG_TEXT, LONG_TEXT])
    original = PageObject.extract_text
    calls = {"n": 0}

    def flaky_extract(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("corrupt content stream")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky_extract)
    with caplog.at_level(logging.WARNING, logger="ingest.loader"):
        pages = load(pdf)
    assert len(pages) == 2
    assert pages[0].text == ""
    assert "software architecture" in pages[1].text
    assert "extraction failed" in caplog.text


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "does_not_exist.pdf")


def test_directory_is_not_a_pdf(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


@pytest.mark.parametrize("content", [b"", b"this is not a pdf at all"])
def test_unreadable_file_raises_value_error_with_filename(tmp_path, content):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(content)
    with pytest.raises(ValueError, match="broken.pdf"):
        load(bad)


def test_encrypted_pdf_raises(tmp_path):
    pdf = make_pdf(tmp_path / "locked.pdf", [LONG_TEXT], password="secret")
    with pytest.raises(ValueError, match="encrypted"):
        load(pdf)


# ---------- file_sha256 ----------


def test_hash_is_stable_and_changes_with_content(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"version 1")
    h1 = file_sha256(f)
    assert file_sha256(f) == h1  # same bytes -> same hash
    assert len(h1) == 64  # hex SHA-256

    f.write_bytes(b"version 2")
    assert file_sha256(f) != h1  # any change -> new hash


def test_hash_does_not_depend_on_block_size(tmp_path):
    # Reading in blocks must give the same result as hashing the whole file at once.
    f = tmp_path / "x.bin"
    f.write_bytes(bytes(range(256)) * 100)
    assert file_sha256(f, block_size=7) == file_sha256(f, block_size=1 << 20)


# ---------- load_manifest ----------

HEADER = "filename,doc_id,title,category,version\n"


def test_manifest_keyed_by_filename(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text(
        HEADER
        + 'a.pdf,doc_a,"Title, with comma",fda_guidance,1\n'
        + "b.pdf,doc_b,Title B,eu_mdr,2\n",
        encoding="utf-8",
    )
    rows = load_manifest(m)
    assert set(rows) == {"a.pdf", "b.pdf"}
    assert rows["a.pdf"]["title"] == "Title, with comma"  # quoted commas parsed correctly
    assert rows["b.pdf"]["category"] == "eu_mdr"


def test_manifest_missing_columns_raises(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text("filename,title\na.pdf,Title A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="category.*doc_id"):
        load_manifest(m)


def test_real_manifest_is_valid():
    # Guards the committed data/manifest.csv: required columns present, ids unique.
    path = Path(__file__).resolve().parents[1] / "data" / "manifest.csv"
    rows = load_manifest(path)
    doc_ids = [r["doc_id"] for r in rows.values()]
    assert len(doc_ids) == len(set(doc_ids)), "duplicate doc_id in manifest"
    assert all(r["title"] and r["category"] for r in rows.values())
