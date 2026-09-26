"""Tests for the PDF reader's error handling (the round trip is covered in test_pdfwriter.py)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import DependencyError

from wellbrief.corpus.pdfwriter import render_pdf
from wellbrief.readers import NoTextLayer, ReaderError
from wellbrief.readers.pdf import PdfPage, read_pdf

LINES = ["DAILY DRILLING REPORT", "Operator: Quillfen Energy    Field: Orrindale    Well: ORD-105"]


def _encrypted(tmp_path: Path, user_password: str) -> Path:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(render_pdf(LINES))))
    writer.encrypt(user_password=user_password, owner_password="owner-secret", algorithm="RC4-128")
    path = tmp_path / "encrypted.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_pages_are_numbered_from_one(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(render_pdf(LINES * 40))
    pages = read_pdf(path)
    assert [page.page_no for page in pages] == [1, 2]
    assert pages[0] == PdfPage(page_no=1, text="\n".join((LINES * 40)[:64]))


def test_owner_password_only_file_is_readable(tmp_path: Path) -> None:
    pages = read_pdf(_encrypted(tmp_path, user_password=""))
    assert [page.text for page in pages] == ["\n".join(LINES)]


def test_password_protected_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="encrypted and needs a password"):
        read_pdf(_encrypted(tmp_path, user_password="user-secret"))


def test_missing_crypto_dependency_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _encrypted(tmp_path, user_password="")

    def fail(self: PdfReader, password: str | bytes) -> None:
        raise DependencyError("cryptography is required for AES algorithm")

    monkeypatch.setattr(PdfReader, "decrypt", fail)
    with pytest.raises(ReaderError, match="cryptography"):
        read_pdf(path)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"hello, this is not a PDF", id="text"),
        pytest.param(b"%PDF-1.4\n%%EOF\n", id="header-only"),
        pytest.param(b"%PDF-1.4\n" + bytes(range(256)) * 20, id="garbage"),
    ],
)
def test_corrupt_files_raise_reader_error(data: bytes, tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(data)
    with pytest.raises(ReaderError):
        read_pdf(path)


def test_file_without_pages_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "no-pages.pdf"
    with path.open("wb") as handle:
        PdfWriter().write(handle)
    with pytest.raises(ReaderError, match="no pages"):
        read_pdf(path)


def test_file_without_a_text_layer_is_reported(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    path = tmp_path / "scanned.pdf"
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(NoTextLayer) as caught:
        read_pdf(path)
    assert isinstance(caught.value, ReaderError)
    assert str(caught.value) == "PDF has no text layer; OCR is not supported in v0.1"


def test_whitespace_only_text_counts_as_no_text_layer(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    path.write_bytes(render_pdf(["", "   ", ""]))
    with pytest.raises(NoTextLayer):
        read_pdf(path)


def test_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / "absent.pdf")
