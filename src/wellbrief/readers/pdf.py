"""PDF text-layer reader built on ``pypdf``.

Text is extracted page by page with ``pypdf``'s plain extraction mode, which follows the order of
the text operators in each content stream. Scanned documents without a text layer are reported
with :class:`~wellbrief.readers.NoTextLayer`; OCR is out of scope.

Encrypted files are opened with the empty user password, which covers the common case of a file
restricted only by an owner password. Files that need a real password, or AES-encrypted files when
the optional ``cryptography`` package is not installed, raise :class:`~wellbrief.readers.ReaderError`.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

from pypdf import PasswordType, PdfReader
from pypdf.errors import DependencyError

from . import NoTextLayer, ReaderError

__all__ = ["NoTextLayer", "PdfPage", "ReaderError", "read_pdf"]


@dataclass(frozen=True, slots=True)
class PdfPage:
    """Extracted text of one PDF page.

    Attributes:
        page_no: Page number, starting at 1.
        text: Text of the page, lines separated by newline characters.
    """

    page_no: int
    text: str


def read_pdf(path: str | os.PathLike[str]) -> list[PdfPage]:
    """Extract the text layer of every page of a PDF file.

    Raises:
        NoTextLayer: If the document has pages but no page yields any non-whitespace text.
        ReaderError: If the file is not a readable PDF, is encrypted with a password, has no
            pages, or cannot be decrypted without an optional dependency.
        OSError: If the file cannot be read.
    """
    data = Path(path).read_bytes()
    try:
        pages = _extract_pages(data)
    except ReaderError:
        raise
    except Exception as exc:  # pypdf raises many exception types on malformed input
        raise ReaderError(f"cannot read PDF: {type(exc).__name__}: {exc}") from exc
    if not pages:
        raise ReaderError("PDF contains no pages")
    if not any(page.text.strip() for page in pages):
        raise NoTextLayer()
    return pages


def _extract_pages(data: bytes) -> list[PdfPage]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
        except DependencyError as exc:
            raise ReaderError(
                "PDF is encrypted with an algorithm that needs the optional 'cryptography' package"
            ) from exc
        if result == PasswordType.NOT_DECRYPTED:
            raise ReaderError("PDF is encrypted and needs a password")
    pages: list[PdfPage] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="plain")
        pages.append(PdfPage(page_no=number, text=text.replace("\r\n", "\n").replace("\r", "\n")))
    return pages
