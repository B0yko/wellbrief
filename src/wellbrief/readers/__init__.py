"""File readers used by ingestion.

Each reader turns one file into text (or, for a CSV NPT ledger, into rows) and knows nothing about
report types, parsing or storage:

* :func:`wellbrief.readers.txt.read_text` for ``.txt`` and ``.md`` files;
* :func:`wellbrief.readers.pdf.read_pdf` for the text layer of ``.pdf`` files, page by page;
* :func:`wellbrief.readers.docx.read_docx` for body paragraphs and tables of ``.docx`` files;
* :func:`wellbrief.readers.csvledger.read_ledger` for NPT ledgers in ``.csv`` files.

Every reader reads the whole file first, so a missing or unreadable file raises the usual
:class:`OSError`. Problems with the content raise :class:`ReaderError`, whose message is meant to
be shown to the user as the reason a file was skipped. The submodules are imported on demand, so
importing this package does not import ``pypdf``.
"""

from __future__ import annotations

__all__ = ["NoTextLayer", "ReaderError"]


class ReaderError(Exception):
    """A file could not be read because of its content (corrupt, encrypted, unsupported)."""


class NoTextLayer(ReaderError):
    """A PDF has pages but none of them carries extractable text (for example a scanned file)."""

    def __init__(self, message: str = "PDF has no text layer; OCR is not supported in v0.1") -> None:
        super().__init__(message)
