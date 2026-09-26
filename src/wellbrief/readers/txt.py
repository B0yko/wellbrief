"""Plain-text reader for ``.txt`` and ``.md`` files."""

from __future__ import annotations

import codecs
import os
from pathlib import Path

from . import ReaderError

__all__ = ["decode_text", "read_text"]


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a UTF-8 text file and return its text with normalised line endings.

    A leading byte order mark is removed, and CRLF and lone CR line endings become LF.

    Raises:
        ReaderError: If the file is not valid UTF-8.
        OSError: If the file cannot be read.
    """
    return decode_text(Path(path).read_bytes())


def decode_text(data: bytes) -> str:
    """Decode UTF-8 ``data`` as :func:`read_text` does.

    Raises:
        ReaderError: If ``data`` is not valid UTF-8; the message gives the byte offset.
    """
    data = data.removeprefix(codecs.BOM_UTF8)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(f"file is not valid UTF-8 (invalid byte at offset {exc.start})") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")
