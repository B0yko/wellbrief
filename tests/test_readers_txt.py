"""Tests for the plain-text reader and the package's import surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from wellbrief.readers import ReaderError
from wellbrief.readers.txt import decode_text, read_text

SRC = Path(__file__).resolve().parents[1] / "src"


def test_utf8_text_is_returned_unchanged(tmp_path: Path) -> None:
    text = 'Hole section          : 17 1/2"\n  12\N{VULGAR FRACTION ONE QUARTER}" \N{EN DASH} ok\n'
    path = tmp_path / "ddr.txt"
    path.write_text(text, encoding="utf-8")
    assert read_text(path) == text


def test_bom_is_stripped_and_line_endings_are_normalised(tmp_path: Path) -> None:
    path = tmp_path / "ddr.md"
    path.write_bytes(b"\xef\xbb\xbfline 1\r\nline 2\rline 3\n\r\n")
    assert read_text(path) == "line 1\nline 2\nline 3\n\n"


def test_only_a_leading_bom_is_stripped() -> None:
    assert decode_text(b"a\xef\xbb\xbfb") == "a\N{ZERO WIDTH NO-BREAK SPACE}b"


def test_invalid_utf8_reports_the_offset(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"caf\xe9 au lait")
    with pytest.raises(ReaderError, match="offset 3"):
        read_text(path)


def test_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text(tmp_path / "absent.txt")


def test_importing_readers_and_writers_does_not_import_pypdf() -> None:
    code = (
        "import sys\n"
        "import wellbrief.readers, wellbrief.readers.txt, wellbrief.readers.docx\n"
        "import wellbrief.readers.csvledger, wellbrief.corpus.pdfwriter, wellbrief.corpus.docxwriter\n"
        "assert 'pypdf' not in sys.modules, 'pypdf imported'\n"
        "import wellbrief.readers.pdf\n"
        "assert 'pypdf' in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
