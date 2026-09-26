"""Tests for the built-in DOCX writer, including the text round trip through the DOCX reader."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from wellbrief.corpus.docxwriter import (
    DOCX_ENTRY_ORDER,
    MONOSPACE_FONT,
    ZIP_DATE_TIME,
    render_docx,
    write_docx,
)
from wellbrief.readers.docx import read_docx

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _round_trip(lines: list[str], tmp_path: Path) -> str:
    path = tmp_path / "doc.docx"
    write_docx(lines, path)
    return read_docx(path)


def _document(data: bytes) -> ET.Element:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return ET.fromstring(archive.read("word/document.xml"))


# Round trip ---------------------------------------------------------------------------------------


def test_round_trip_reproduces_sample_reports_exactly(sample_lines: list[str], tmp_path: Path) -> None:
    assert _round_trip(sample_lines, tmp_path) == "\n".join(sample_lines)


def test_round_trip_keeps_spaces_blank_lines_and_markup_characters(tmp_path: Path) -> None:
    lines = [
        "",
        "  leading",
        "runs    of    spaces   ",
        "   ",
        "",
        "<tag attr=\"v\"> & 'quotes' ]]>",
        "a\tb\t\tc\t",
        "\tleading tab",
        "",
    ]
    assert _round_trip(lines, tmp_path) == "\n".join(lines)


def test_non_ascii_text_round_trips(tmp_path: Path) -> None:
    line = (
        '12\N{VULGAR FRACTION ONE QUARTER}" \N{EN DASH} 118 \N{DEGREE SIGN}C \N{EURO SIGN} '
        "\N{MATHEMATICAL BOLD CAPITAL A}"
    )
    lines = [line]
    assert _round_trip(lines, tmp_path) == lines[0]


def test_characters_xml_cannot_hold_are_replaced(tmp_path: Path) -> None:
    lines = ["a\x00b\x0bc\U0000fffed", "lone \U0000d800 surrogate"]
    assert _round_trip(lines, tmp_path) == (
        "a\N{REPLACEMENT CHARACTER}b\N{REPLACEMENT CHARACTER}c\N{REPLACEMENT CHARACTER}d\n"
        "lone \N{REPLACEMENT CHARACTER} surrogate"
    )


@pytest.mark.parametrize("bad", ["a\nb", "a\rb", "a\r\nb"])
def test_line_breaks_inside_a_line_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="line break"):
        render_docx(["ok", bad])


def test_empty_document_reads_as_empty_text(tmp_path: Path) -> None:
    assert _round_trip([], tmp_path) == ""


# Structure ----------------------------------------------------------------------------------------


def test_one_paragraph_per_line_with_preserved_space_runs(ddr_lines: list[str]) -> None:
    body = _document(render_docx(ddr_lines)).find(f"{W}body")
    assert body is not None
    paragraphs = body.findall(f"{W}p")

    assert len(paragraphs) == len(ddr_lines)
    for paragraph, line in zip(paragraphs, ddr_lines, strict=True):
        texts = paragraph.findall(f".//{W}t")
        assert "".join(t.text or "" for t in texts) == line
        assert all(t.get(XML_SPACE) == "preserve" for t in texts)


def test_runs_use_a_monospace_font() -> None:
    fonts = _document(render_docx(["x"])).find(f".//{W}rFonts")
    assert fonts is not None
    assert fonts.get(f"{W}ascii") == MONOSPACE_FONT


def test_package_contains_only_the_three_required_parts(sample_lines: list[str]) -> None:
    with zipfile.ZipFile(io.BytesIO(render_docx(sample_lines))) as archive:
        assert tuple(archive.namelist()) == DOCX_ENTRY_ORDER
        assert archive.testzip() is None
        content_types = archive.read("[Content_Types].xml").decode("utf-8")
        rels = archive.read("_rels/.rels").decode("utf-8")
    assert "wordprocessingml.document.main+xml" in content_types
    assert 'Target="word/document.xml"' in rels


# Determinism and metadata -------------------------------------------------------------------------


def test_output_is_byte_identical_for_identical_input(sample_lines: list[str], tmp_path: Path) -> None:
    first, second = tmp_path / "a.docx", tmp_path / "b.docx"
    write_docx(sample_lines, first)
    write_docx(list(sample_lines), second)
    assert first.read_bytes() == second.read_bytes() == render_docx(sample_lines)


def test_output_carries_no_metadata_and_fixed_entry_attributes(sample_lines: list[str]) -> None:
    with zipfile.ZipFile(io.BytesIO(render_docx(sample_lines))) as archive:
        infos = archive.infolist()
        comment = archive.comment
    assert not any(info.filename.startswith("docProps") for info in infos)
    assert all(info.date_time == ZIP_DATE_TIME == (1980, 1, 1, 0, 0, 0) for info in infos)
    assert {info.compress_type for info in infos} == {zipfile.ZIP_STORED}
    assert {info.external_attr >> 16 for info in infos} == {0o100644}
    assert {info.create_system for info in infos} == {3}
    assert all(info.extra == b"" and info.comment == b"" for info in infos)
    assert comment == b""
