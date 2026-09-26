"""Tests for the built-in PDF writer, including the text round trip through the PDF reader."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, NumberObject

from wellbrief.corpus.pdfwriter import (
    LINES_PER_PAGE,
    encode_line,
    escape_pdf_string,
    layout_for,
    render_pdf,
    write_pdf,
)
from wellbrief.readers import NoTextLayer
from wellbrief.readers.pdf import read_pdf


def _round_trip(lines: list[str], tmp_path: Path) -> str:
    path = tmp_path / "doc.pdf"
    write_pdf(lines, path)
    return "\n".join(page.text for page in read_pdf(path))


def _content_streams(data: bytes) -> list[bytes]:
    reader = PdfReader(io.BytesIO(data))
    return [page.get_contents().get_data() for page in reader.pages]  # type: ignore[union-attr]


# Round trip ---------------------------------------------------------------------------------------


def test_round_trip_reproduces_sample_reports_exactly(sample_lines: list[str], tmp_path: Path) -> None:
    assert _round_trip(sample_lines, tmp_path) == "\n".join(sample_lines)


def test_round_trip_keeps_space_runs_used_by_the_header_parser(ddr_lines: list[str], tmp_path: Path) -> None:
    path = tmp_path / "ddr.pdf"
    write_pdf(ddr_lines, path)
    extracted = read_pdf(path)[0].text.split("\n")

    assert extracted[1] == "Operator: Quillfen Energy    Field: Orrindale    Well: ORD-105"
    fields = re.split(r" {2,}", extracted[1])
    assert fields == ["Operator: Quillfen Energy", "Field: Orrindale", "Well: ORD-105"]
    assert extracted[7] == "  Depth at end          : 1,402 m MD"
    assert extracted[9] == '  Hole section          : 17 1/2"'


def test_page_numbers_are_correct_across_a_page_break(ddr_lines: list[str], tmp_path: Path) -> None:
    assert len(ddr_lines) > LINES_PER_PAGE
    path = tmp_path / "ddr.pdf"
    write_pdf(ddr_lines, path)

    pages = read_pdf(path)

    assert [page.page_no for page in pages] == [1, 2]
    assert pages[0].text == "\n".join(ddr_lines[:LINES_PER_PAGE])
    assert pages[1].text == "\n".join(ddr_lines[LINES_PER_PAGE:])
    assert pages[1].text.split("\n")[0] == ddr_lines[LINES_PER_PAGE]


@pytest.mark.parametrize(
    "blank_positions",
    [
        pytest.param({0}, id="first-line"),
        pytest.param({5, 6, 7}, id="consecutive"),
        pytest.param({LINES_PER_PAGE - 1}, id="last-on-page-1"),
        pytest.param({LINES_PER_PAGE}, id="first-on-page-2"),
        pytest.param({LINES_PER_PAGE - 2, LINES_PER_PAGE - 1, LINES_PER_PAGE, LINES_PER_PAGE + 1}, id="span"),
        pytest.param({2 * LINES_PER_PAGE - 1}, id="last-line-of-document"),
        pytest.param({2 * LINES_PER_PAGE - 2, 2 * LINES_PER_PAGE - 1}, id="two-trailing"),
    ],
)
def test_blank_lines_survive_anywhere_on_the_page(blank_positions: set[int], tmp_path: Path) -> None:
    lines = ["" if i in blank_positions else f"  line {i:03d}  x" for i in range(2 * LINES_PER_PAGE)]
    assert _round_trip(lines, tmp_path) == "\n".join(lines)


def test_lines_of_spaces_and_trailing_spaces_survive(tmp_path: Path) -> None:
    lines = ["A  ", "   ", "", "    indented    ", "B"]
    assert _round_trip(lines, tmp_path) == "\n".join(lines)


def test_parentheses_and_backslashes_are_escaped(tmp_path: Path) -> None:
    lines = ["(a) b\\c ((nested))", "unbalanced ( here", "and ) there \\", "\\(\\)"]
    assert _round_trip(lines, tmp_path) == "\n".join(lines)


CP1252_LINE = (
    'Hole 12\N{VULGAR FRACTION ONE QUARTER}" \N{EN DASH} BHT 118 \N{DEGREE SIGN}C '
    "\N{PLUS-MINUS SIGN} 2 \N{EM DASH} \N{EURO SIGN} 1.5 M "
    "\N{LEFT DOUBLE QUOTATION MARK}quoted\N{RIGHT DOUBLE QUOTATION MARK} "
    "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
)
OUTSIDE_CP1252 = (
    "\N{LATIN CAPITAL LETTER L WITH STROKE}\N{LATIN SMALL LETTER O WITH ACUTE}d"
    "\N{LATIN SMALL LETTER Z WITH ACUTE} \N{LATIN SMALL LETTER O WITH DOUBLE ACUTE}"
)


def test_cp1252_characters_round_trip(tmp_path: Path) -> None:
    lines = [CP1252_LINE]
    assert _round_trip(lines, tmp_path) == lines[0]


def test_long_line_shrinks_font_and_round_trips(tmp_path: Path) -> None:
    lines = ["x" * 120, "  short"]
    data = render_pdf(lines)
    assert b"/F1 7.5 Tf" in data
    assert _round_trip(lines, tmp_path) == "\n".join(lines)


def test_very_long_line_widens_the_media_box(eowr_lines: list[str], tmp_path: Path) -> None:
    longest = max(len(line) for line in eowr_lines)
    assert longest > 150
    data = render_pdf(eowr_lines)
    box = PdfReader(io.BytesIO(data)).pages[0].mediabox

    assert float(box.height) == 792
    assert float(box.width) > 612
    assert float(box.width) >= 72 + longest * 0.6 * 6
    assert _round_trip(eowr_lines, tmp_path) == "\n".join(eowr_lines)


# Encoding -----------------------------------------------------------------------------------------


def test_characters_outside_cp1252_fall_back() -> None:
    assert encode_line(OUTSIDE_CP1252 + " CO\N{SUBSCRIPT TWO} \N{ALMOST EQUAL TO}") == b"?\xf3dz o CO2 ?"


def test_tabs_expand_and_control_characters_become_question_marks() -> None:
    assert encode_line("a\tb") == b"a       b"
    assert encode_line("x\x00y\x7fz") == b"x?y?z"


def test_fallback_characters_are_read_back_as_written(tmp_path: Path) -> None:
    assert _round_trip([OUTSIDE_CP1252], tmp_path) == "?\N{LATIN SMALL LETTER O WITH ACUTE}dz o"


@pytest.mark.parametrize("bad", ["a\nb", "a\rb", "a\r\nb"])
def test_line_breaks_inside_a_line_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="line break"):
        render_pdf(["ok", bad])


def test_escape_pdf_string() -> None:
    assert escape_pdf_string(b"(a)\\") == "\\(a\\)\\\\"
    assert escape_pdf_string(b"\n\xe9 ") == "\\012\\351 "


# Layout -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_chars", "font_size", "width"),
    [
        (0, 9.0, 612.0),
        (100, 9.0, 612.0),
        (101, 8.5, 612.0),
        (105, 8.5, 612.0),
        (106, 8.0, 612.0),
        (150, 6.0, 612.0),
        (151, 6.0, 616.0),
        (300, 6.0, 1152.0),
    ],
)
def test_layout_for(max_chars: int, font_size: float, width: float) -> None:
    layout = layout_for(max_chars)
    assert (layout.font_size, layout.page_width, layout.page_height) == (font_size, width, 792.0)
    assert layout.leading == round(font_size * 1.2, 2)


def test_layout_for_rejects_negative_width() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        layout_for(-1)


def test_last_baseline_stays_on_the_page() -> None:
    layout = layout_for(0)
    assert layout.first_baseline - (LINES_PER_PAGE - 1) * layout.leading > 36


# Structure ----------------------------------------------------------------------------------------


def test_each_line_is_exactly_one_tj_operation(ddr_lines: list[str]) -> None:
    streams = _content_streams(render_pdf(ddr_lines))
    per_page = [ddr_lines[:LINES_PER_PAGE], ddr_lines[LINES_PER_PAGE:]]

    assert len(streams) == 2
    for stream, lines in zip(streams, per_page, strict=True):
        ops = stream.decode("ascii").split("\n")
        assert sum(op.endswith(") Tj") for op in ops) == len(lines)
        assert sum(op.endswith(" Tm") for op in ops) == len(lines)
        assert b"TJ" not in stream
        assert ops[0] == "BT"
        assert ops[-1] == "ET"


def test_font_is_builtin_courier_with_winansi_and_widths() -> None:
    page = PdfReader(io.BytesIO(render_pdf(["x"]))).pages[0]
    resources = page["/Resources"]
    assert isinstance(resources, DictionaryObject)
    fonts = resources["/Font"]
    assert isinstance(fonts, DictionaryObject)
    font = fonts["/F1"].get_object()
    assert isinstance(font, DictionaryObject)
    widths = font["/Widths"]
    assert isinstance(widths, ArrayObject)

    assert (str(font["/Subtype"]), str(font["/BaseFont"])) == ("/Type1", "/Courier")
    assert str(font["/Encoding"]) == "/WinAnsiEncoding"
    descriptor = font["/FontDescriptor"].get_object()
    assert isinstance(descriptor, DictionaryObject)
    assert str(descriptor["/FontName"]) == "/Courier"
    bbox, flags = descriptor["/FontBBox"], descriptor["/Flags"]
    assert isinstance(bbox, ArrayObject)
    assert isinstance(flags, NumberObject)
    assert [int(value) for value in bbox] == [-23, -250, 715, 805]
    assert int(flags) & 1  # FixedPitch
    assert not any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
    first_char, last_char = font["/FirstChar"], font["/LastChar"]
    assert isinstance(first_char, NumberObject)
    assert isinstance(last_char, NumberObject)
    assert (int(first_char), int(last_char)) == (32, 255)
    assert len(widths) == 224
    assert {int(width) for width in widths} == {600}


def test_xref_offsets_point_at_their_objects(ddr_lines: list[str]) -> None:
    data = render_pdf(ddr_lines)
    startxref = int(data.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    assert data[startxref:].startswith(b"xref\n0 ")

    header, *entries = data[startxref:].split(b"trailer", 1)[0].split(b"\n")[1:]
    count = int(header.split()[1])
    entries = [entry for entry in entries if entry]
    assert len(entries) == count
    assert entries[0] == b"0000000000 65535 f "
    for number, entry in enumerate(entries[1:], start=1):
        assert len(entry) + 1 == 20
        offset = int(entry[:10])
        assert data[offset:].startswith(b"%d 0 obj\n" % number)
    assert data.endswith(b"%%EOF\n")


def test_strict_pypdf_parse_accepts_the_file(ddr_lines: list[str]) -> None:
    reader = PdfReader(io.BytesIO(render_pdf(ddr_lines)), strict=True)
    assert len(reader.pages) == 2


def test_stream_lengths_match(ddr_lines: list[str]) -> None:
    data = render_pdf(ddr_lines)
    for match in re.finditer(rb"<< /Length (\d+) >>\nstream\n", data):
        length = int(match.group(1))
        assert data[match.end() + length : match.end() + length + 10] == b"\nendstream"


# Determinism and metadata -------------------------------------------------------------------------


def test_output_is_byte_identical_for_identical_input(sample_lines: list[str], tmp_path: Path) -> None:
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    write_pdf(sample_lines, first)
    write_pdf(list(sample_lines), second)
    assert first.read_bytes() == second.read_bytes() == render_pdf(sample_lines)


def test_output_carries_no_metadata(sample_lines: list[str]) -> None:
    data = render_pdf(sample_lines)
    for token in (b"/Info", b"/CreationDate", b"/ModDate", b"/Producer", b"/Author", b"/Creator", b"/ID"):
        assert token not in data
    assert PdfReader(io.BytesIO(data)).metadata is None


def test_output_is_ascii_after_the_binary_marker_comment(sample_lines: list[str]) -> None:
    data = render_pdf(sample_lines)
    first_line, marker, rest = data.split(b"\n", 2)
    assert first_line == b"%PDF-1.4"
    assert marker.startswith(b"%")
    assert min(marker[1:]) >= 128
    rest.decode("ascii")


# Empty input --------------------------------------------------------------------------------------


def test_empty_document_has_one_page_and_no_text_layer(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    write_pdf([], path)
    assert len(PdfReader(path).pages) == 1
    with pytest.raises(NoTextLayer):
        read_pdf(path)
