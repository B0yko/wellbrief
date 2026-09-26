"""Minimal, dependency-free PDF writer for synthetic corpus documents.

The corpus generator uses this module to render plain-text reports as PDF files, so that
ingestion exercises the PDF reader on documents whose exact text is known. The output is built
for a lossless text round trip through ``pypdf`` and for byte-identical output:

* Every page uses the standard Type1 font Courier with ``/WinAnsiEncoding``. No font program is
  embedded; the font dictionary carries ``/FirstChar``, ``/LastChar``, a ``/Widths`` array (every
  Courier glyph advances 600/1000 em) and a ``/FontDescriptor`` with the metrics of Adobe's
  ``Courier.afm``, so every reader gets the same advance widths.
* Every source line is exactly one ``Tj`` operation positioned with an absolute ``Tm`` matrix.
  Runs of spaces and leading indentation are part of the shown string, never produced by glyph
  positioning, so text extraction returns them unchanged.
* A blank line is shown as a single line-feed byte (code 10, which has no glyph and no width in
  WinAnsiEncoding). ``pypdf`` emits a newline for a line move only when its output does not
  already end with one, so consecutive line moves would otherwise collapse into one newline.
  The last line of a page is the exception: when it is blank it is shown as an empty string,
  because the line move to it already produces the newline that separates it.
* Pages hold at most :data:`LINES_PER_PAGE` lines. The page is US Letter portrait; the font size
  is the largest of 9, 8.5, ... 6 pt at which the longest line fits between the margins, and when
  a line is too long even at 6 pt the page is widened for that document.
* There is no ``/Info`` dictionary, no ``/ID`` and no timestamp, and content streams are stored
  uncompressed, so identical input gives identical bytes regardless of the zlib build.

Text is encoded in cp1252, which is what WinAnsiEncoding denotes. A character outside cp1252 is
replaced by the cp1252 characters of its compatibility decomposition without combining marks (an
``o`` with a double acute accent becomes ``o``, a subscript two becomes ``2``) or, failing that,
by ``"?"``. Tabs are expanded to spaces at 8-column stops and other control characters become
``"?"``. The round trip is therefore exact for any line made of printable cp1252 characters.
"""

from __future__ import annotations

import math
import os
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LINES_PER_PAGE",
    "PageLayout",
    "encode_line",
    "escape_pdf_string",
    "layout_for",
    "render_pdf",
    "write_pdf",
]

LINES_PER_PAGE = 64
"""Maximum number of source lines on one page."""

PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0
MARGIN_X_PT = 36.0
MARGIN_TOP_PT = 40.0
MAX_FONT_SIZE_PT = 9.0
MIN_FONT_SIZE_PT = 6.0
FONT_SIZE_STEP_PT = 0.5
LEADING_FACTOR = 1.2
COURIER_ADVANCE = 600
"""Advance width of every Courier glyph in thousandths of an em (Adobe Courier AFM)."""

_COURIER_DESCRIPTOR = (
    "<< /Type /FontDescriptor /FontName /Courier /Flags 33 /FontBBox [-23 -250 715 805] "
    "/ItalicAngle 0 /Ascent 629 /Descent -157 /CapHeight 562 /XHeight 426 /StemV 51 >>"
)
"""Metrics from Adobe's Courier.afm; flags 33 are FixedPitch and Nonsymbolic."""
_FIRST_CHAR = 32
_LAST_CHAR = 255
_TAB_SIZE = 8
_LINE_FEED = b"\n"
_HEADER = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"


@dataclass(frozen=True, slots=True)
class PageLayout:
    """Geometry shared by every page of one document.

    Attributes:
        font_size: Courier size in points.
        leading: Distance between consecutive baselines in points.
        page_width: MediaBox width in points (612 unless a line needed more room).
        page_height: MediaBox height in points.
    """

    font_size: float
    leading: float
    page_width: float
    page_height: float

    @property
    def first_baseline(self) -> float:
        """Baseline of the first line on a page, measured from the bottom edge."""
        return self.page_height - MARGIN_TOP_PT - self.font_size


def layout_for(max_chars: int) -> PageLayout:
    """Choose the font size and page width for a document whose longest line has ``max_chars``.

    The font size is the largest multiple of :data:`FONT_SIZE_STEP_PT` between
    :data:`MAX_FONT_SIZE_PT` and :data:`MIN_FONT_SIZE_PT` at which the line fits on a Letter
    page between the side margins. If the line does not fit at the minimum size, the page is
    widened just enough (rounded up to a whole point) to hold it.
    """
    if max_chars < 0:
        raise ValueError("max_chars must not be negative")
    # Work in thousandths of a point per half-point step so the fit test is exact.
    usable_milli = round((PAGE_WIDTH_PT - 2 * MARGIN_X_PT) * 1000)
    steps = round((MAX_FONT_SIZE_PT - MIN_FONT_SIZE_PT) / FONT_SIZE_STEP_PT)
    for step in range(steps + 1):
        size = MAX_FONT_SIZE_PT - step * FONT_SIZE_STEP_PT
        if max_chars * COURIER_ADVANCE * size <= usable_milli:
            return PageLayout(size, round(size * LEADING_FACTOR, 2), PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
    size = MIN_FONT_SIZE_PT
    width = float(math.ceil((2 * MARGIN_X_PT * 1000 + max_chars * COURIER_ADVANCE * size) / 1000))
    return PageLayout(size, round(size * LEADING_FACTOR, 2), width, PAGE_HEIGHT_PT)


def encode_line(line: str) -> bytes:
    """Encode one line of text as WinAnsi (cp1252) bytes, applying the documented fallbacks.

    Raises:
        ValueError: If the line contains a line break; each line must be passed separately.
    """
    if "\n" in line or "\r" in line:
        raise ValueError("a line passed to the PDF writer must not contain a line break")
    out = bytearray()
    for ch in line.expandtabs(_TAB_SIZE):
        out += _encode_char(ch)
    return bytes(out)


def escape_pdf_string(data: bytes) -> str:
    """Return ``data`` as the body of a PDF literal string (without the enclosing parentheses).

    Parentheses and backslashes are escaped with a backslash. Bytes outside printable ASCII are
    written as three-digit octal escapes, which keeps the file 7-bit clean and stops readers from
    normalising a line-feed byte inside a string.
    """
    parts: list[str] = []
    for byte in data:
        if byte in (0x28, 0x29, 0x5C):
            parts.append("\\" + chr(byte))
        elif 0x20 <= byte < 0x7F:
            parts.append(chr(byte))
        else:
            parts.append(f"\\{byte:03o}")
    return "".join(parts)


def render_pdf(lines: Sequence[str]) -> bytes:
    """Render ``lines`` as a PDF document and return its bytes.

    An empty sequence yields a single page without text.

    Raises:
        TypeError: If ``lines`` is a single string instead of a sequence of lines.
        ValueError: If a line contains a line break.
    """
    if isinstance(lines, str):
        raise TypeError("lines must be a sequence of strings, not a single string")
    encoded = [encode_line(line) for line in lines]
    layout = layout_for(max((len(data) for data in encoded), default=0))
    pages = [encoded[i : i + LINES_PER_PAGE] for i in range(0, len(encoded), LINES_PER_PAGE)] or [[]]

    objects: list[bytes] = []
    # Objects: 1 catalog, 2 page tree, 3 font, 4 font descriptor, then a page and its contents.
    page_ids = [5 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))
    objects.append(_font_dictionary(descriptor_id=4))
    objects.append(_COURIER_DESCRIPTOR.encode("ascii"))
    media_box = f"[0 0 {_num(layout.page_width)} {_num(layout.page_height)}]"
    for page_id, page_lines in zip(page_ids, pages, strict=True):
        stream = _content_stream(page_lines, layout)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox {media_box} "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    return _assemble(objects)


def write_pdf(lines: Sequence[str], path: str | os.PathLike[str]) -> None:
    """Render ``lines`` with :func:`render_pdf` and write the bytes to ``path``.

    Raises:
        TypeError: If ``lines`` is a single string.
        ValueError: If a line contains a line break.
    """
    Path(path).write_bytes(render_pdf(lines))


def _is_control(ch: str) -> bool:
    return ch < " " or ch == "\x7f"


def _encode_char(ch: str) -> bytes:
    if _is_control(ch):
        return b"?"
    try:
        return ch.encode("cp1252")
    except UnicodeEncodeError:
        pass
    base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
    if base and not any(_is_control(c) for c in base):
        try:
            return base.encode("cp1252")
        except UnicodeEncodeError:
            pass
    return b"?"


def _font_dictionary(descriptor_id: int) -> bytes:
    widths = " ".join([str(COURIER_ADVANCE)] * (_LAST_CHAR - _FIRST_CHAR + 1))
    return (
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding "
        f"/FirstChar {_FIRST_CHAR} /LastChar {_LAST_CHAR} /Widths [{widths}] "
        f"/FontDescriptor {descriptor_id} 0 R >>"
    ).encode("ascii")


def _content_stream(page_lines: Sequence[bytes], layout: PageLayout) -> bytes:
    ops = ["BT", f"/F1 {_num(layout.font_size)} Tf"]
    last = len(page_lines) - 1
    for index, data in enumerate(page_lines):
        if not data and index != last:
            data = _LINE_FEED
        y = layout.first_baseline - index * layout.leading
        ops.append(f"1 0 0 1 {_num(MARGIN_X_PT)} {_num(y)} Tm")
        ops.append(f"({escape_pdf_string(data)}) Tj")
    ops.append("ET")
    return "\n".join(ops).encode("ascii")


def _assemble(objects: Sequence[bytes]) -> bytes:
    out = bytearray(_HEADER)
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(out)


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text
