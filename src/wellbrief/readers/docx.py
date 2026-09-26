"""DOCX reader built on the standard library (``zipfile`` and ``xml.etree``).

The reader extracts the main document part, ``word/document.xml``, and returns its body as text:

* body paragraphs and tables in document order, one paragraph per line;
* paragraph text is the concatenation of its ``w:t`` runs, with ``w:tab`` and ``w:ptab`` as a tab
  and ``w:br`` and ``w:cr`` as a newline; tab-stop definitions and run properties are ignored;
* each table row becomes one line whose cell texts are joined with ``" | "``; paragraphs inside a
  cell (and nested tables) are joined with spaces so that the row stays on one line;
* content inside content controls (``w:sdt``), custom XML, hyperlinks, fields and tracked
  insertions is kept; tracked deletions and moved-from text are dropped;
* text-box paragraphs are emitted as separate lines after the paragraph that anchors them, and the
  fallback branch of a markup-compatibility block is skipped so text is not duplicated.

Headers, footers, footnotes, endnotes and comments live in other parts and are not read.

Both the transitional and the strict WordprocessingML namespaces are accepted.

Untrusted input is handled defensively:

* The main part is decompressed through a bounded read of at most ``max_xml_bytes`` plus one byte,
  whatever size the zip directory declares.
* Office Open XML parts do not use a DTD, so a part with a ``<!DOCTYPE`` declaration is rejected
  before parsing. This rules out entity-expansion attacks, and ``xml.etree`` does not fetch
  external entities in any case.
* The part is parsed as a stream. Each top-level block of the body (a paragraph, a table, a
  block-level content control) is turned into text as soon as its end tag is read and is then
  dropped, so the element tree holds one block at a time. A block with more than
  ``max_block_elements`` elements is rejected.

Peak memory is therefore bounded by the size of the ``.docx`` file, plus ``max_xml_bytes`` for the
decompressed part, plus the element tree of one block (``max_block_elements`` elements of roughly
90 to 250 bytes each on CPython 3.12, so about 45 to 125 MB at the default cap), plus the extracted
text and one list entry per output line.
"""

from __future__ import annotations

import io
import os
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from . import ReaderError

__all__ = ["DOCUMENT_PART", "MAX_BLOCK_ELEMENTS", "MAX_XML_BYTES", "read_docx"]

DOCUMENT_PART = "word/document.xml"
MAX_XML_BYTES = 16 * 1024 * 1024
"""Default cap on the uncompressed size of the main document part."""
MAX_BLOCK_ELEMENTS = 500_000
"""Default cap on the number of XML elements in one top-level block of the body."""

_WORD_NAMESPACES = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
)
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_DTD_MARKERS = (
    b"<!DOCTYPE",
    "<!DOCTYPE".encode("utf-16-le"),
    "<!DOCTYPE".encode("utf-16-be"),
)
_ZIP_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    NotImplementedError,
    RuntimeError,
    EOFError,
    OSError,
    ValueError,
    zlib.error,
)


def read_docx(
    path: str | os.PathLike[str],
    *,
    max_xml_bytes: int = MAX_XML_BYTES,
    max_block_elements: int = MAX_BLOCK_ELEMENTS,
) -> str:
    """Return the body text of a DOCX file, one paragraph or table row per line.

    Args:
        path: The ``.docx`` file.
        max_xml_bytes: Largest accepted uncompressed size of ``word/document.xml``.
        max_block_elements: Largest accepted number of XML elements in one top-level block of the
            body (a paragraph, a table or a block-level content control), and in any other subtree
            of the document element.

    Raises:
        ReaderError: If the file is not a zip archive, has no ``word/document.xml``, the part is
            larger than ``max_xml_bytes``, contains a DTD, is not well-formed XML, is not a
            WordprocessingML document, or has a block larger than ``max_block_elements``.
        OSError: If the file cannot be read.
    """
    xml = _read_document_part(Path(path).read_bytes(), max_xml_bytes)
    if any(marker in xml for marker in _DTD_MARKERS):
        raise ReaderError(f"{DOCUMENT_PART} contains a DTD, which Office Open XML does not use")
    try:
        return "\n".join(_body_lines(xml, max_block_elements))
    except ET.ParseError as exc:
        raise ReaderError(f"{DOCUMENT_PART} is not well-formed XML: {exc}") from exc
    except RecursionError as exc:
        raise ReaderError(f"{DOCUMENT_PART} is nested too deeply to read") from exc


def _read_document_part(data: bytes, max_xml_bytes: int) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            try:
                info = archive.getinfo(DOCUMENT_PART)
            except KeyError:
                raise ReaderError(f"DOCX has no {DOCUMENT_PART} part") from None
            with archive.open(info) as stream:
                xml = stream.read(max_xml_bytes + 1)
    except ReaderError:
        raise
    except _ZIP_ERRORS as exc:
        raise ReaderError(f"file is not a readable DOCX (zip) archive: {exc}") from exc
    if len(xml) > max_xml_bytes:
        raise ReaderError(f"{DOCUMENT_PART} is larger than the {max_xml_bytes} byte limit")
    return xml


def _body_lines(xml: bytes, max_block_elements: int) -> list[str]:
    """Stream-parse the part and return the lines of the body, one block at a time.

    ``open_elements`` is the path from the document element to the element being parsed. When a
    direct child of ``w:body`` ends, its lines are collected and it is removed from the tree; any
    other direct child of the document element is removed when it ends. The tree therefore never
    holds more than the block being parsed, whose size is checked against ``max_block_elements``.
    """
    events = ET.iterparse(io.BytesIO(xml), events=("start", "end"))
    _, root = next(events)
    namespace = root.tag[1:].partition("}")[0] if root.tag.startswith("{") else ""
    if namespace not in _WORD_NAMESPACES or root.tag != f"{{{namespace}}}document":
        raise ReaderError(f"{DOCUMENT_PART} is not a WordprocessingML document")
    walker = _Walker(namespace)
    lines: list[str] = []
    open_elements = [root]
    held = 0
    for event, element in events:
        if event == "start":
            open_elements.append(element)
            if len(open_elements) > 2 or element.tag != walker.body:
                held += 1
                if held > max_block_elements:
                    raise ReaderError(
                        f"{DOCUMENT_PART} has a block of more than {max_block_elements} XML elements"
                    )
            continue
        open_elements.pop()
        parent_depth = len(open_elements)
        if parent_depth == 2 and open_elements[1].tag == walker.body:
            lines.extend(walker.block_lines_of(element))
            open_elements[1].remove(element)
            held = 0
        elif parent_depth == 1:
            root.remove(element)
            held = 0
    return lines


class _Walker:
    """Collects text from a WordprocessingML body for one namespace."""

    def __init__(self, namespace: str) -> None:
        w = f"{{{namespace}}}"
        self.body = f"{w}body"
        self.paragraph = f"{w}p"
        self.table = f"{w}tbl"
        self.row = f"{w}tr"
        self.cell = f"{w}tc"
        self.text = f"{w}t"
        self.text_box = f"{w}txbxContent"
        self.containers = {f"{w}sdt", f"{w}sdtContent", f"{w}customXml"}
        self.tabs = {f"{w}tab", f"{w}ptab"}
        self.breaks = {f"{w}br", f"{w}cr"}
        self.hyphen = f"{w}noBreakHyphen"
        self.skipped = {
            f"{w}pPr",
            f"{w}rPr",
            f"{w}sdtPr",
            f"{w}sdtEndPr",
            f"{w}del",
            f"{w}moveFrom",
            _MC_FALLBACK,
        }

    def block_lines(self, container: ET.Element) -> list[str]:
        """Return the lines of the paragraphs and tables directly inside ``container``."""
        lines: list[str] = []
        for child in container:
            lines.extend(self.block_lines_of(child))
        return lines

    def block_lines_of(self, element: ET.Element) -> list[str]:
        """Return the lines of one block-level element; elements without text give no lines."""
        if element.tag == self.paragraph:
            return self.paragraph_lines(element)
        if element.tag == self.table:
            return self.table_lines(element)
        if element.tag in self.containers:
            return self.block_lines(element)
        return []

    def paragraph_lines(self, paragraph: ET.Element) -> list[str]:
        """Return the paragraph text followed by the lines of any text boxes it anchors."""
        parts: list[str] = []
        text_box_lines: list[str] = []
        self._collect_inline(paragraph, parts, text_box_lines)
        return ["".join(parts), *text_box_lines]

    def table_lines(self, table: ET.Element) -> list[str]:
        """Return one line per table row, cells joined with ``" | "``."""
        lines: list[str] = []
        for row in self._children(table, self.row):
            cells = [self._cell_text(cell) for cell in self._children(row, self.cell)]
            lines.append(" | ".join(cells))
        return lines

    def _cell_text(self, cell: ET.Element) -> str:
        return " ".join(line.replace("\n", " ") for line in self.block_lines(cell))

    def _children(self, parent: ET.Element, tag: str) -> Iterator[ET.Element]:
        for child in parent:
            if child.tag == tag:
                yield child
            elif child.tag in self.containers:
                yield from self._children(child, tag)

    def _collect_inline(self, element: ET.Element, parts: list[str], text_box_lines: list[str]) -> None:
        for child in element:
            tag = child.tag
            if tag == self.text:
                parts.append(child.text or "")
            elif tag in self.tabs:
                parts.append("\t")
            elif tag in self.breaks:
                parts.append("\n")
            elif tag == self.hyphen:
                parts.append("-")
            elif tag == self.text_box:
                text_box_lines.extend(self.block_lines(child))
            elif tag not in self.skipped:
                self._collect_inline(child, parts, text_box_lines)
