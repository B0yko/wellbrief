"""Minimal, dependency-free DOCX writer for synthetic corpus documents.

The package holds only the three parts a WordprocessingML consumer needs: ``[Content_Types].xml``,
``_rels/.rels`` and ``word/document.xml``. There is no ``docProps`` part, so the file carries no
author, application or timestamp metadata.

Each source line becomes one paragraph. Text runs use ``xml:space="preserve"`` so that leading
and repeated spaces survive, and a monospace run font keeps column layouts readable when the file
is opened in a word processor. A tab character becomes a ``<w:tab/>`` element, which the reader
maps back to a tab. Characters that XML 1.0 cannot represent are replaced by U+FFFD.

The zip container is deterministic: entries are written in a fixed order with a fixed
``date_time`` of 1980-01-01 00:00:00 (the earliest a zip header can hold), fixed permissions, a
fixed creator system and ``ZIP_STORED`` (no compression), so the bytes do not depend on the
platform or the zlib build.
"""

from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

__all__ = ["DOCX_ENTRY_ORDER", "ZIP_DATE_TIME", "render_docx", "write_docx"]

ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
"""Timestamp stored for every zip entry."""

DOCX_ENTRY_ORDER = ("[Content_Types].xml", "_rels/.rels", "word/document.xml")
"""Zip entries in the order they are written."""

MONOSPACE_FONT = "Courier New"

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_CONTENT_TYPES = (
    _XML_DECL
    + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    + '<Default Extension="rels" '
    + 'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    + '<Default Extension="xml" ContentType="application/xml"/>'
    + '<Override PartName="/word/document.xml" ContentType="application/'
    + 'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    + "</Types>"
)
_PACKAGE_RELS = (
    _XML_DECL
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    + '<Relationship Id="rId1" '
    + 'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    + 'Target="word/document.xml"/>'
    + "</Relationships>"
)
_RUN_PROPERTIES = (
    f'<w:rPr><w:rFonts w:ascii="{MONOSPACE_FONT}" w:hAnsi="{MONOSPACE_FONT}" '
    f'w:cs="{MONOSPACE_FONT}"/></w:rPr>'
)
# Letter portrait with 0.75 in margins, in twentieths of a point.
_SECTION_PROPERTIES = (
    '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
    '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
    'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
)
_XML_INVALID = re.compile("[^\t\n\r\x20-\U0000d7ff\U0000e000-\U0000fffd\U00010000-\U0010ffff]")
_EXTERNAL_ATTR = 0o100644 << 16
_CREATE_SYSTEM_UNIX = 3


def render_docx(lines: Sequence[str]) -> bytes:
    """Render ``lines`` as a DOCX package and return its bytes.

    Raises:
        TypeError: If ``lines`` is a single string instead of a sequence of lines.
        ValueError: If a line contains a line break; each line must be passed separately.
    """
    if isinstance(lines, str):
        raise TypeError("lines must be a sequence of strings, not a single string")
    parts = {
        "[Content_Types].xml": _CONTENT_TYPES,
        "_rels/.rels": _PACKAGE_RELS,
        "word/document.xml": _document_xml(lines),
    }
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in DOCX_ENTRY_ORDER:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = _EXTERNAL_ATTR
            info.create_system = _CREATE_SYSTEM_UNIX
            archive.writestr(info, parts[name].encode("utf-8"))
    return buffer.getvalue()


def write_docx(lines: Sequence[str], path: str | os.PathLike[str]) -> None:
    """Render ``lines`` with :func:`render_docx` and write the bytes to ``path``.

    Raises:
        TypeError: If ``lines`` is a single string.
        ValueError: If a line contains a line break.
    """
    Path(path).write_bytes(render_docx(lines))


def _document_xml(lines: Sequence[str]) -> str:
    body = "".join(_paragraph(line) for line in lines)
    return (
        f'{_XML_DECL}<w:document xmlns:w="{_W_NS}"><w:body>{body}{_SECTION_PROPERTIES}</w:body></w:document>'
    )


def _paragraph(line: str) -> str:
    if "\n" in line or "\r" in line:
        raise ValueError("a line passed to the DOCX writer must not contain a line break")
    if not line:
        return "<w:p/>"
    content: list[str] = []
    for index, piece in enumerate(_XML_INVALID.sub("\N{REPLACEMENT CHARACTER}", line).split("\t")):
        if index:
            content.append("<w:tab/>")
        if piece:
            content.append(f'<w:t xml:space="preserve">{escape(piece)}</w:t>')
    return f"<w:p><w:r>{_RUN_PROPERTIES}{''.join(content)}</w:r></w:p>"
