"""Tests for the DOCX reader on hand-built packages that resemble word-processor output."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from wellbrief.readers import ReaderError
from wellbrief.readers.docx import MAX_XML_BYTES, read_docx

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STRICT_NS = "http://purl.oclc.org/ooxml/wordprocessingml/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _docx(tmp_path: Path, document_xml: str | bytes | None, name: str = "doc.docx") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        if document_xml is not None:
            data = document_xml.encode("utf-8") if isinstance(document_xml, str) else document_xml
            archive.writestr("word/document.xml", data)
    return path


def _document(body: str, namespace: str = W_NS) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{namespace}" xmlns:mc="{MC_NS}"><w:body>{body}</w:body></w:document>'
    )


def _p(*runs: str, props: str = "") -> str:
    return f"<w:p>{props}{''.join(runs)}</w:p>"


def _r(text: str) -> str:
    return f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>'


def _cell(*paragraphs: str) -> str:
    return f"<w:tc><w:tcPr><w:tcW w:w='2000'/></w:tcPr>{''.join(paragraphs)}</w:tc>"


WORD_LIKE_BODY = "".join(
    [
        # Tab-stop definitions in pPr must not become tab characters; runs split mid-word join up.
        _p(
            _r("Operator: Quill"),
            _r("fen Energy    Field: Orrindale"),
            props='<w:pPr><w:tabs><w:tab w:val="left" w:pos="720"/></w:tabs></w:pPr>',
        ),
        _p(_r("Code"), "<w:r><w:tab/></w:r>", _r(": STUCK_PIPE")),
        _p(_r("line one"), "<w:r><w:br/></w:r>", _r("line two"), "<w:r><w:cr/></w:r>", _r("three")),
        _p(_r("non"), "<w:r><w:noBreakHyphen/></w:r>", _r("productive")),
        # Hyperlinks, tracked insertions and simple fields are kept; deletions and moves are dropped.
        _p(
            '<w:hyperlink r:id="rId9" xmlns:r="urn:r">',
            _r("see EOWR"),
            "</w:hyperlink>",
            '<w:ins w:id="1" w:author="x">',
            _r(" (updated)"),
            "</w:ins>",
            '<w:del w:id="2" w:author="x"><w:r><w:delText>removed</w:delText></w:r></w:del>',
            '<w:moveFrom w:id="3" w:author="x">',
            _r("moved away"),
            "</w:moveFrom>",
            '<w:fldSimple w:instr=" PAGE ">',
            _r(" p1"),
            "</w:fldSimple>",
            "<w:r><w:fldChar w:fldCharType='begin'/></w:r>",
            "<w:r><w:instrText> DATE </w:instrText></w:r>",
            "<w:r><w:fldChar w:fldCharType='separate'/></w:r>",
            _r(" 2026-03-18"),
            "<w:r><w:fldChar w:fldCharType='end'/></w:r>",
        ),
        # A block-level content control wraps a paragraph.
        "<w:sdt><w:sdtPr><w:alias w:val='Remarks'/></w:sdtPr><w:sdtContent>",
        _p(_r("inside a content control")),
        "</w:sdtContent></w:sdt>",
        # A table: one line per row, cells joined with " | ", paragraphs in a cell joined by spaces.
        "<w:tbl><w:tblPr/><w:tblGrid><w:gridCol w:w='2000'/></w:tblGrid>",
        "<w:tr>",
        _cell(_p(_r("Code"))),
        _cell(_p(_r("Hours"))),
        _cell(_p(_r("Description"))),
        "</w:tr>",
        "<w:tr>",
        _cell(_p(_r("LOST_CIRCULATION"))),
        _cell(_p(_r("31.5"))),
        _cell(_p(_r("Total losses")), _p(_r("at 2,662 m"), "<w:r><w:br/></w:r>", _r("MD"))),
        "</w:tr>",
        "<w:tr>",
        _cell(_p()),
        _cell(
            "<w:tbl><w:tr>",
            _cell(_p(_r("nested a"))),
            _cell(_p(_r("nested b"))),
            "</w:tr></w:tbl>",
        ),
        _cell(_p(_r("last"))),
        "</w:tr>",
        # Repeating-section content controls wrap whole rows, and single cells can be wrapped too.
        "<w:sdt><w:sdtPr/><w:sdtContent><w:tr>",
        "<w:sdt><w:sdtContent>",
        _cell(_p(_r("wrapped cell"))),
        "</w:sdtContent></w:sdt>",
        _cell(_p(_r("plain cell"))),
        "</w:tr></w:sdtContent></w:sdt>",
        "</w:tbl>",
        # A text box: the choice branch is read once, after its anchor paragraph; the fallback is skipped.
        _p(
            _r("anchor"),
            "<w:r><mc:AlternateContent><mc:Choice Requires='wps'><w:drawing><w:txbxContent>",
            _p(_r("text box line")),
            "</w:txbxContent></w:drawing></mc:Choice><mc:Fallback><w:pict><w:txbxContent>",
            _p(_r("text box line")),
            "</w:txbxContent></w:pict></mc:Fallback></mc:AlternateContent></w:r>",
        ),
        _p(),
        _p(_r("after an empty paragraph")),
        "<w:sectPr><w:pgSz w:w='12240' w:h='15840'/></w:sectPr>",
    ]
)

WORD_LIKE_LINES = [
    "Operator: Quillfen Energy    Field: Orrindale",
    "Code\t: STUCK_PIPE",
    "line one\nline two\nthree",
    "non-productive",
    "see EOWR (updated) p1 2026-03-18",
    "inside a content control",
    "Code | Hours | Description",
    "LOST_CIRCULATION | 31.5 | Total losses at 2,662 m MD",
    " | nested a | nested b | last",
    "wrapped cell | plain cell",
    "anchor",
    "text box line",
    "",
    "after an empty paragraph",
]
WORD_LIKE_TEXT = "\n".join(WORD_LIKE_LINES)


def test_word_like_document_is_read_in_document_order(tmp_path: Path) -> None:
    assert read_docx(_docx(tmp_path, _document(WORD_LIKE_BODY))) == WORD_LIKE_TEXT


def test_strict_namespace_is_accepted(tmp_path: Path) -> None:
    body = _p(_r("strict")) + "<w:tbl><w:tr>" + _cell(_p(_r("a"))) + _cell(_p(_r("b"))) + "</w:tr></w:tbl>"
    assert read_docx(_docx(tmp_path, _document(body, STRICT_NS))) == "strict\na | b"


def test_document_without_body_reads_as_empty(tmp_path: Path) -> None:
    xml = f'<w:document xmlns:w="{W_NS}"/>'
    assert read_docx(_docx(tmp_path, xml)) == ""


def test_utf16_document_part_is_read(tmp_path: Path) -> None:
    xml = _document(_p(_r("utf-16 text"))).replace('encoding="UTF-8"', 'encoding="UTF-16"')
    assert read_docx(_docx(tmp_path, xml.encode("utf-16"))) == "utf-16 text"


# Errors -------------------------------------------------------------------------------------------


def test_not_a_zip_file(tmp_path: Path) -> None:
    path = tmp_path / "fake.docx"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(ReaderError, match="zip"):
        read_docx(path)


def test_truncated_zip_file(tmp_path: Path) -> None:
    good = _docx(tmp_path, _document(_p(_r("x"))))
    path = tmp_path / "truncated.docx"
    path.write_bytes(good.read_bytes()[:-30])
    with pytest.raises(ReaderError):
        read_docx(path)


def test_missing_document_part(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match=r"no word/document\.xml"):
        read_docx(_docx(tmp_path, None))


def test_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_docx(tmp_path / "absent.docx")


def test_malformed_xml(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="well-formed"):
        read_docx(_docx(tmp_path, "<w:document><w:body>"))


def test_wrong_root_element(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="not a WordprocessingML document"):
        read_docx(_docx(tmp_path, f'<w:styles xmlns:w="{W_NS}"/>'))


def test_unknown_namespace(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="not a WordprocessingML document"):
        read_docx(_docx(tmp_path, '<document xmlns="urn:other"><body/></document>'))


ENTITY_EXPANSION = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE w:document [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
    f'<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>'
)
EXTERNAL_ENTITY = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE w:document [<!ENTITY secret SYSTEM "file:///etc/hostname">]>'
    f'<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>&secret;</w:t></w:r></w:p></w:body></w:document>'
)


@pytest.mark.parametrize("xml", [ENTITY_EXPANSION, EXTERNAL_ENTITY], ids=["expansion", "external"])
def test_documents_with_a_dtd_are_rejected(xml: str, tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="DTD"):
        read_docx(_docx(tmp_path, xml))


def test_utf16_dtd_is_rejected(tmp_path: Path) -> None:
    xml = ENTITY_EXPANSION.replace('<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-16"?>')
    with pytest.raises(ReaderError, match="DTD"):
        read_docx(_docx(tmp_path, xml.encode("utf-16")))


def test_highly_compressible_part_over_the_cap_is_rejected(tmp_path: Path) -> None:
    padding = "<w:p/>" * 200_000  # about 1.2 MB of XML that deflates to a few kilobytes
    path = _docx(tmp_path, _document(padding))
    assert path.stat().st_size < 50_000
    with pytest.raises(ReaderError, match="byte limit"):
        read_docx(path, max_xml_bytes=1_000_000)
    assert read_docx(path) == "\n" * 199_999


def test_default_part_cap_is_16_mib() -> None:
    assert MAX_XML_BYTES == 16 * 1024 * 1024


def test_block_over_the_element_cap_is_rejected(tmp_path: Path) -> None:
    body = "<w:p><w:r>" + "<w:tab/>" * 1_000 + "</w:r></w:p>"
    path = _docx(tmp_path, _document(body))
    with pytest.raises(ReaderError, match="block of more than 999 XML elements"):
        read_docx(path, max_block_elements=999)
    assert read_docx(path, max_block_elements=1_002) == "\t" * 1_000


def test_element_cap_applies_per_block_not_per_document(tmp_path: Path) -> None:
    # 3,000 paragraphs of 3 elements each: far more elements than the cap in total, but each block is
    # dropped once read, so the tree never holds more than one.
    body = _p("<w:r><w:t>x</w:t></w:r>") * 3_000 + "<w:tbl><w:tr>" + _cell(_p(_r("a"))) + "</w:tr></w:tbl>"
    text = read_docx(_docx(tmp_path, _document(body)), max_block_elements=12)
    assert text == "\n".join(["x"] * 3_000 + ["a"])


def test_element_cap_applies_outside_the_body(tmp_path: Path) -> None:
    xml = (
        f'<w:document xmlns:w="{W_NS}"><w:background>'
        + "<w:x/>" * 50
        + "</w:background><w:body>"
        + _p(_r("text"))
        + "</w:body></w:document>"
    )
    path = _docx(tmp_path, xml)
    with pytest.raises(ReaderError, match="block of more than 40 XML elements"):
        read_docx(path, max_block_elements=40)
    assert read_docx(path, max_block_elements=60) == "text"


def test_deeply_nested_markup_is_reported_not_crashed(tmp_path: Path) -> None:
    depth = 5_000
    body = "<w:sdt><w:sdtContent>" * depth + _p(_r("deep")) + "</w:sdtContent></w:sdt>" * depth
    with pytest.raises(ReaderError, match="nested too deeply"):
        read_docx(_docx(tmp_path, _document(body)))
