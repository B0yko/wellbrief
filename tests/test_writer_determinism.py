"""Golden digests for the corpus writers.

The corpus manifest hash covers the generated PDF and DOCX files, so the writers must produce the
same bytes on every platform and Python build. These digests were recorded once; a change means
the on-disk format changed and every corpus hash changes with it, which must be deliberate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import pytest

from wellbrief.corpus.docxwriter import render_docx
from wellbrief.corpus.pdfwriter import render_pdf

PDF_SHA256 = {
    "ddr": "41e4c0a877cdb1dacbc089df020347abd67ee9409d2d8573c4bcc41e46da6887",
    "eowr": "e99cdad99c2a27eb0be7ed3fd23d7a4a80e9805ae03b6444e9069a09a0a39f39",
    "incident": "c528016ba11ef7547479dae033e8c341968638d40f18a6b55626f104601d961b",
}
DOCX_SHA256 = {
    "ddr": "de1bafc818597b25192d8023825db737c50b698c577e1c5d08daba27db738016",
    "eowr": "b55f0094cf4800f8b43e5cef60dce2f0c65f27e4b239ffecdf0c0f910f02d388",
    "incident": "5b21388dd956d52ee7f5c3ffdc4c20b4fa58cd1ddad8b428bd23c23469746d7d",
}


def test_pdf_bytes_match_the_recorded_digest(named_sample: tuple[str, list[str]]) -> None:
    name, lines = named_sample
    assert hashlib.sha256(render_pdf(lines)).hexdigest() == PDF_SHA256[name]


def test_docx_bytes_match_the_recorded_digest(named_sample: tuple[str, list[str]]) -> None:
    name, lines = named_sample
    assert hashlib.sha256(render_docx(lines)).hexdigest() == DOCX_SHA256[name]


@pytest.mark.parametrize("render", [render_pdf, render_docx], ids=["pdf", "docx"])
def test_a_single_string_is_rejected(render: Callable[[Sequence[str]], bytes]) -> None:
    with pytest.raises(TypeError, match="not a single string"):
        render("DAILY DRILLING REPORT")
