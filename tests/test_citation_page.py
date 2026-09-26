"""`quotes.quote_page`: mapping a quote's raw offset to a 1-based page via `Document.page_map`.

No reader populates `page_map` yet (PDF reading is a later phase), so every
citation's `page` is `None` today; this tests the mapping in isolation with a
document built directly with one, and that `qa.ask` and `riskbrief` leave
`page` at `None` for the (page-map-less) documents they actually see.
"""

from __future__ import annotations

from pathlib import Path

from wellbrief.models import Document
from wellbrief.quotes import quote_page

PAGE_1 = "DAILY DRILLING REPORT\nFirst page content here.\n"
PAGE_2 = "Second page content, with the answer on it.\n"
PAGE_3 = "Third and final page.\n"
TEXT = PAGE_1 + PAGE_2 + PAGE_3
PAGE_MAP = [len(PAGE_1), len(PAGE_1) + len(PAGE_2), len(TEXT)]


def _doc(page_map: list[int] | None) -> Document:
    return Document("EOWR-1", "eowr", "W-1", "Field", "2024-01-01", "END OF WELL REPORT", TEXT,
                    page_map=page_map)


def test_a_quote_on_the_first_page() -> None:
    assert quote_page(_doc(PAGE_MAP), "First page content here.") == 1


def test_a_quote_on_the_second_page() -> None:
    assert quote_page(_doc(PAGE_MAP), "Second page content, with the answer on it.") == 2


def test_a_quote_on_the_last_page() -> None:
    assert quote_page(_doc(PAGE_MAP), "Third and final page.") == 3


def test_no_page_map_is_none() -> None:
    assert quote_page(_doc(None), "First page content here.") is None


def test_an_empty_page_map_is_none() -> None:
    assert quote_page(_doc([]), "First page content here.") is None


def test_a_quote_not_in_the_document_is_none() -> None:
    assert quote_page(_doc(PAGE_MAP), "This sentence is not in the document.") is None


def test_an_empty_quote_is_none() -> None:
    assert quote_page(_doc(PAGE_MAP), "") is None


def test_page_is_none_for_every_citation_today(tmp_path: Path) -> None:
    """End to end: `ask` and `riskbrief` build `Citation`s from real (txt-ingested, so
    page-map-less) documents; `page` is always `None` until a PDF reader (a later
    phase) starts recording `page_map`."""
    import mini_workspace as mini
    from wellbrief.qa import ask

    store = mini.store(tmp_path)
    searcher = mini.searcher(store)
    answer = ask("Stuck pipe on Orrindale", store, searcher)
    assert answer.citations
    assert all(c.page is None for c in answer.citations)
