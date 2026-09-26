"""Per-field retrieval: a named field searches only its own index; otherwise every
field's ranking is fused with RRF."""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief.models import Document
from wellbrief.search import Searcher
from wellbrief.store import Store
from wellbrief.workspace import build_field_indexes


def _doc(doc_id: str, well: str, field: str, text: str) -> Document:
    return Document(doc_id, "ddr", well, field, "2024-01-01", "DAILY DRILLING REPORT", text)


@pytest.fixture
def two_field_store(tmp_path: Path) -> Store:
    """Two fields, each with a document only its own vocabulary matches, and neither
    text touching an NPT code synonym (so a question naming no field can still search
    both indexes without abstaining on an unknown code)."""
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        _doc("DDR-NHV-1", "NHV-101", "Northaven", "Calibrated the seismic sensor array on location."),
        _doc("DDR-SMR-1", "SMR-201", "Southmoor", "Painted the accommodation module on the rig floor."),
    ])
    return store


@pytest.fixture
def searcher(two_field_store: Store) -> Searcher:
    return Searcher(two_field_store, build_field_indexes(two_field_store), _embedder())


def _embedder():
    from wellbrief.embed import get_embedder
    return get_embedder("offline")


# ---------------------------------------------------------------------------
# A named field searches only that field's index
# ---------------------------------------------------------------------------


def test_naming_a_field_only_calls_that_fields_ranking(
        two_field_store: Store, searcher: Searcher, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    original = Searcher._field_ranking

    def spy(self: Searcher, field_name: str, *a: object, **kw: object) -> list:
        called.append(field_name)
        return original(self, field_name, *a, **kw)

    monkeypatch.setattr(Searcher, "_field_ranking", spy)
    hits, plan = searcher.search("seismic sensor array on Northaven")
    assert plan.fields == ["Northaven"]
    assert called == ["Northaven"]
    assert {h.doc_id for h in hits} == {"DDR-NHV-1"}


def test_a_field_named_question_never_returns_another_fields_document(searcher: Searcher) -> None:
    hits, _ = searcher.search("What happened on Southmoor?")
    assert hits and all(h.document.field_name == "Southmoor" for h in hits)


# ---------------------------------------------------------------------------
# No field named: every field is searched and the results are fused
# ---------------------------------------------------------------------------


def test_no_field_named_calls_every_fields_ranking(
        two_field_store: Store, searcher: Searcher, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    original = Searcher._field_ranking

    def spy(self: Searcher, field_name: str, *a: object, **kw: object) -> list:
        called.append(field_name)
        return original(self, field_name, *a, **kw)

    monkeypatch.setattr(Searcher, "_field_ranking", spy)
    hits, plan = searcher.search("What happened on NHV-101 and SMR-201?")
    assert plan.fields == []
    assert set(called) == {"Northaven", "Southmoor"}
    assert {h.doc_id for h in hits} == {"DDR-NHV-1", "DDR-SMR-1"}


def test_cross_field_hits_are_fused_into_one_ranking(searcher: Searcher) -> None:
    hits, plan = searcher.search("What happened on NHV-101 and SMR-201?")
    assert not plan.fields
    assert len(hits) == 2
    # RRF-fused scores: both present, sorted best first, never negative infinity or NaN.
    assert all(h.score == h.score for h in hits)  # not NaN
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_a_field_missing_from_the_index_map_yields_no_hits_for_it(two_field_store: Store) -> None:
    """A field the store knows about but whose index was never built (e.g. `ensure_field_indexes`
    has not run yet) degrades to no hits for that field, not a crash."""
    partial = {"Northaven": build_field_indexes(two_field_store)["Northaven"]}
    searcher = Searcher(two_field_store, partial, _embedder())
    hits, plan = searcher.search("What happened on Southmoor?")
    assert plan.fields == ["Southmoor"]
    assert hits == []
