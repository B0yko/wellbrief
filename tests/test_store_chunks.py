"""The `files`, `chunks` and `meta` tables, and the chunker (`store.chunk_document`)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from wellbrief.models import Document, FileRecord, NptEvent
from wellbrief.store import Store, chunk_document, chunk_document_id

# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


def test_a_short_document_is_one_chunk() -> None:
    text = "DAILY DRILLING REPORT\n\nDrilled ahead.\n"
    chunks = chunk_document("DDR-1", text)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "DDR-1#0"
    assert chunks[0].start == 0 and chunks[0].end == len(text)
    assert chunks[0].text == text


def test_the_threshold_is_4000_characters() -> None:
    assert len(chunk_document("D", "x" * 4000)) == 1
    assert len(chunk_document("D", "x" * 4001)) > 1


def test_a_long_document_is_split_into_chunks_of_at_most_1500_characters() -> None:
    text = "A" * 100 + "\n\n" + "B" * 1400 + "\n\n" + "C" * 1400 + "\n\n" + "D" * 3000
    chunks = chunk_document("DDR-2", text)
    assert all(c.end - c.start <= 1500 for c in chunks)
    # Contiguous and covers the whole text: no gap, no overlap.
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for a, b in pairwise(chunks):
        assert a.end == b.start
    assert all(c.text == text[c.start:c.end] for c in chunks)
    assert [c.n for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [f"DDR-2#{i}" for i in range(len(chunks))]


def test_a_long_document_prefers_to_split_at_a_blank_line() -> None:
    # Three 1,400-character paragraphs, each separated by a blank line: every
    # cut lands exactly on a paragraph boundary (reachable within the 1,500
    # limit), never mid-paragraph.
    paragraphs = ["A" * 1400, "B" * 1400, "C" * 1400]
    text = "\n\n".join(paragraphs)
    chunks = chunk_document("DDR-3", text)
    assert [c.text for c in chunks] == ["A" * 1400 + "\n\n", "B" * 1400 + "\n\n", "C" * 1400]
    assert "".join(c.text for c in chunks) == text


def test_a_long_document_prefers_to_split_at_a_heading() -> None:
    text = "X" * 1490 + "\nNPT DETAIL\n" + "Y" * 1500 + "\n\n" + "Z" * 1200
    chunks = chunk_document("DDR-4", text)
    assert any(c.text.startswith("NPT DETAIL") for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_chunk_document_id_recovers_the_document() -> None:
    assert chunk_document_id("DDR-ORD-101-005#0") == "DDR-ORD-101-005"
    assert chunk_document_id("DDR-ORD-101-005#12") == "DDR-ORD-101-005"


# ---------------------------------------------------------------------------
# Store: chunks follow documents
# ---------------------------------------------------------------------------


def _doc(doc_id: str, well: str, field: str, text: str) -> Document:
    return Document(doc_id, "ddr", well, field, "2024-01-01", "DAILY DRILLING REPORT", text)


def test_put_documents_chunks_every_document_it_stores(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "short text")])
    chunks = store.chunks(doc_id="DDR-1")
    assert len(chunks) == 1 and chunks[0].text == "short text"


def test_replacing_a_document_replaces_its_chunks(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "x" * 5000)])
    assert len(store.chunks(doc_id="DDR-1")) > 1
    store.put_documents([_doc("DDR-1", "W-1", "Field", "short")])
    chunks = store.chunks(doc_id="DDR-1")
    assert len(chunks) == 1 and chunks[0].text == "short"


def test_chunks_filters_by_field(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        _doc("DDR-A", "W-A", "Alpha", "alpha text"),
        _doc("DDR-B", "W-B", "Beta", "beta text"),
    ])
    assert {c.doc_id for c in store.chunks(field_name="Alpha")} == {"DDR-A"}
    assert {c.doc_id for c in store.chunks()} == {"DDR-A", "DDR-B"}


def test_field_names_includes_a_document_filed_under_an_empty_field_name(tmp_path: Path) -> None:
    # field_name is NOT NULL, so an empty string is a legitimate (if unusual)
    # value, not an absent one, and must not be silently dropped.
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-A", "W-A", "", "text")])
    assert store.field_names() == [""]


def test_field_names_and_field_counts(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        _doc("DDR-A", "W-A", "Alpha", "alpha text"),
        _doc("DDR-A2", "W-A", "Alpha", "x" * 5000),
        _doc("DDR-B", "W-B", "Beta", "beta text"),
    ])
    assert store.field_names() == ["Alpha", "Beta"]
    counts = store.field_counts("Alpha")
    assert counts["documents"] == 2
    assert counts["chunks"] > 2          # the long document split into more than one chunk
    assert counts["npt_events"] == 0


def test_reset_clears_chunks_and_files(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "text")])
    store.put_files([FileRecord("DDR-1.txt", "abc", 4, 0.0, ["DDR-1"], "ingested", None, "Field")])
    store.reset()
    assert store.chunks() == [] and store.files() == []


# ---------------------------------------------------------------------------
# npt_events table
# ---------------------------------------------------------------------------


def _event(doc_id: str, code: str, hours: float) -> NptEvent:
    return NptEvent(doc_id=doc_id, well="W-1", field_name="Field", date="2024-01-01", code=code,
                    hours=hours, hole_section='17 1/2"', formation="Keldra Salt", depth_m=1400.0,
                    mud_weight_sg=1.4, rig="Orrin-1", description="text")


def test_put_npt_replaces_a_documents_events_instead_of_appending(tmp_path: Path) -> None:
    # Re-running an ingest on an unchanged file (no incremental skip yet, see
    # ingest.py) must not double the NPT totals: put_npt mirrors put_chunks
    # and replaces a document's events rather than accumulating a second copy.
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "text")])
    store.put_npt([_event("DDR-1", "STUCK_PIPE", 23.5)])
    store.put_npt([_event("DDR-1", "STUCK_PIPE", 23.5)])
    events = store.npt(doc_id="DDR-1")
    assert len(events) == 1
    assert store.npt_summary([])["hours"] == 23.5


def test_put_npt_keeps_two_events_from_the_same_document(tmp_path: Path) -> None:
    # A single report can carry more than one NPT entry; replacing by doc_id
    # must not drop siblings written in the same call.
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "text")])
    store.put_npt([_event("DDR-1", "STUCK_PIPE", 23.5), _event("DDR-1", "WAIT_ON_WEATHER", 4.0)])
    assert {e.code for e in store.npt(doc_id="DDR-1")} == {"STUCK_PIPE", "WAIT_ON_WEATHER"}


def test_put_npt_leaves_other_documents_events_alone(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "a"), _doc("DDR-2", "W-1", "Field", "b")])
    store.put_npt([_event("DDR-1", "STUCK_PIPE", 1.0), _event("DDR-2", "STUCK_PIPE", 2.0)])
    store.put_npt([_event("DDR-1", "STUCK_PIPE", 1.0)])
    assert len(store.npt(doc_id="DDR-1")) == 1
    assert len(store.npt(doc_id="DDR-2")) == 1


# ---------------------------------------------------------------------------
# files table
# ---------------------------------------------------------------------------


def test_put_files_and_get_file(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_files([
        FileRecord("DDR-1.txt", "sha1", 100, 12.5, ["DDR-1"], "ingested", None, "Alpha"),
        FileRecord("_truth.json", "", 0, 0.0, [], "skipped", "ground-truth sidecar", ""),
    ])
    assert [f.path for f in store.files()] == ["DDR-1.txt", "_truth.json"]
    rec = store.get_file("DDR-1.txt")
    assert rec is not None and rec.doc_ids == ["DDR-1"] and rec.status == "ingested"
    skipped = store.get_file("_truth.json")
    assert skipped is not None and skipped.reason == "ground-truth sidecar"
    assert store.get_file("nope.txt") is None


def test_put_files_replaces_by_path(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_files([FileRecord("a.txt", "old", 1, 0.0, ["A"], "ingested", None, "F")])
    store.put_files([FileRecord("a.txt", "new", 2, 0.0, ["A"], "ingested", None, "F")])
    rows = store.files()
    assert len(rows) == 1 and rows[0].sha256 == "new"


# ---------------------------------------------------------------------------
# meta table
# ---------------------------------------------------------------------------


def test_meta_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    assert store.get_meta("schema_version") is None
    assert store.get_meta("schema_version", 1) == 1
    store.set_meta("schema_version", 2)
    assert store.get_meta("schema_version") == 2


def test_documents_carry_source_and_page_map(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    doc = Document("EOWR-1", "eowr", "W-1", "Field", "2024-01-01", "END OF WELL REPORT", "page one\fpage two",
                   source="reports/EOWR-1.pdf", page_map=[9, 18])
    store.put_documents([doc])
    back = store.get_document("EOWR-1")
    assert back is not None
    assert back.source == "reports/EOWR-1.pdf"
    assert back.page_map == [9, 18]


def test_a_document_with_no_page_map_reads_back_as_none(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([_doc("DDR-1", "W-1", "Field", "text")])
    back = store.get_document("DDR-1")
    assert back is not None and back.page_map is None and back.source == ""


@pytest.mark.parametrize("threshold", [4000])
def test_chunk_document_threshold_constant_matches_config(threshold: int) -> None:
    from wellbrief.config import CHUNK_MAX_CHARS, CHUNK_THRESHOLD_CHARS

    assert threshold == CHUNK_THRESHOLD_CHARS
    assert CHUNK_MAX_CHARS == 1500
