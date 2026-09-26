"""`ingest.ingest_folder`: incremental ingestion by sha256, type and field detection, the CSV
ledger, `--prune` and `--dry-run`."""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief.config import SYNTHETIC_FOOTER
from wellbrief.corpus import build_corpus, write_corpus
from wellbrief.corpus.docxwriter import write_docx
from wellbrief.corpus.pdfwriter import write_pdf
from wellbrief.ingest import ingest_folder
from wellbrief.store import Store

FOOTER = SYNTHETIC_FOOTER


def _ddr_text(well: str, field: str, rig: str = "Orrin-1", code: str | None = None,
             hours: float = 0.0) -> str:
    entries = [
        "NPT DETAIL",
        f"  Code                  : {code}",
        f"  Hours                 : {hours}",
        "  Depth                 : 1,400 m MD",
        "  Formation             : Keldra Salt",
        "  Description           : Packed off while pulling out of hole.",
        "",
    ] if code else ["NPT DETAIL", "  None reported this period.", ""]
    return "\n".join([
        "DAILY DRILLING REPORT",
        f"Operator: Quillfen Energy    Field: {field}    Well: {well}",
        f"Rig: {rig}    Report No: 001    Date: 2026-01-01",
        "",
        "DEPTH",
        "  Depth at start        : 1,300 m MD",
        "  Depth at end          : 1,400 m MD",
        '  Hole section          : 17 1/2"',
        "  Formation at TD       : Keldra Salt",
        "",
        "MUD",
        "  Weight                : 1.40 sg",
        "",
        "OPERATIONS SUMMARY (24 h)",
        "  06:00-06:00  Drilled ahead.",
        "",
        *entries,
        "REMARKS",
        "",
        FOOTER,
    ])


def test_ingests_txt_and_md_and_reports_coverage(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    (root / "DDR-ORD-101-002.md").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")

    store = Store(tmp_path / "wellbrief.db")
    coverage = ingest_folder(store, root)

    assert coverage.files_seen == 2
    assert coverage.ingested == 2
    assert coverage.unchanged == 0
    assert coverage.documents == 2
    assert coverage.fields == {"Orrindale"}
    assert coverage.ddr_extraction["section"] == 1.0
    assert coverage.ddr_extraction["npt_blocks"] == 1.0
    assert store.get_document("DDR-ORD-101-001") is not None
    assert store.get_document("DDR-ORD-101-002") is not None


def test_a_second_run_on_unchanged_files_ingests_nothing(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)

    coverage = ingest_folder(store, root)

    assert coverage.ingested == 0
    assert coverage.unchanged == 1
    assert coverage.documents == 0


def test_a_changed_file_replaces_its_document_and_events(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    path = root / "DDR-ORD-101-001.txt"
    path.write_text(_ddr_text("ORD-101", "Orrindale", code="STUCK_PIPE", hours=10.0), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)
    assert store.npt(doc_id="DDR-ORD-101-001")[0].hours == 10.0

    path.write_text(_ddr_text("ORD-101", "Orrindale", code="STUCK_PIPE", hours=20.0), encoding="utf-8")
    coverage = ingest_folder(store, root)

    assert coverage.ingested == 1
    events = store.npt(doc_id="DDR-ORD-101-001")
    assert len(events) == 1 and events[0].hours == 20.0


def test_prune_removes_documents_whose_files_are_gone(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    keep = root / "DDR-ORD-101-001.txt"
    gone = root / "DDR-ORD-101-002.txt"
    keep.write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    gone.write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)
    gone.unlink()

    without_prune = ingest_folder(store, root)
    assert without_prune.pruned == 0
    assert store.get_document("DDR-ORD-101-002") is not None

    with_prune = ingest_folder(store, root, prune=True)
    assert with_prune.pruned == 1
    assert store.get_document("DDR-ORD-101-002") is None
    assert store.get_document("DDR-ORD-101-001") is not None
    assert store.get_file("DDR-ORD-101-002.txt") is None


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root, dry_run=True)

    assert coverage.ingested == 1
    assert coverage.documents == 1
    assert store.counts()["documents"] == 0
    assert store.files() == []


def test_dry_run_reports_a_duplicate_it_cannot_persist(tmp_path: Path) -> None:
    """The CSV/DDR dedup check must not depend on anything actually being written, since a
    dry run never writes: it needs to see it purely from this run's own in-memory events."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(
        _ddr_text("ORD-101", "Orrindale", code="STUCK_PIPE", hours=10.0), encoding="utf-8")
    (root / "ledger.csv").write_text(
        "well,date,code,hours,field\nORD-101,2026-01-01,STUCK_PIPE,10.0,Orrindale\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root, dry_run=True)

    assert coverage.duplicates_merged == 1
    assert coverage.npt_events == 1  # only the DDR entry, not a doubled CSV row


def test_doc_id_gets_a_hash_suffix_on_a_stem_collision(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "report.txt").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    (root / "report.md").write_text(_ddr_text("ORD-102", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.ingested == 2
    ids = {f.doc_ids[0] for f in store.files()}
    assert len(ids) == 2
    assert all(i.startswith("report-") and len(i) == len("report-") + 8 for i in ids)


def test_no_collision_keeps_the_plain_stem_as_doc_id(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    ingest_folder(store, root)

    assert store.get_document("DDR-ORD-101-001") is not None


def test_unsupported_extension_is_skipped_with_a_reason(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "notes.rtf").write_text("not a supported format", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.files_seen == 1
    assert coverage.ingested == 0
    assert [s.reason for s in coverage.skipped] == ["unsupported extension"]


def test_underscore_files_and_wellbrief_toml_are_never_seen(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "_truth.json").write_text("{}", encoding="utf-8")
    (root / "wellbrief.toml").write_text("[parse]\n", encoding="utf-8")
    (root / "DDR-ORD-101-001.txt").write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.files_seen == 1
    assert coverage.skipped == []


def test_a_document_with_no_detected_field_falls_back_to_the_field_flag(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "notes.txt").write_text("Just some field notes with no header at all.\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    ingest_folder(store, root, field="Orrindale")
    doc = store.get_document("notes")
    assert doc is not None
    assert doc.doc_type == "other"
    assert doc.field_name == "Orrindale"


def test_a_document_with_no_detected_field_and_no_flag_is_unassigned(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "notes.txt").write_text("Just some field notes with no header at all.\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    ingest_folder(store, root)

    doc = store.get_document("notes")
    assert doc is not None and doc.field_name == "unassigned"


# ---------------------------------------------------------------------------
# CSV ledger
# ---------------------------------------------------------------------------


def test_csv_rows_become_npt_events_and_citable_documents(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "ledger.csv").write_text(
        "well,date,code,hours,field,depth,section,formation,rig,description\n"
        'ORD-501,2026-01-01,STUCK_PIPE,12.5,Orrindale,1400,17 1/2",Keldra Salt,Orrin-1,Packed off.\n',
        encoding="utf-8",
    )
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.documents == 1
    assert coverage.npt_events == 1
    events = store.npt(well="ORD-501")
    assert len(events) == 1
    assert events[0].code == "STUCK_PIPE" and events[0].hours == 12.5
    docs = store.documents(well="ORD-501")
    assert len(docs) == 1
    assert docs[0].doc_type == "csv"
    assert "Packed off." in docs[0].text


def test_a_csv_row_matching_a_ddr_entry_is_merged_once(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(
        _ddr_text("ORD-101", "Orrindale", code="STUCK_PIPE", hours=23.5), encoding="utf-8")
    (root / "ledger.csv").write_text(
        "well,date,code,hours,field\nORD-101,2026-01-01,STUCK_PIPE,23.52,Orrindale\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.duplicates_merged == 1
    # One event (the DDR's), not two; the CSV row is still a document.
    assert store.npt_summary([])["events"] == 1
    assert store.npt_summary([])["hours"] == 23.5
    assert coverage.documents == 2  # the DDR document and the CSV row document


def test_a_csv_row_that_does_not_match_any_ddr_entry_is_kept(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "DDR-ORD-101-001.txt").write_text(
        _ddr_text("ORD-101", "Orrindale", code="STUCK_PIPE", hours=23.5), encoding="utf-8")
    (root / "ledger.csv").write_text(
        "well,date,code,hours,field\nORD-101,2026-01-01,STUCK_PIPE,5.0,Orrindale\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.duplicates_merged == 0
    assert store.npt_summary([])["events"] == 2
    assert store.npt_summary([])["hours"] == 28.5


def test_reingesting_an_unchanged_csv_row_with_no_ddr_is_not_a_duplicate(tmp_path: Path) -> None:
    """A CSV row must only ever be deduped against a real DDR NPT entry, never against another
    CSV row's own previously-stored event -- including its own prior self on re-ingest."""
    root = tmp_path / "corpus"
    root.mkdir()
    path = root / "ledger.csv"
    path.write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)
    assert store.npt_summary([])["events"] == 1

    # Append an unrelated row so the file's sha256 changes and it is re-read this run.
    path.write_text(
        "well,date,code,hours,field\n"
        "ORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n"
        "ORD-502,2026-01-02,STUCK_PIPE,2.0,Orrindale\n",
        encoding="utf-8",
    )
    coverage = ingest_folder(store, root)

    assert coverage.duplicates_merged == 0
    assert store.npt_summary([])["events"] == 2
    assert store.npt_summary([])["hours"] == 3.0


def test_two_csv_files_with_the_same_row_are_not_merged_into_one_event(tmp_path: Path) -> None:
    """CSV-vs-CSV dedup is not defined: two independent ledgers that happen to describe the same
    well/date/code/hours must both keep their own event, since neither is a DDR."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "ledger-a.csv").write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n", encoding="utf-8")
    (root / "ledger-b.csv").write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.duplicates_merged == 0
    assert store.npt_summary([])["events"] == 2
    assert store.npt_summary([])["hours"] == 2.0


def test_a_csv_code_the_taxonomy_does_not_know_becomes_other(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "ledger.csv").write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,SITE SPECIFIC CODE,1.0,Orrindale\n",
        encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    ingest_folder(store, root)

    assert store.npt(well="ORD-501")[0].code == "OTHER"


def test_a_file_that_stops_parsing_no_longer_leaves_its_old_document_in_the_store(
        tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    path = root / "DDR-ORD-101-001.txt"
    path.write_text(_ddr_text("ORD-101", "Orrindale"), encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)
    assert store.get_document("DDR-ORD-101-001") is not None

    path.write_bytes(b"caf\xe9 au lait")  # no longer valid UTF-8
    coverage = ingest_folder(store, root)

    assert len(coverage.skipped) == 1
    assert store.get_document("DDR-ORD-101-001") is None
    record = store.get_file("DDR-ORD-101-001.txt")
    assert record is not None and record.status == "skipped"


def test_a_csv_ledger_that_becomes_unreadable_removes_its_old_rows(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    path = root / "ledger.csv"
    path.write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,12.5,Orrindale\n", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)
    assert store.npt(well="ORD-501")

    path.write_text("", encoding="utf-8")  # an empty CSV file is a ReaderError, not zero rows
    coverage = ingest_folder(store, root)

    assert len(coverage.skipped) == 1
    assert store.npt(well="ORD-501") == []


def test_a_malformed_csv_ledger_is_skipped_with_its_reason(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "ledger.csv").write_text("", encoding="utf-8")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.documents == 0
    assert len(coverage.skipped) == 1
    assert "empty" in coverage.skipped[0].reason


def test_a_bad_csv_row_is_reported_and_the_good_rows_are_kept(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "ledger.csv").write_text(
        "well,date,code,hours,field\n"
        "ORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n"
        "ORD-502,2026-01-02,STUCK_PIPE,not-a-number,Orrindale\n",
        encoding="utf-8",
    )
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert store.npt(well="ORD-501")
    assert store.npt(well="ORD-502") == []
    assert len(coverage.csv_row_errors) == 1
    assert "ledger.csv" in coverage.csv_row_errors[0]
    assert "line 3" in coverage.csv_row_errors[0]


def test_a_changed_csv_ledger_replaces_rows_that_are_no_longer_produced(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    path = root / "ledger.csv"
    path.write_text(
        "well,date,code,hours,field\n"
        "ORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n"
        "ORD-502,2026-01-02,STUCK_PIPE,2.0,Orrindale\n",
        encoding="utf-8",
    )
    store = Store(tmp_path / "wellbrief.db")
    ingest_folder(store, root)
    assert len(store.npt(field_name="Orrindale")) == 2

    path.write_text("well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,1.0,Orrindale\n",
                    encoding="utf-8")
    ingest_folder(store, root)

    remaining = store.npt(field_name="Orrindale")
    assert len(remaining) == 1
    assert remaining[0].well == "ORD-501"


# ---------------------------------------------------------------------------
# PDF and DOCX
# ---------------------------------------------------------------------------


def test_a_pdf_document_carries_a_page_map(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    lines = _ddr_text("ORD-101", "Orrindale").splitlines()
    write_pdf(lines, root / "DDR-ORD-101-001.pdf")
    store = Store(tmp_path / "wellbrief.db")

    ingest_folder(store, root)

    doc = store.get_document("DDR-ORD-101-001")
    assert doc is not None
    assert doc.doc_type == "ddr"
    assert doc.page_map is not None and doc.page_map[-1] == len(doc.text)


def test_a_docx_document_ingests_correctly(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    lines = _ddr_text("ORD-101", "Orrindale").splitlines()
    write_docx(lines, root / "DDR-ORD-101-001.docx")
    store = Store(tmp_path / "wellbrief.db")

    ingest_folder(store, root)

    doc = store.get_document("DDR-ORD-101-001")
    assert doc is not None
    assert doc.doc_type == "ddr"
    assert doc.well == "ORD-101"
    assert doc.page_map is None


def test_a_pdf_with_no_text_layer_is_skipped_with_its_reason(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    write_pdf([], root / "scan.pdf")
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, root)

    assert coverage.documents == 0
    assert len(coverage.skipped) == 1
    assert "text layer" in coverage.skipped[0].reason


# ---------------------------------------------------------------------------
# Whole generated corpus, rendered "mixed" with an NPT ledger CSV
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("mixed-corpus")
    write_corpus(build_corpus(), out, "mixed", ledger_csv=True)
    return out


def test_a_mixed_format_corpus_with_a_ledger_ingests_without_doubling_npt_hours(
        mixed_corpus_dir: Path, tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")

    coverage = ingest_folder(store, mixed_corpus_dir)

    assert coverage.duplicates_merged == coverage.npt_events  # every CSV row matched its DDR entry
    assert store.npt_summary([])["events"] == coverage.npt_events
    assert store.documents(doc_type="ddr")
    assert store.documents(doc_type="eowr")
    assert store.documents(doc_type="incident")
    assert store.documents(doc_type="csv")
