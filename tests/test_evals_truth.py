"""The truth database: a faithful, read-only copy of the generator's sidecar."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from wellbrief.corpus import build_corpus, write_corpus
from wellbrief.evals import truth


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("truth-corpus")
    write_corpus(build_corpus(), out)
    return out


@pytest.fixture(scope="module")
def sidecar(corpus_dir: Path) -> dict[str, Any]:
    return json.loads((corpus_dir / truth.SIDECAR).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def db(corpus_dir: Path) -> truth.Truth:
    return truth.load_dir(corpus_dir)


def test_every_record_becomes_one_row(db: truth.Truth, sidecar: dict[str, Any]) -> None:
    assert db.meta("seed") == sidecar["seed"]
    assert db.meta("footer") == sidecar["footer"]
    for table, key in (("truth_fields", "fields"), ("truth_wells", "wells"), ("truth_docs", "documents"),
                       ("truth_events", "npt_events"), ("truth_sentences", "sentences")):
        assert db.scalar(f"SELECT COUNT(*) FROM {table}") == len(sidecar[key]), table
    keys = sum(len(p["keys"]) for p in sidecar["planted_patterns"])
    assert db.scalar("SELECT COUNT(*) FROM truth_patterns") == keys


def test_event_fields_survive_including_the_parts_of_long_events(db: truth.Truth, sidecar: dict[str, Any]) -> None:
    long_event = next(e for e in sidecar["npt_events"] if e["parts"] > 1)
    row = db.rows("SELECT * FROM truth_events WHERE event_id = ?", (long_event["event_id"],))[0]
    for key in ("doc_id", "well", "field", "date", "code", "hours", "depth_m", "section", "formation", "rig",
                "mwd", "pattern", "kind", "variant", "occurrence", "part", "parts"):
        assert row[key] == long_event[key], key
    total = sum(e["hours"] for e in sidecar["npt_events"])
    assert db.scalar("SELECT SUM(hours) FROM truth_events") == pytest.approx(total)


def test_documents_wells_sentences_and_patterns(db: truth.Truth, sidecar: dict[str, Any]) -> None:
    incident = next(d for d in sidecar["documents"] if d["type"] == "incident")
    row = db.doc(incident["doc_id"])
    assert row is not None
    assert (row["code"], row["hours"], row["occurrence"]) == (incident["code"], incident["hours"],
                                                             incident["occurrence"])
    assert json.loads(row["event_ids"]) == incident["event_ids"]
    assert json.loads(row["sections"]) == incident["sections"]
    assert db.doc("DDR-NOPE-001") is None

    well = sidecar["wells"][0]
    w = db.rows("SELECT * FROM truth_wells WHERE well = ?", (well["well"],))[0]
    assert [w["g1"], w["g2"], w["g3"], w["g4"]] == [well["patterns"][p] for p in ("G1", "G2", "G3", "G4")]

    sentence = sidecar["sentences"][0]
    s = db.rows("SELECT * FROM truth_sentences ORDER BY id LIMIT 1")[0]
    assert (s["text"], s["label"], json.loads(s["patterns"])) == (sentence["text"], sentence["label"],
                                                                 sentence["patterns"])

    g3 = db.rows("SELECT * FROM truth_patterns WHERE id = 'G3' AND field = 'Orrindale'")[0]
    assert (g3["code"], g3["mwd"], g3["section"], g3["driver"]) == ("DOWNHOLE_TOOL_FAILURE", "PJ-3", '12 1/4"',
                                                                    "tool")


def test_json_lists_can_be_queried(db: truth.Truth) -> None:
    salt_reports = db.scalar("SELECT COUNT(*) FROM truth_docs d, json_each(d.formations) f "
                             "WHERE d.type = 'ddr' AND f.value = 'Keldra Salt'")
    assert salt_reports > 0


def test_named_parameters(db: truth.Truth) -> None:
    hours = db.scalar("SELECT SUM(hours) FROM truth_events WHERE field = :f", {"f": "Orrindale"})
    assert hours > 0


def test_the_database_is_read_only(db: truth.Truth) -> None:
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute("DELETE FROM truth_events")


def test_scalar_needs_exactly_one_row(db: truth.Truth) -> None:
    with pytest.raises(truth.TruthError):
        db.scalar("SELECT well FROM truth_wells")


def _broken(sidecar: dict[str, Any], table: str, index: int, change: dict[str, Any],
            drop: str | None = None) -> dict[str, Any]:
    records = [dict(r) for r in sidecar[table]]
    records[index].update(change)
    if drop:
        del records[index][drop]
    return {**sidecar, table: records}


@pytest.mark.parametrize("table, change, drop, message", [
    ("npt_events", {}, "hours", "truth_events record lacks hours"),
    ("wells", {}, "rig", "truth_wells record lacks rig"),
    ("sentences", {"label": "practise"}, None, "label 'practise'"),
    ("wells", {"patterns": {"G1": "affectd"}}, None, "pattern status"),
    ("documents", {"type": "memo"}, None, "unknown type"),
])
def test_a_record_of_the_wrong_shape_is_a_truth_error(sidecar: dict[str, Any], table: str, change: dict[str, Any],
                                                      drop: str | None, message: str) -> None:
    with pytest.raises(truth.TruthError, match=message):
        truth.load(_broken(sidecar, table, 0, change, drop))


def test_type_specific_document_columns_are_required_of_their_type_only(sidecar: dict[str, Any]) -> None:
    incident = next(i for i, d in enumerate(sidecar["documents"]) if d["type"] == "incident")
    with pytest.raises(truth.TruthError, match="truth_docs record lacks code"):
        truth.load(_broken(sidecar, "documents", incident, {}, "code"))
    eowr = next(d for d in sidecar["documents"] if d["type"] == "eowr")
    assert "code" not in eowr and "report_no" not in eowr


def test_a_missing_or_foreign_sidecar_is_a_truth_error(tmp_path: Path) -> None:
    with pytest.raises(truth.TruthError):
        truth.load_dir(tmp_path)
    (tmp_path / truth.SIDECAR).write_text('{"schema": 2}', encoding="utf-8")
    with pytest.raises(truth.TruthError):
        truth.load_dir(tmp_path)
    with pytest.raises(truth.TruthError):
        truth.load({"schema": 1, "fields": []})
