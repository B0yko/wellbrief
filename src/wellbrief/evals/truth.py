"""The ground truth of a generated corpus, as an in-memory SQLite database.

This is the only module that reads the generator's sidecar (`_truth.json`).
It copies every record into tables so the case files can state their gold
answers as SQL (`gold_sql`, `relevant_sql`, `precondition_sql`). Those
queries run here, never against the product's own database, and nothing in
this module calls product code, so the gold answers do not depend on the
code path under test.

Tables (list-valued fields are stored as JSON text; query them with
`json_each`):

  truth_meta       key, value: seed, scale, formats, footer
  truth_fields     name, prefix, operator, rigs, mwd_tools, patterns
  truth_wells      one row per well, with G1-G4 status (affected | clean | out_of_scope)
  truth_docs       one row per document (ddr, eowr, incident)
  truth_events     one row per NPT DETAIL entry of a daily report (the ledger rows);
                   an event that runs over several reports has one row per part,
                   joined by `occurrence`
  truth_sentences  one row per candidate sentence (lesson, recommendation,
                   corrective_action) with its true label
  truth_patterns   one row per planted-pattern key (a pattern with three codes
                   has three rows)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SIDECAR = "_truth.json"

Params = tuple[Any, ...] | dict[str, Any]

SCHEMA = """
CREATE TABLE truth_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE truth_fields (
    name TEXT PRIMARY KEY, prefix TEXT, operator TEXT, rigs TEXT, mwd_tools TEXT, patterns TEXT);
CREATE TABLE truth_wells (
    well TEXT PRIMARY KEY, field TEXT, rig TEXT, mwd TEXT, mwd_by_section TEXT, spud TEXT,
    release TEXT, td_m REAL, eowr_date TEXT, days_on_well INTEGER,
    g1 TEXT, g2 TEXT, g3 TEXT, g4 TEXT,
    salt_mw_sg REAL, salt_mw_ok INTEGER, lcm_pretreat INTEGER, cites TEXT);
CREATE TABLE truth_docs (
    doc_id TEXT PRIMARY KEY, type TEXT, field TEXT, well TEXT, date TEXT, file TEXT,
    sections TEXT, formations TEXT, rig TEXT, mwd TEXT,
    report_no INTEGER, drilled INTEGER, depth_start_m REAL, depth_end_m REAL, formation_at_td TEXT,
    mud_weight_sg REAL, bht_c REAL, npt_hours REAL, productive_hours REAL,
    code TEXT, hours REAL, depth_m REAL, section TEXT, formation TEXT,
    event_id TEXT, event_ids TEXT, occurrence TEXT);
CREATE TABLE truth_events (
    event_id TEXT PRIMARY KEY, doc_id TEXT, well TEXT, field TEXT, date TEXT, code TEXT,
    hours REAL, depth_m REAL, section TEXT, formation TEXT, rig TEXT, mwd TEXT, pattern TEXT,
    kind TEXT, variant TEXT, occurrence TEXT, part INTEGER, parts INTEGER);
CREATE TABLE truth_sentences (
    id INTEGER PRIMARY KEY, doc_id TEXT, well TEXT, field TEXT, kind TEXT, text TEXT,
    label TEXT, patterns TEXT, code TEXT);
CREATE TABLE truth_patterns (
    id TEXT, field TEXT, scope TEXT, driver TEXT, key_json TEXT,
    code TEXT, section TEXT, formation TEXT, rig TEXT, mwd TEXT);
"""

# Columns copied from each sidecar record; the column names are the record keys.
_COLUMNS: dict[str, tuple[str, ...]] = {
    "truth_fields": ("name", "prefix", "operator", "rigs", "mwd_tools", "patterns"),
    "truth_wells": ("well", "field", "rig", "mwd", "mwd_by_section", "spud", "release", "td_m", "eowr_date",
                    "days_on_well", "salt_mw_sg", "salt_mw_ok", "lcm_pretreat", "cites"),
    "truth_docs": ("doc_id", "type", "field", "well", "date", "file", "sections", "formations", "rig", "mwd",
                   "report_no", "drilled", "depth_start_m", "depth_end_m", "formation_at_td", "mud_weight_sg",
                   "bht_c", "npt_hours", "productive_hours", "code", "hours", "depth_m", "section", "formation",
                   "event_id", "event_ids", "occurrence"),
    "truth_events": ("event_id", "doc_id", "well", "field", "date", "code", "hours", "depth_m", "section",
                     "formation", "rig", "mwd", "pattern", "kind", "variant", "occurrence", "part", "parts"),
    "truth_sentences": ("doc_id", "well", "field", "kind", "text", "label", "patterns", "code"),
}

# Document columns that only one document type carries; every other listed column is required of every record.
_DOC_TYPE_COLUMNS: dict[str, tuple[str, ...]] = {
    "ddr": ("report_no", "drilled", "depth_start_m", "depth_end_m", "formation_at_td", "mud_weight_sg", "bht_c",
            "npt_hours", "productive_hours"),
    "eowr": (),
    "incident": ("code", "hours", "depth_m", "section", "formation", "event_id", "event_ids", "occurrence"),
}
_TYPED = {c for cols in _DOC_TYPE_COLUMNS.values() for c in cols}

LABELS = ("practice", "failure", "neutral")
STATUSES = ("affected", "clean", "out_of_scope")


class TruthError(Exception):
    """The sidecar is missing, unreadable or not in the expected shape."""


def _value(v: Any) -> Any:
    """SQLite-ready value: lists and dicts as JSON text, booleans as 0/1."""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=True)
    if isinstance(v, bool):
        return int(v)
    return v


def _required(table: str, record: dict[str, Any]) -> tuple[str, ...]:
    names = _COLUMNS[table]
    if table != "truth_docs":
        return names
    typed = _DOC_TYPE_COLUMNS.get(str(record.get("type")))
    if typed is None:
        raise TruthError(f"document {record.get('doc_id')!r} has an unknown type {record.get('type')!r}")
    return tuple(n for n in names if n not in _TYPED or n in typed)


def _insert(conn: sqlite3.Connection, table: str, records: list[dict[str, Any]]) -> None:
    """Copy records into `table`. A record that lacks a column it must carry is an error, never a NULL."""
    names = _COLUMNS[table]
    rows = []
    for r in records:
        missing = [n for n in _required(table, r) if n not in r]
        if missing:
            raise TruthError(f"a {table} record lacks {', '.join(missing)}: {str(r)[:120]}")
        rows.append(tuple(_value(r.get(n)) for n in names))
    conn.executemany(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})", rows)


def _check_values(data: dict[str, Any]) -> None:
    for s in data["sentences"]:
        if s["label"] not in LABELS:
            raise TruthError(f"sentence label {s['label']!r} is not one of {', '.join(LABELS)}")
    for w in data["wells"]:
        bad = {p: v for p, v in w["patterns"].items() if v not in STATUSES}
        if bad:
            raise TruthError(f"{w['well']}: pattern status {bad} is not one of {', '.join(STATUSES)}")


class Truth:
    """Read-only access to the truth database of one generated corpus."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def meta(self, key: str) -> Any:
        row = self.conn.execute("SELECT value FROM truth_meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def rows(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        """Run a query; `params` are positional (?) or named (:name)."""
        return list(self.conn.execute(sql, params))

    def scalar(self, sql: str, params: Params = ()) -> Any:
        """The first column of the first row; the query must return exactly one row."""
        rows = self.rows(sql, params)
        if len(rows) != 1:
            raise TruthError(f"expected one row, the query returned {len(rows)}: {sql}")
        return rows[0][0]

    def column(self, sql: str, params: Params = ()) -> list[Any]:
        """The first column of every row."""
        return [r[0] for r in self.rows(sql, params)]

    def doc(self, doc_id: str) -> sqlite3.Row | None:
        rows = self.rows("SELECT * FROM truth_docs WHERE doc_id = ?", (doc_id,))
        return rows[0] if rows else None


def load(data: dict[str, Any]) -> Truth:
    """Build the truth database from the parsed sidecar. The database is read-only afterwards."""
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise TruthError("unsupported truth schema")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        _check_values(data)
        meta = {k: data.get(k) for k in ("schema", "seed", "scale", "formats", "footer")}
        conn.executemany("INSERT INTO truth_meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
        _insert(conn, "truth_fields", data["fields"])
        _insert(conn, "truth_wells", data["wells"])
        for w in data["wells"]:
            status = w["patterns"]
            conn.execute("UPDATE truth_wells SET g1 = ?, g2 = ?, g3 = ?, g4 = ? WHERE well = ?",
                         (status.get("G1"), status.get("G2"), status.get("G3"), status.get("G4"), w["well"]))
        _insert(conn, "truth_docs", data["documents"])
        _insert(conn, "truth_events", data["npt_events"])
        _insert(conn, "truth_sentences", data["sentences"])
        for p in data["planted_patterns"]:
            for key in p["keys"]:
                conn.execute(
                    "INSERT INTO truth_patterns VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (p["id"], p["field"], p["scope"], p["driver"], json.dumps(key, sort_keys=True),
                     key.get("code"), key.get("section"), key.get("formation"), key.get("rig"), key.get("mwd")))
    except (KeyError, TypeError, AttributeError, sqlite3.Error) as exc:
        raise TruthError(f"the truth file is not in the expected shape: {exc!r}") from exc
    conn.commit()
    conn.execute("PRAGMA query_only = ON")
    return Truth(conn)


def load_dir(corpus_dir: Path | str) -> Truth:
    """Read the sidecar of the corpus in `corpus_dir`."""
    path = Path(corpus_dir) / SIDECAR
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TruthError(f"cannot read the corpus ground truth ({SIDECAR}): {exc}") from exc
    return load(data)
