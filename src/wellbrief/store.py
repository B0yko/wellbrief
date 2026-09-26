"""SQLite store: the corpus, the parsed NPT ledger, the well register and the chunk table.

One SQLite file (`files`, `documents`, `chunks`, `npt_events`, `wells`,
`meta`) holds the whole state of a workspace's data; the per-field retrieval
indexes built over its chunks (BM25 postings, hashing vectors, a manifest)
live next to it as described in `workspace.py`. Nothing else is read at
query time, and no network access is needed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import CHUNK_MAX_CHARS, CHUNK_THRESHOLD_CHARS
from .models import Chunk, Document, FileRecord, NptEvent, Well

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    doc_type   TEXT NOT NULL,
    well       TEXT NOT NULL,
    field_name TEXT NOT NULL,
    date       TEXT NOT NULL,
    title      TEXT NOT NULL,
    text       TEXT NOT NULL,
    meta       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT '',
    page_map   TEXT
);
CREATE INDEX IF NOT EXISTS ix_documents_well  ON documents(well);
CREATE INDEX IF NOT EXISTS ix_documents_field ON documents(field_name);
CREATE INDEX IF NOT EXISTS ix_documents_type  ON documents(doc_type);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL REFERENCES documents(doc_id),
    n        INTEGER NOT NULL,
    start    INTEGER NOT NULL,
    end      INTEGER NOT NULL,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS npt_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id),
    well          TEXT NOT NULL,
    field_name    TEXT NOT NULL,
    date          TEXT NOT NULL,
    code          TEXT NOT NULL,
    hours         REAL NOT NULL,
    hole_section  TEXT NOT NULL,
    formation     TEXT NOT NULL,
    depth_m       REAL NOT NULL,
    mud_weight_sg REAL NOT NULL,
    rig           TEXT NOT NULL,
    description   TEXT NOT NULL,
    mwd           TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_npt_field ON npt_events(field_name);
CREATE INDEX IF NOT EXISTS ix_npt_code  ON npt_events(code);
CREATE INDEX IF NOT EXISTS ix_npt_well  ON npt_events(well);

CREATE TABLE IF NOT EXISTS wells (
    name       TEXT PRIMARY KEY,
    field_name TEXT NOT NULL,
    rig        TEXT NOT NULL,
    spud_date  TEXT NOT NULL,
    td_m       REAL NOT NULL,
    sections   TEXT NOT NULL,
    formations TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    size       INTEGER NOT NULL,
    mtime      REAL NOT NULL,
    doc_ids    TEXT NOT NULL,
    status     TEXT NOT NULL,
    reason     TEXT,
    field_name TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        for table in ("chunks", "npt_events", "documents", "wells", "files", "meta"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    # -- writes: documents --------------------------------------------------
    def put_documents(self, docs: Iterable[Document]) -> int:
        """Insert or replace documents, and (re-)chunk each one (see `chunk_document`).

        A document's chunks are always derived from its current text and
        kept in step with it here, so every writer that stores a document --
        `ingest.ingest`, a hand-built test fixture, a future incremental
        re-ingest of a changed file -- gets a chunk table for free, and a
        replaced document never keeps a stale chunk from its previous text.
        """
        docs = list(docs)
        rows = [
            (d.doc_id, d.doc_type, d.well, d.field_name, d.date, d.title, d.text, json.dumps(d.meta),
             d.source, json.dumps(d.page_map) if d.page_map is not None else None)
            for d in docs
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()
        self.put_chunks(chunk for d in docs for chunk in chunk_document(d.doc_id, d.text))
        return len(rows)

    def put_wells(self, wells: Iterable[Well]) -> int:
        rows = [
            (w.name, w.field_name, w.rig, w.spud_date, w.td_m,
             json.dumps(w.sections), json.dumps(w.formations))
            for w in wells
        ]
        self.conn.executemany("INSERT OR REPLACE INTO wells VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def put_npt(self, events: Iterable[NptEvent]) -> int:
        """Replace the NPT events of every document `events` touches, then insert them.

        Mirrors `put_chunks`: a document's whole set of events is always
        written together, so re-ingesting the same file (for example running
        `wellbrief ingest` twice on an unchanged folder) replaces its events
        rather than appending a second copy of them.
        """
        rows = [
            (e.doc_id, e.well, e.field_name, e.date, e.code, e.hours, e.hole_section,
             e.formation, e.depth_m, e.mud_weight_sg, e.rig, e.description, e.mwd)
            for e in events
        ]
        doc_ids = sorted({r[0] for r in rows})
        if doc_ids:
            self.conn.executemany("DELETE FROM npt_events WHERE doc_id = ?", [(d,) for d in doc_ids])
        self.conn.executemany(
            "INSERT INTO npt_events (doc_id, well, field_name, date, code, hours, hole_section,"
            " formation, depth_m, mud_weight_sg, rig, description, mwd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    # -- writes: chunks -------------------------------------------------------
    def put_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Replace the chunks of every document `chunks` touches, then insert them.

        A document's whole set of chunks is always written together, so a
        re-ingested (changed) document never keeps a stale chunk from a
        shorter earlier version of itself.
        """
        rows = list(chunks)
        doc_ids = sorted({c.doc_id for c in rows})
        if doc_ids:
            self.conn.executemany("DELETE FROM chunks WHERE doc_id = ?", [(d,) for d in doc_ids])
        self.conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, n, start, end, text) VALUES (?,?,?,?,?,?)",
            [(c.chunk_id, c.doc_id, c.n, c.start, c.end, c.text) for c in rows],
        )
        self.conn.commit()
        return len(rows)

    def chunks(self, field_name: str | None = None, doc_id: str | None = None) -> list[Chunk]:
        sql = "SELECT c.chunk_id, c.doc_id, c.n, c.start, c.end, c.text FROM chunks c"
        clauses, params = [], []
        if field_name is not None:
            sql += " JOIN documents d ON d.doc_id = c.doc_id"
            clauses.append("d.field_name = ?")
            params.append(field_name)
        if doc_id is not None:
            clauses.append("c.doc_id = ?")
            params.append(doc_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY c.doc_id, c.n"
        return [
            Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], n=r["n"], start=r["start"], end=r["end"],
                  text=r["text"])
            for r in self.conn.execute(sql, params)
        ]

    # -- writes: files ----------------------------------------------------
    def put_files(self, records: Iterable[FileRecord]) -> int:
        rows = [
            (r.path, r.sha256, r.size, r.mtime, json.dumps(r.doc_ids), r.status, r.reason, r.field_name)
            for r in records
        ]
        self.conn.executemany("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    @staticmethod
    def _file(row: sqlite3.Row) -> FileRecord:
        return FileRecord(path=row["path"], sha256=row["sha256"], size=row["size"], mtime=row["mtime"],
                          doc_ids=json.loads(row["doc_ids"]), status=row["status"], reason=row["reason"],
                          field_name=row["field_name"])

    def files(self) -> list[FileRecord]:
        return [self._file(r) for r in self.conn.execute("SELECT * FROM files ORDER BY path")]

    def get_file(self, path: str) -> FileRecord | None:
        row = self.conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return self._file(row) if row else None

    # -- meta (small keyed values, e.g. schema bookkeeping) ------------------
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return json.loads(row["v"]) if row else default

    # -- reads: documents ---------------------------------------------------
    @staticmethod
    def _doc(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"], doc_type=row["doc_type"], well=row["well"],
            field_name=row["field_name"], date=row["date"], title=row["title"],
            text=row["text"], meta=json.loads(row["meta"]), source=row["source"],
            page_map=json.loads(row["page_map"]) if row["page_map"] is not None else None,
        )

    def get_document(self, doc_id: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    @staticmethod
    def _documents_where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for col in ("doc_type", "well", "field_name"):
            if filters.get(col):
                val = filters[col]
                if isinstance(val, (list, tuple, set)):
                    clauses.append(f"{col} IN ({','.join('?' * len(val))})")
                    params.extend(sorted(val) if isinstance(val, set) else val)
                else:
                    clauses.append(f"{col}=?")
                    params.append(val)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def documents(self, **filters: Any) -> list[Document]:
        where, params = self._documents_where(filters)
        sql = "SELECT * FROM documents" + where + " ORDER BY doc_id"
        return [self._doc(r) for r in self.conn.execute(sql, params)]

    def document_ids(self, **filters: Any) -> set[str]:
        """The ids of the documents `documents(**filters)` would return, without their text."""
        where, params = self._documents_where(filters)
        return {r[0] for r in self.conn.execute("SELECT doc_id FROM documents" + where, params)}

    def documents_drilled_through(self, depth_min: float, depth_max: float, **filters: Any) -> set[str]:
        """The daily reports whose drilled interval (depth at start to depth at end, as
        parsed) meets [depth_min, depth_max]; `filters` as for `documents`."""
        where, params = self._documents_where({**filters, "doc_type": "ddr"})
        start, end = "json_extract(meta, '$.depth_start_m')", "json_extract(meta, '$.depth_end_m')"
        sql = f"SELECT doc_id FROM documents{where} AND {end} >= ? AND COALESCE({start}, {end}) <= ?"
        return {r[0] for r in self.conn.execute(sql, [*params, depth_min, depth_max])}

    def field_names(self) -> list[str]:
        """Every distinct `field_name` documents are filed under, whatever it is
        (including a placeholder like `"unassigned"` or `"UNKNOWN"`, or even an
        empty string): the set a per-field index is built over. `field_name` is
        `NOT NULL`, so nothing here is actually excluded.
        `catalog()["fields"]` is the narrower, planner-facing vocabulary of
        *named* fields."""
        return sorted({r[0] for r in self.conn.execute("SELECT DISTINCT field_name FROM documents")})

    def field_counts(self, field_name: str) -> dict[str, int]:
        """Documents, chunks and NPT events filed under one field (`wellbrief status`)."""
        documents = self.conn.execute(
            "SELECT COUNT(*) FROM documents WHERE field_name=?", (field_name,)).fetchone()[0]
        chunks = self.conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE d.field_name=?",
            (field_name,)).fetchone()[0]
        npt_events = self.conn.execute(
            "SELECT COUNT(*) FROM npt_events WHERE field_name=?", (field_name,)).fetchone()[0]
        return {"documents": documents, "chunks": chunks, "npt_events": npt_events}

    def wells(self, field_name: str | None = None) -> list[Well]:
        sql, params = "SELECT * FROM wells", []
        if field_name:
            sql += " WHERE field_name=?"
            params.append(field_name)
        sql += " ORDER BY name"
        return [
            Well(name=r["name"], field_name=r["field_name"], rig=r["rig"], spud_date=r["spud_date"],
                 td_m=r["td_m"], sections=json.loads(r["sections"]), formations=json.loads(r["formations"]))
            for r in self.conn.execute(sql, params)
        ]

    # Columns of npt_events a caller may filter or group on.
    NPT_COLUMNS = ("field_name", "well", "code", "hole_section", "formation", "rig", "mwd", "doc_id")

    @classmethod
    def _npt_where(cls, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """The WHERE clause shared by every ledger query.

        A column filter is a value or a list of values; `since` / `until` bound
        the date and `depth_min` / `depth_max` the depth (both inclusive).
        """
        clauses: list[str] = []
        params: list[Any] = []
        for col in cls.NPT_COLUMNS:
            if filters.get(col):
                val = filters[col]
                if isinstance(val, (list, tuple, set)):
                    clauses.append(f"{col} IN ({','.join('?' * len(val))})")
                    params.extend(sorted(val) if isinstance(val, set) else val)
                else:
                    clauses.append(f"{col}=?")
                    params.append(val)
        for key, clause in (("since", "date >= ?"), ("until", "date <= ?"),
                            ("depth_min", "depth_m >= ?"), ("depth_max", "depth_m <= ?")):
            if filters.get(key) is not None:
                clauses.append(clause)
                params.append(filters[key])
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def npt(self, **filters: Any) -> list[NptEvent]:
        where, params = self._npt_where(filters)
        sql = "SELECT * FROM npt_events" + where + " ORDER BY date, doc_id, id"
        return [
            NptEvent(
                doc_id=r["doc_id"], well=r["well"], field_name=r["field_name"], date=r["date"],
                code=r["code"], hours=r["hours"], hole_section=r["hole_section"],
                formation=r["formation"], depth_m=r["depth_m"], mud_weight_sg=r["mud_weight_sg"],
                rig=r["rig"], description=r["description"], mwd=r["mwd"],
            )
            for r in self.conn.execute(sql, params)
        ]

    def ddr_features(self, field_name: str | None = None) -> list[dict[str, Any]]:
        """One row per daily report of a field (or the whole workspace): well,
        hole section, formation, rig and MWD tool, straight from the parsed
        metadata. Used to count drilling days and exposed wells per scope
        without loading full document text."""
        filters: dict[str, Any] = {"doc_type": "ddr"}
        if field_name:
            filters["field_name"] = field_name
        where, params = self._documents_where(filters)
        sql = "SELECT doc_id, well, meta FROM documents" + where
        out = []
        for r in self.conn.execute(sql, params):
            meta = json.loads(r["meta"])
            out.append({
                "doc_id": r["doc_id"], "well": r["well"],
                "hole_section": meta.get("hole_section") or "",
                "formation": meta.get("formation") or "",
                "rig": meta.get("rig") or "",
                "mwd": meta.get("mwd") or "",
            })
        return out

    def npt_summary(self, avoidable_codes: Iterable[str], **filters: Any) -> dict[str, Any]:
        """Totals over the matching ledger rows, aggregated in SQL.

        Hours are raw sums (no rounding); `avoidable_hours` counts the rows
        whose code is in `avoidable_codes`.
        """
        where, params = self._npt_where(filters)
        codes = sorted(avoidable_codes)
        avoidable = (f"COALESCE(SUM(CASE WHEN code IN ({','.join('?' * len(codes))}) "
                     "THEN hours ELSE 0 END), 0)") if codes else "0"
        sql = ("SELECT COUNT(*) AS events, COUNT(DISTINCT well) AS wells, COUNT(DISTINCT doc_id) AS reports,"
               f" COALESCE(SUM(hours), 0) AS hours, {avoidable} AS avoidable_hours"
               " FROM npt_events" + where)
        row = self.conn.execute(sql, [*codes, *params]).fetchone()
        return dict(row)

    def npt_breakdown(self, column: str, **filters: Any) -> list[dict[str, Any]]:
        """Hours, rows and wells per value of one ledger column, largest first, aggregated in SQL.

        Grouped by `doc_id`, each row also carries the report's well, date and codes.
        """
        if column not in self.NPT_COLUMNS:
            raise ValueError(f"cannot group the NPT ledger by {column!r}")
        where, params = self._npt_where(filters)
        sql = (f"SELECT {column} AS key, SUM(hours) AS hours, COUNT(*) AS events,"
               " COUNT(DISTINCT well) AS wells, MIN(well) AS well, MIN(date) AS date,"
               " GROUP_CONCAT(DISTINCT code) AS codes"
               f" FROM npt_events{where} GROUP BY {column} ORDER BY hours DESC, key")
        return [dict(r) for r in self.conn.execute(sql, params)]

    def catalog(self) -> dict[str, set[str]]:
        """Every field, well, hole section, formation and NPT code the workspace knows by name."""
        def column(sql: str) -> set[str]:
            return {str(r[0]) for r in self.conn.execute(sql) if r[0] and str(r[0]) != "UNKNOWN"}

        out = {
            "fields": column("SELECT DISTINCT field_name FROM documents")
            | column("SELECT DISTINCT field_name FROM wells"),
            "wells": column("SELECT DISTINCT well FROM documents") | column("SELECT name FROM wells"),
            "sections": column("SELECT DISTINCT hole_section FROM npt_events"),
            "formations": column("SELECT DISTINCT formation FROM npt_events"),
            "codes": column("SELECT DISTINCT code FROM npt_events"),
        }
        for row in self.conn.execute("SELECT sections, formations FROM wells"):
            out["sections"].update(s for s in json.loads(row["sections"]) if s)
            out["formations"].update(f for f in json.loads(row["formations"]) if f)
        return out

    def counts(self) -> dict[str, int]:
        out = {}
        for table in ("documents", "chunks", "npt_events", "wells", "files"):
            out[table] = self.conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        return out

    def corpus_hash(self, field_name: str | None = None) -> str:
        """sha256 over the sorted (doc_id, sha256(content)) pairs of every
        document in the store (or one field of it): changes whenever a
        document is added, removed or edited, independent of insertion order.
        """
        filters = {"field_name": field_name} if field_name else {}
        where, params = self._documents_where(filters)
        rows = self.conn.execute("SELECT doc_id, text FROM documents" + where, params)
        pairs = sorted((doc_id, hashlib.sha256(text.encode("utf-8")).hexdigest())
                       for doc_id, text in rows)
        digest = hashlib.sha256()
        for doc_id, content_hash in pairs:
            digest.update(f"{doc_id}:{content_hash}\n".encode())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_document_id(chunk_id: str) -> str:
    """The document id a chunk id (`<doc_id>#<n>`) belongs to.

    This is the only place that mapping is recorded: a per-field index's
    postings and vectors are keyed on chunk id alone, and this function
    recovers the document without a separate id-mapping file on disk.
    """
    return chunk_id.rsplit("#", 1)[0]


# A line that is its own heading: short, all upper case (ignoring digits,
# spaces and light punctuation), at least two letters. Deliberately looser
# than `quotes._heading_key` (chunk boundaries only need to be plausible
# breaks, not exactly the sections a citation quote is chosen from).
_HEADING_LINE = re.compile(r"^\d{0,2}\.?\s*[A-Z][A-Z0-9 /()\"'.,:&-]{1,78}$")


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped != stripped.upper():
        return False
    if sum(c.isalpha() for c in stripped) < 2:
        return False
    return bool(_HEADING_LINE.match(stripped))


def _split_points(text: str) -> list[int]:
    """Offsets where a chunk may start: right after a blank line, and at the
    start of a heading-like line ("NPT DETAIL", "4. LESSONS LEARNED")."""
    points: set[int] = set()
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        if line.strip() == "":
            after = end + 1
            if after <= len(text):
                points.add(after)
        elif _is_heading_line(line):
            points.add(pos)
        pos = end + 1
    return sorted(points)


def _chunk_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    n = len(text)
    if n == 0:
        return [(0, 0)]
    points = _split_points(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        limit = start + max_chars
        if limit >= n:
            spans.append((start, n))
            break
        candidates = [b for b in points if start < b <= limit]
        cut = max(candidates) if candidates else limit
        spans.append((start, cut))
        start = cut
    return spans


def chunk_document(doc_id: str, text: str, max_chars: int = CHUNK_MAX_CHARS,
                   threshold: int = CHUNK_THRESHOLD_CHARS) -> list[Chunk]:
    """Split one document's text into the chunks its per-field index is built over.

    A document of `threshold` characters or fewer stays one chunk. A longer
    one is cut into pieces of at most `max_chars`, preferring a break at a
    heading or a blank line; a stretch with no such break within `max_chars`
    is cut hard. Chunks are contiguous and cover the whole text.
    """
    if len(text) <= threshold:
        return [Chunk(f"{doc_id}#0", doc_id, 0, 0, len(text), text)]
    return [
        Chunk(f"{doc_id}#{i}", doc_id, i, start, end, text[start:end])
        for i, (start, end) in enumerate(_chunk_spans(text, max_chars))
    ]
