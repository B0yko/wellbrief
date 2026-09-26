"""SQLite store plus the on-disk retrieval index.

One SQLite file holds the corpus, the parsed NPT ledger and the well
register. Sidecar files next to it hold the BM25 postings and the dense
vectors. Together they are the whole state of an index: nothing else is read
at query time, and no network access is needed.
"""

from __future__ import annotations

import array
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .config import DB_PATH, EMBED_BACKEND
from .embed import Embedder, get_embedder
from .models import Document, NptEvent, Well

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    doc_type   TEXT NOT NULL,
    well       TEXT NOT NULL,
    field_name TEXT NOT NULL,
    date       TEXT NOT NULL,
    title      TEXT NOT NULL,
    text       TEXT NOT NULL,
    meta       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_well  ON documents(well);
CREATE INDEX IF NOT EXISTS ix_documents_field ON documents(field_name);
CREATE INDEX IF NOT EXISTS ix_documents_type  ON documents(doc_type);

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
    description   TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


class Store:
    def __init__(self, db_path: Path | str = DB_PATH):
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
        for table in ("npt_events", "documents", "wells", "kv"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    # -- writes -----------------------------------------------------------
    def put_documents(self, docs: Iterable[Document]) -> int:
        rows = [
            (d.doc_id, d.doc_type, d.well, d.field_name, d.date, d.title, d.text, json.dumps(d.meta))
            for d in docs
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()
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
        rows = [
            (e.doc_id, e.well, e.field_name, e.date, e.code, e.hours, e.hole_section,
             e.formation, e.depth_m, e.mud_weight_sg, e.rig, e.description)
            for e in events
        ]
        self.conn.executemany(
            "INSERT INTO npt_events (doc_id, well, field_name, date, code, hours, hole_section,"
            " formation, depth_m, mud_weight_sg, rig, description) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def set_kv(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row["v"]) if row else default

    # -- reads ------------------------------------------------------------
    @staticmethod
    def _doc(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"], doc_type=row["doc_type"], well=row["well"],
            field_name=row["field_name"], date=row["date"], title=row["title"],
            text=row["text"], meta=json.loads(row["meta"]),
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
    NPT_COLUMNS = ("field_name", "well", "code", "hole_section", "formation", "rig", "doc_id")

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
                rig=r["rig"], description=r["description"],
            )
            for r in self.conn.execute(sql, params)
        ]

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
        for table in ("documents", "npt_events", "wells"):
            out[table] = self.conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        return out


# --------------------------------------------------------------------------
# Retrieval index
# --------------------------------------------------------------------------

class VectorIndex:
    """Dense side of the hybrid, stored as a flat float32 file."""

    def __init__(self, doc_ids: list[str], vectors: list[list[float]], dim: int, backend: str):
        self.doc_ids = doc_ids
        self.vectors = vectors
        self.dim = dim
        self.backend = backend

    @classmethod
    def build(cls, items: list[tuple[str, str]], embedder: Embedder) -> VectorIndex:
        doc_ids = [d for d, _ in items]
        vectors = embedder.embed_many([t for _, t in items])
        dim = len(vectors[0]) if vectors else embedder.dim
        return cls(doc_ids, vectors, dim, embedder.name)

    def save(self, base: Path) -> None:
        base.parent.mkdir(parents=True, exist_ok=True)
        flat = array.array("f")
        for v in self.vectors:
            flat.extend(v)
        with open(base.with_suffix(".vec"), "wb") as fh:
            flat.tofile(fh)
        base.with_suffix(".vecmeta.json").write_text(
            json.dumps({"doc_ids": self.doc_ids, "dim": self.dim, "backend": self.backend}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, base: Path) -> VectorIndex:
        meta = json.loads(base.with_suffix(".vecmeta.json").read_text(encoding="utf-8"))
        dim = meta["dim"]
        flat = array.array("f")
        with open(base.with_suffix(".vec"), "rb") as fh:
            flat.fromfile(fh, len(meta["doc_ids"]) * dim)
        vectors = [list(flat[i * dim:(i + 1) * dim]) for i in range(len(meta["doc_ids"]))]
        return cls(meta["doc_ids"], vectors, dim, meta["backend"])

    def search(self, query_vec: list[float], top_k: int = 20,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for doc_id, vec in zip(self.doc_ids, self.vectors, strict=True):
            if allowed is not None and doc_id not in allowed:
                continue
            scored.append((doc_id, sum(a * b for a, b in zip(query_vec, vec, strict=False))))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [(d, round(s, 6)) for d, s in scored[:top_k]]


def index_paths(db_path: Path) -> tuple[Path, Path]:
    base = db_path.with_suffix("")
    return base.with_name(base.name + "_bm25").with_suffix(".json"), base.with_name(base.name + "_vec")


def build_indexes(store: Store, embed_backend: str = EMBED_BACKEND) -> tuple[BM25Index, VectorIndex]:
    docs = store.documents()
    items = [(d.doc_id, f"{d.title}\n{d.text}") for d in docs]
    bm25 = BM25Index.build(items)
    vectors = VectorIndex.build(items, get_embedder(embed_backend))

    bm25_path, vec_base = index_paths(store.db_path)
    bm25_path.write_text(json.dumps(bm25.to_json()), encoding="utf-8")
    vectors.save(vec_base)
    store.set_kv("index_backend", vectors.backend)
    store.set_kv("index_docs", len(items))
    return bm25, vectors


def load_indexes(store: Store) -> tuple[BM25Index, VectorIndex]:
    bm25_path, vec_base = index_paths(store.db_path)
    if not bm25_path.exists() or not vec_base.with_suffix(".vec").exists():
        return build_indexes(store)
    bm25 = BM25Index.from_json(json.loads(bm25_path.read_text(encoding="utf-8")))
    vectors = VectorIndex.load(vec_base)
    return bm25, vectors
