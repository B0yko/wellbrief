"""The workspace: $WELLBRIEF_HOME/<name>, and the per-field chunk indexes inside it.

A workspace is one self-contained set of ingested data and its indexes:

    $WELLBRIEF_HOME/<name>/wellbrief.db          the SQLite store (see `store.py`)
    $WELLBRIEF_HOME/<name>/fields/<slug>/bm25.json       BM25 postings over the field's chunks
    $WELLBRIEF_HOME/<name>/fields/<slug>/vectors.f32     hashing vectors over the same chunks
    $WELLBRIEF_HOME/<name>/fields/<slug>/manifest.json   document/chunk counts, corpus hash, embedder
    $WELLBRIEF_HOME/<name>/audit.jsonl                   one line per `ask` / `brief`
    $WELLBRIEF_HOME/<name>/egress.jsonl                  one line per outbound LLM call
    $WELLBRIEF_HOME/<name>/wellbrief.toml                optional site configuration

`$WELLBRIEF_HOME` defaults to `~/.wellbrief` and is overridden by the
environment variable of the same name; the workspace name defaults to
`"default"` and is chosen with the global `--workspace` flag or
`WELLBRIEF_WORKSPACE`. Nothing here talks to the network: building or
loading an index only reads the store and the files next to it.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import netguard
from .bm25 import BM25Index
from .config import BM25_B, BM25_K1, EMBED_BACKEND, RRF_K
from .egress import resolve_network_mode
from .embed import Embedder, get_embedder
from .settings import Settings
from .store import Store, chunk_document_id

if TYPE_CHECKING:
    from .search import Searcher

DEFAULT_HOME = Path.home() / ".wellbrief"
DEFAULT_WORKSPACE = "default"


def resolve_home(explicit: str | os.PathLike[str] | None = None) -> Path:
    """`$WELLBRIEF_HOME`: `explicit` (the CLI would pass one only for testing),
    else the environment variable, else `~/.wellbrief`."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("WELLBRIEF_HOME")
    return Path(env) if env else DEFAULT_HOME


def resolve_workspace_name(explicit: str | None = None) -> str:
    """The workspace name: `--workspace` (`explicit`), else `WELLBRIEF_WORKSPACE`, else `"default"`."""
    if explicit:
        return explicit
    return os.environ.get("WELLBRIEF_WORKSPACE") or DEFAULT_WORKSPACE


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def field_slug(field_name: str) -> str:
    """A filesystem-safe, lowercase directory name for a field."""
    slug = _SLUG_RE.sub("-", field_name.strip().lower()).strip("-")
    return slug or "unassigned"


def check_field_slugs(field_names: Iterable[str]) -> None:
    """Raise a clear error if two distinct field names would share a
    `fields/<slug>/` directory (for example "North Field" and "north-field"),
    instead of letting one silently overwrite the other's on-disk index.

    Called wherever a full set of fields is about to be built or listed;
    a caller that only ever touches one field at a time (`rebuild_field_index`)
    cannot detect this on its own.
    """
    by_slug: dict[str, str] = {}
    for name in field_names:
        slug = field_slug(name)
        seen = by_slug.get(slug)
        if seen is not None and seen != name:
            raise ValueError(
                f"field {seen!r} and {name!r} both map to fields/{slug}/; "
                "rename one of them so each field gets its own index directory"
            )
        by_slug[slug] = name


@dataclass(frozen=True)
class Workspace:
    """Resolved paths for one `$WELLBRIEF_HOME/<name>` directory. Nothing here touches disk by
    itself; call `open_store()` or the `build_field_index` / `save_field_index` / `load_field_index`
    functions below to actually read or write."""

    home: Path
    name: str

    @classmethod
    def resolve(cls, workspace: str | None = None, home: str | os.PathLike[str] | None = None) -> Workspace:
        return cls(home=resolve_home(home), name=resolve_workspace_name(workspace))

    @property
    def root(self) -> Path:
        return self.home / self.name

    @property
    def db_path(self) -> Path:
        return self.root / "wellbrief.db"

    @property
    def fields_dir(self) -> Path:
        return self.root / "fields"

    @property
    def audit_path(self) -> Path:
        return self.root / "audit.jsonl"

    @property
    def egress_path(self) -> Path:
        return self.root / "egress.jsonl"

    @property
    def config_path(self) -> Path:
        return self.root / "wellbrief.toml"

    def field_dir(self, field_name: str) -> Path:
        return self.fields_dir / field_slug(field_name)

    def open_store(self) -> Store:
        return Store(self.db_path)


# --------------------------------------------------------------------------
# Per-field index: BM25 + hashing vectors over one field's chunks
# --------------------------------------------------------------------------


class VectorIndex:
    """The dense side of one field's index, over chunk ids, stored as a flat float32 file."""

    def __init__(self, ids: list[str], vectors: list[list[float]], dim: int, backend: str):
        self.ids = ids
        self.vectors = vectors
        self.dim = dim
        self.backend = backend

    def search(self, query_vec: list[float], top_k: int = 20,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for chunk_id, vec in zip(self.ids, self.vectors, strict=True):
            if allowed is not None and chunk_id not in allowed:
                continue
            scored.append((chunk_id, sum(a * b for a, b in zip(query_vec, vec, strict=False))))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [(c, round(s, 6)) for c, s in scored[:top_k]]

    def write(self, path: Path) -> None:
        flat = array.array("f")
        for v in self.vectors:
            flat.extend(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            flat.tofile(fh)

    @classmethod
    def read(cls, path: Path, ids: list[str], dim: int, backend: str) -> VectorIndex:
        flat = array.array("f")
        with open(path, "rb") as fh:
            flat.fromfile(fh, len(ids) * dim)
        vectors = [list(flat[i * dim:(i + 1) * dim]) for i in range(len(ids))]
        return cls(ids, vectors, dim, backend)


@dataclass
class FieldManifest:
    """`fields/<slug>/manifest.json`: what a field's index was built from and with."""

    field: str
    documents: int
    chunks: int
    corpus_hash: str
    embedder: str
    dim: int
    build_seconds: float
    built_at: str   # UTC, "%Y-%m-%dT%H:%M:%SZ"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> FieldManifest:
        return cls(**{k: blob[k] for k in
                      ("field", "documents", "chunks", "corpus_hash", "embedder", "dim", "build_seconds",
                       "built_at")})


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class FieldIndex:
    """BM25 + hashing-vector index over one field's chunks, plus the manifest describing it."""

    def __init__(self, field: str, bm25: BM25Index, vectors: VectorIndex, manifest: FieldManifest):
        self.field = field
        self.bm25 = bm25
        self.vectors = vectors
        self.manifest = manifest

    @property
    def doc_ids(self) -> frozenset[str]:
        """Every document this index covers (recovered from its chunk ids)."""
        return frozenset(chunk_document_id(c) for c in self.bm25.doc_ids)

    @classmethod
    def build(cls, store: Store, field: str, embedder: Embedder,
             bm25_k1: float = BM25_K1, bm25_b: float = BM25_B) -> FieldIndex:
        """`bm25_k1`/`bm25_b` are `[retrieval]`'s `k1`/`b` (`settings.RetrievalSettings`),
        default the built-in `config.BM25_K1`/`BM25_B`."""
        started = time.monotonic()
        chunks = store.chunks(field_name=field)
        items = [(c.chunk_id, c.text) for c in chunks]
        bm25 = BM25Index.build(items, k1=bm25_k1, b=bm25_b)
        vecs = embedder.embed_many([t for _, t in items]) if items else []
        vectors = VectorIndex([c.chunk_id for c in chunks], vecs, embedder.dim, embedder.name)
        manifest = FieldManifest(
            field=field, documents=len({c.doc_id for c in chunks}), chunks=len(chunks),
            corpus_hash=store.corpus_hash(field), embedder=embedder.name, dim=embedder.dim,
            build_seconds=round(time.monotonic() - started, 4), built_at=_utcnow(),
        )
        return cls(field=field, bm25=bm25, vectors=vectors, manifest=manifest)


def _write_atomic(path: Path, write: Any) -> None:
    """Call `write(tmp_path)` to fill a temp file next to `path`, then `os.replace` it into place.

    `os.replace` is a single directory-entry update, so a reader that opens `path` either sees
    the old, complete file or the new, complete one -- never a truncated or half-written one.
    Without this, a concurrent `load_field_index` (any of `bm25.json`, `vectors.f32` or
    `manifest.json` opened while this function's plain `write_text`/`write` truncated it) would
    intermittently read a partial file and raise a decode error instead of a clean result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_field_index(ws: Workspace, index: FieldIndex) -> None:
    d = ws.field_dir(index.field)
    d.mkdir(parents=True, exist_ok=True)
    _write_atomic(d / "bm25.json",
                  lambda p: p.write_text(json.dumps(index.bm25.to_json()), encoding="utf-8"))
    _write_atomic(d / "vectors.f32", index.vectors.write)
    _write_atomic(d / "manifest.json",
                  lambda p: p.write_text(json.dumps(index.manifest.to_dict(), indent=2), encoding="utf-8"))


def load_field_index(ws: Workspace, field: str) -> FieldIndex | None:
    d = ws.field_dir(field)
    bm25_path, vectors_path, manifest_path = d / "bm25.json", d / "vectors.f32", d / "manifest.json"
    if not (bm25_path.exists() and vectors_path.exists() and manifest_path.exists()):
        return None
    manifest = FieldManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    bm25 = BM25Index.from_json(json.loads(bm25_path.read_text(encoding="utf-8")))
    vectors = VectorIndex.read(vectors_path, bm25.doc_ids, manifest.dim, manifest.embedder)
    return FieldIndex(field=field, bm25=bm25, vectors=vectors, manifest=manifest)


def index_files_hash(ws: Workspace, field: str) -> str:
    """sha256 over one field's on-disk index files, empty when it is not built yet
    (stands in for `index_manifest_hash` in a brief's provenance block)."""
    d = ws.field_dir(field)
    paths = [d / "bm25.json", d / "vectors.f32", d / "manifest.json"]
    if not all(p.exists() for p in paths):
        return ""
    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Building indexes without a workspace directory (in-memory: tests, evals)
# --------------------------------------------------------------------------


def build_field_indexes(store: Store, embed_backend: str = EMBED_BACKEND,
                        bm25_k1: float = BM25_K1, bm25_b: float = BM25_B) -> dict[str, FieldIndex]:
    """One in-memory `FieldIndex` per field the store currently holds documents for."""
    fields = store.field_names()
    check_field_slugs(fields)
    embedder = get_embedder(embed_backend)
    return {field: FieldIndex.build(store, field, embedder, bm25_k1, bm25_b) for field in fields}


def build_searcher(store: Store, embed_backend: str = EMBED_BACKEND, bm25_k1: float = BM25_K1,
                   bm25_b: float = BM25_B, rrf_k: int = RRF_K) -> Searcher:
    """A `search.Searcher` over freshly built, in-memory per-field indexes.

    Import of `search` is local to avoid a cycle (`search` imports `FieldIndex`
    from this module; nothing here needs to import `search` at module load
    time, only inside this one function).
    """
    from .search import Searcher as _Searcher

    indexes = build_field_indexes(store, embed_backend, bm25_k1, bm25_b)
    return _Searcher(store, indexes, get_embedder(embed_backend), rrf_k=rrf_k)


# --------------------------------------------------------------------------
# Persisted indexes: the CLI's `index` command and query-time auto-build
# --------------------------------------------------------------------------


def rebuild_field_index(ws: Workspace, store: Store, field: str, embed_backend: str = EMBED_BACKEND,
                        bm25_k1: float = BM25_K1, bm25_b: float = BM25_B) -> FieldIndex:
    """Build one field's index from the store and persist it, unconditionally
    (`wellbrief index` rebuilds even when the on-disk copy is already fresh)."""
    index = FieldIndex.build(store, field, get_embedder(embed_backend), bm25_k1, bm25_b)
    save_field_index(ws, index)
    return index


_index_locks: dict[Path, threading.Lock] = {}
_index_locks_guard = threading.Lock()


def _index_lock(ws: Workspace) -> threading.Lock:
    """One `threading.Lock` per workspace root, created on first use and kept for the life of
    the process. `ensure_field_indexes` holds it across its whole read-or-rebuild-and-save
    sequence, so two threads serving concurrent requests against the same never-indexed (or
    stale) workspace -- `ThreadingHTTPServer` gives every request its own thread -- never both
    decide a field needs rebuilding and race to write its index files at once.
    """
    root = ws.root
    with _index_locks_guard:
        lock = _index_locks.get(root)
        if lock is None:
            lock = threading.Lock()
            _index_locks[root] = lock
        return lock


def ensure_field_indexes(ws: Workspace, store: Store, embed_backend: str = EMBED_BACKEND,
                         bm25_k1: float = BM25_K1, bm25_b: float = BM25_B) -> dict[str, FieldIndex]:
    """Every field's index, loaded from disk when it is present, matches the store's current
    corpus hash and embedder, and was built with the same BM25 `k1`/`b` (`[retrieval]`), rebuilt
    (and persisted) otherwise.

    Used to wire `ask` and `brief` so they work right after `ingest` without
    a separate `index` step, while `wellbrief index` stays the explicit,
    unconditional rebuild. Serialized per workspace (`_index_lock`) so concurrent callers never
    race on the same on-disk files; `save_field_index`'s writes are also atomic on their own, as
    a second, independent guard against a reader ever observing a partial file.
    """
    fields = store.field_names()
    check_field_slugs(fields)
    embedder = get_embedder(embed_backend)
    out: dict[str, FieldIndex] = {}
    with _index_lock(ws):
        for field in fields:
            existing = load_field_index(ws, field)
            fresh = (existing is not None and existing.manifest.embedder == embedder.name
                    and existing.manifest.corpus_hash == store.corpus_hash(field)
                    and existing.bm25.k1 == bm25_k1 and existing.bm25.b == bm25_b)
            out[field] = existing if fresh and existing is not None else rebuild_field_index(
                ws, store, field, embed_backend, bm25_k1, bm25_b)
    return out


# --------------------------------------------------------------------------
# Shared by the CLI and the HTTP API: one query needs one `Searcher`, and
# `status` (`wellbrief status`, `GET /api/status`) needs one payload.
# --------------------------------------------------------------------------


def wire_searcher(ws: Workspace, store: Store, settings: Settings) -> Searcher:
    """A `search.Searcher` over `ws`'s per-field indexes, loaded from disk or rebuilt as
    `ensure_field_indexes` decides, using `settings.retrieval`'s BM25 and RRF knobs.

    The one place `ask` and `brief` -- the CLI's own commands (`cli._wire`) and the HTTP
    API's handlers (`server.py`) alike -- turn a resolved `Settings` into a `Searcher`, so
    a query answered through either path is retrieved the same way.
    """
    from .search import Searcher as _Searcher

    indexes = ensure_field_indexes(ws, store, bm25_k1=settings.retrieval.k1, bm25_b=settings.retrieval.b)
    return _Searcher(store, indexes, get_embedder(), rrf_k=settings.retrieval.rrf_k)


def _index_age_seconds(built_at: str) -> float:
    built = datetime.strptime(built_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (datetime.now(UTC) - built).total_seconds()


def status_payload(ws: Workspace, store: Store, narrator: str) -> dict[str, Any]:
    """What is loaded in `ws`, and how fresh its indexes are: the payload `wellbrief status
    --json` and `GET /api/status` both return, built once here so the two never drift apart.

    `narrator` is the name (`"offline"` or `"llm"`) the network mode is reported for; it is
    read from `WELLBRIEF_LLM_BASE_URL` when the narrator is `llm` (`egress.resolve_network_mode`),
    not from anything this function is passed, since neither the CLI nor a long-running server
    process has looked at that variable more than once already.
    """
    fields = store.field_names()
    check_field_slugs(fields)
    rows = []
    for f in fields:
        counts = store.field_counts(f)
        current_hash = store.corpus_hash(f)
        built = load_field_index(ws, f)
        manifest = built.manifest if built else None
        rows.append({
            "field": f,
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "npt_events": counts["npt_events"],
            "corpus_hash": current_hash,
            "embedder": manifest.embedder if manifest else None,
            "index_built_at": manifest.built_at if manifest else None,
            "index_age_seconds": round(_index_age_seconds(manifest.built_at), 1) if manifest else None,
            "index_stale": manifest is None or manifest.corpus_hash != current_hash,
        })
    return {
        "workspace": ws.name,
        "home": str(ws.home),
        "database": str(ws.db_path),
        "network_mode": resolve_network_mode(narrator),
        "outbound_connection_attempts": netguard.stats()["blocked"],
        "fields": rows,
    }
