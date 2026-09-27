"""`wellbrief demo`: one command from an empty `$WELLBRIEF_HOME` to a running web UI.

`prepare()` generates the synthetic corpus, ingests it and builds its per-field indexes inside
a workspace always named `"demo"` (`WORKSPACE_NAME`), regardless of `--workspace` or
`$WELLBRIEF_WORKSPACE`: the point of `demo` is that the same command always finds, or rebuilds,
the same known workspace. `cli.cmd_demo` then starts the same local server `wellbrief serve` does
(`server.create_server`) and opens a browser on it, unless `--no-browser`.

Repeated runs reuse the existing workspace instead of regenerating it: `prepare()` writes a small
marker file (`demo.json`, next to the workspace's database) recording the wellbrief version, the
seed, the output format and a content hash of the generated corpus (`_content_hash`, over every
document's id and text -- independent of which file format it was written as, so it is stable
across `--formats mixed`/`txt`); the next call reuses the workspace only when every one of those,
plus the format itself (which format a corpus was written in can still affect how a reader parses
it back, even when the underlying text is identical), still matches and the workspace's database
is present. `--rebuild` skips the check and always regenerates.

`wells_per_field_from_env()` reads a test-only knob, `WELLBRIEF_DEMO_WELLS_PER_FIELD`: no public
CLI flag sets it, and the real `demo` command never sets it either, so an ordinary run always
demonstrates the full corpus. It exists only so the demo integration test can shrink the
generated corpus to a couple of wells per field, keeping `demo --no-browser` fast in the test
suite; both fields stay represented (`brief` needs at least one well from each), and the shrink
happens before the corpus is written or hashed, so it is a normal `prepare()` call from every
other module's point of view.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .corpus import Corpus, build_corpus, write_corpus
from .ingest import ingest_folder
from .settings import resolve_settings
from .store import Store
from .workspace import Workspace, check_field_slugs, rebuild_field_index

WORKSPACE_NAME = "demo"
_MARKER_NAME = "demo.json"
_CORPUS_DIRNAME = "corpus"


def wells_per_field_from_env() -> int | None:
    """`WELLBRIEF_DEMO_WELLS_PER_FIELD` as a positive int, else `None` (the default: every well).
    See the module docstring; this is a test-only knob, not a CLI flag."""
    raw = os.environ.get("WELLBRIEF_DEMO_WELLS_PER_FIELD")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _reduce_corpus(generated: Corpus, wells_per_field: int) -> Corpus:
    """`generated` with only the first `wells_per_field` wells of each field kept (by generation
    index, so the same wells every time), and only the documents belonging to a kept well."""
    kept_names: set[str] = set()
    kept_wells = []
    for spec in generated.fields:
        field_wells = sorted((w for w in generated.wells if w.spec.name == spec.name), key=lambda w: w.index)
        for w in field_wells[:wells_per_field]:
            kept_wells.append(w)
            kept_names.add(w.name)
    kept_documents = [d for d in generated.documents if d.well in kept_names]
    return replace(generated, wells=kept_wells, documents=kept_documents)


def _content_hash(generated: Corpus, wells_per_field: int | None) -> str:
    """sha256 over every document's (id, sha256(text)), independent of the output file format
    (`RenderedDocument.text` is the same regardless of which writer will encode it): the cache
    key `prepare()` compares between runs to decide whether the corpus actually changed."""
    digest = hashlib.sha256()
    digest.update(f"seed={generated.seed} scale={generated.scale} "
                 f"wells_per_field={wells_per_field}\n".encode())
    pairs = sorted((d.doc_id, hashlib.sha256(d.text.encode("utf-8")).hexdigest())
                  for d in generated.documents)
    for doc_id, content_hash in pairs:
        digest.update(f"{doc_id}:{content_hash}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class _Marker:
    version: str
    seed: int
    formats: str
    wells_per_field: int | None
    corpus_hash: str
    built_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> _Marker:
        return cls(version=blob["version"], seed=blob["seed"], formats=blob["formats"],
                   wells_per_field=blob.get("wells_per_field"), corpus_hash=blob["corpus_hash"],
                   built_at=blob.get("built_at", ""))


def _marker_path(ws: Workspace) -> Path:
    return ws.root / _MARKER_NAME


def _read_marker(ws: Workspace) -> _Marker | None:
    path = _marker_path(ws)
    if not path.is_file():
        return None
    try:
        return _Marker.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, KeyError, ValueError):
        return None


def _write_marker(ws: Workspace, marker: _Marker) -> None:
    path = _marker_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{_MARKER_NAME}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(marker.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class DemoBuild:
    """What `prepare()` did: `reused` when the existing workspace already matched and was kept
    as is, `seconds` timing the whole call (a fresh build's generate+ingest+index, or a reused
    workspace's much shorter marker check)."""

    workspace: Workspace
    reused: bool
    seed: int
    formats: str
    corpus_hash: str
    fields: tuple[str, ...]
    documents: int
    seconds: float


def _field_totals(store: Store) -> tuple[tuple[str, ...], int]:
    fields = tuple(sorted(store.field_names()))
    documents = sum(store.field_counts(f)["documents"] for f in fields)
    return fields, documents


def prepare(ws: Workspace, *, seed: int, formats: str, rebuild: bool) -> DemoBuild:
    """Make sure `ws` (always the `demo` workspace) holds the generated corpus for `seed` and
    `formats`, ingested and indexed, reusing it as is when `--rebuild` was not given and the
    workspace already matches (see the module docstring). Returns what it found or built.
    """
    started = time.monotonic()
    wells_per_field = wells_per_field_from_env()
    generated = build_corpus(seed=seed, scale=1)
    if wells_per_field is not None:
        generated = _reduce_corpus(generated, wells_per_field)
    content_hash = _content_hash(generated, wells_per_field)

    marker = _read_marker(ws)
    reusable = (not rebuild and marker is not None and marker.version == __version__
               and marker.seed == seed and marker.formats == formats
               and marker.wells_per_field == wells_per_field
               and marker.corpus_hash == content_hash
               and ws.db_path.is_file())
    if reusable:
        store = ws.open_store()
        try:
            fields, documents = _field_totals(store)
        finally:
            store.close()
        return DemoBuild(ws, True, seed, formats, content_hash, fields, documents,
                         round(time.monotonic() - started, 3))

    if ws.root.exists():
        shutil.rmtree(ws.root)
    corpus_dir = ws.root / _CORPUS_DIRNAME
    write_corpus(generated, corpus_dir, formats)

    store = ws.open_store()
    try:
        ingest_folder(store, corpus_dir, workspace_root=ws.root)
        known = store.field_names()
        check_field_slugs(known)
        settings = resolve_settings(workspace_root=ws.root)
        for f in sorted(known):
            rebuild_field_index(ws, store, f, bm25_k1=settings.retrieval.k1, bm25_b=settings.retrieval.b)
        fields, documents = _field_totals(store)
    finally:
        store.close()

    _write_marker(ws, _Marker(version=__version__, seed=seed, formats=formats,
                              wells_per_field=wells_per_field, corpus_hash=content_hash,
                              built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")))
    return DemoBuild(ws, False, seed, formats, content_hash, fields, documents,
                     round(time.monotonic() - started, 3))


__all__ = ["WORKSPACE_NAME", "DemoBuild", "prepare", "wells_per_field_from_env"]
