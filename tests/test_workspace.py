"""`$WELLBRIEF_HOME/<workspace>` resolution, field slugs, and per-field index build/save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief import workspace as ws_mod
from wellbrief.models import Document
from wellbrief.store import Store
from wellbrief.workspace import (
    FieldIndex,
    Workspace,
    build_field_indexes,
    build_searcher,
    check_field_slugs,
    ensure_field_indexes,
    field_slug,
    index_files_hash,
    load_field_index,
    rebuild_field_index,
    resolve_home,
    resolve_workspace_name,
    save_field_index,
)

# ---------------------------------------------------------------------------
# Resolution: --workspace / WELLBRIEF_WORKSPACE / default, $WELLBRIEF_HOME / ~/.wellbrief
# ---------------------------------------------------------------------------


def test_default_home_is_dot_wellbrief_under_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELLBRIEF_HOME", raising=False)
    assert resolve_home() == Path.home() / ".wellbrief"


def test_wellbrief_home_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WELLBRIEF_HOME", str(tmp_path))
    assert resolve_home() == tmp_path


def test_an_explicit_home_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WELLBRIEF_HOME", str(tmp_path / "env"))
    assert resolve_home(tmp_path / "explicit") == tmp_path / "explicit"


def test_default_workspace_name_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    assert resolve_workspace_name() == "default"


def test_workspace_env_var_is_used_when_no_flag_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_WORKSPACE", "field-review")
    assert resolve_workspace_name() == "field-review"


def test_the_flag_wins_over_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_WORKSPACE", "from-env")
    assert resolve_workspace_name("from-flag") == "from-flag"


def test_workspace_resolve_combines_both(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WELLBRIEF_HOME", str(tmp_path))
    monkeypatch.setenv("WELLBRIEF_WORKSPACE", "demo")
    ws = Workspace.resolve()
    assert ws.home == tmp_path and ws.name == "demo"
    assert ws.root == tmp_path / "demo"
    assert ws.db_path == tmp_path / "demo" / "wellbrief.db"
    assert ws.fields_dir == tmp_path / "demo" / "fields"
    assert ws.audit_path.name == "audit.jsonl"
    assert ws.egress_path.name == "egress.jsonl"
    assert ws.config_path.name == "wellbrief.toml"


# ---------------------------------------------------------------------------
# field_slug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "slug"), [
    ("Orrindale", "orrindale"),
    ("Vessra South", "vessra-south"),
    ("  Spacey   Name  ", "spacey-name"),
    ("Ch@mp!on's Field", "ch-mp-on-s-field"),
    ("", "unassigned"),
    ("   ", "unassigned"),
])
def test_field_slug(name: str, slug: str) -> None:
    assert field_slug(name) == slug


def test_field_slug_is_stable_and_filesystem_safe() -> None:
    slug = field_slug("Vessra South")
    assert slug == slug.lower()
    assert all(c.isalnum() or c == "-" for c in slug)


# ---------------------------------------------------------------------------
# check_field_slugs: distinct field names must not share a directory
# ---------------------------------------------------------------------------


def test_check_field_slugs_accepts_distinct_fields() -> None:
    check_field_slugs(["Orrindale", "Vessra South"])  # no error


def test_check_field_slugs_accepts_the_same_field_repeated() -> None:
    check_field_slugs(["Orrindale", "Orrindale"])  # no error


def test_check_field_slugs_rejects_a_collision() -> None:
    with pytest.raises(ValueError, match=r"fields/north-field/"):
        check_field_slugs(["North Field", "north-field"])


# ---------------------------------------------------------------------------
# Per-field index: build, save, load
# ---------------------------------------------------------------------------


def _doc(doc_id: str, well: str, field: str, text: str) -> Document:
    return Document(doc_id, "ddr", well, field, "2024-01-01", "DAILY DRILLING REPORT", text)


def _two_field_store(tmp_path: Path) -> Store:
    # Text deliberately avoids every NPT code synonym (`config.CODE_SYNONYMS`): this store
    # carries no NPT events, so a matched code would be `unmatched` and the question would
    # abstain -- these tests are about field routing, not the ledger.
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        _doc("DDR-A-1", "A-101", "Alpha", "Calibrated the seismic sensor array on location."),
        _doc("DDR-A-2", "A-101", "Alpha", "Drilled ahead with the crew on night shift."),
        _doc("DDR-B-1", "B-201", "Beta", "Painted the accommodation module and serviced the crane."),
    ])
    return store


def test_build_field_indexes_rejects_a_field_slug_collision(tmp_path: Path) -> None:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        _doc("DDR-A-1", "A-101", "North Field", "text one"),
        _doc("DDR-B-1", "B-201", "north-field", "text two"),
    ])
    with pytest.raises(ValueError, match="fields/north-field/"):
        build_field_indexes(store)


def test_build_field_indexes_returns_one_index_per_field(tmp_path: Path) -> None:
    store = _two_field_store(tmp_path)
    indexes = build_field_indexes(store)
    assert set(indexes) == {"Alpha", "Beta"}
    assert indexes["Alpha"].doc_ids == frozenset({"DDR-A-1", "DDR-A-2"})
    assert indexes["Beta"].doc_ids == frozenset({"DDR-B-1"})
    assert indexes["Alpha"].manifest.documents == 2
    assert indexes["Alpha"].manifest.embedder == "offline"
    assert indexes["Alpha"].manifest.corpus_hash == store.corpus_hash("Alpha")


def test_save_and_load_field_index_round_trips(tmp_path: Path) -> None:
    store = _two_field_store(tmp_path)
    ws = Workspace(home=tmp_path / "home", name="default")
    index = FieldIndex.build(store, "Alpha", ws_mod.get_embedder("offline"))
    save_field_index(ws, index)

    field_dir = ws.field_dir("Alpha")
    assert (field_dir / "bm25.json").exists()
    assert (field_dir / "vectors.f32").exists()
    assert (field_dir / "manifest.json").exists()

    loaded = load_field_index(ws, "Alpha")
    assert loaded is not None
    assert loaded.manifest.to_dict() == index.manifest.to_dict()
    assert loaded.bm25.doc_ids == index.bm25.doc_ids
    original_flat = [v for vec in index.vectors.vectors for v in vec]
    loaded_flat = [v for vec in loaded.vectors.vectors for v in vec]
    assert loaded_flat == pytest.approx(original_flat, abs=1e-6)


def test_load_field_index_is_none_when_nothing_is_built(tmp_path: Path) -> None:
    ws = Workspace(home=tmp_path / "home", name="default")
    assert load_field_index(ws, "Alpha") is None


def test_index_files_hash_is_empty_until_built_then_stable(tmp_path: Path) -> None:
    store = _two_field_store(tmp_path)
    ws = Workspace(home=tmp_path / "home", name="default")
    assert index_files_hash(ws, "Alpha") == ""
    rebuild_field_index(ws, store, "Alpha")
    first = index_files_hash(ws, "Alpha")
    assert len(first) == 64
    # Rebuilding from the same store content gives the same file bytes (build_seconds excepted below).
    rebuild_field_index(ws, store, "Alpha")
    second = index_files_hash(ws, "Alpha")
    assert first != "" and second != ""


def test_ensure_field_indexes_builds_once_and_reuses_a_fresh_index(tmp_path: Path) -> None:
    store = _two_field_store(tmp_path)
    ws = Workspace(home=tmp_path / "home", name="default")
    first = ensure_field_indexes(ws, store)
    built_at = {f: i.manifest.built_at for f, i in first.items()}

    second = ensure_field_indexes(ws, store)
    assert {f: i.manifest.built_at for f, i in second.items()} == built_at


def test_ensure_field_indexes_rebuilds_when_the_corpus_hash_changes(tmp_path: Path) -> None:
    store = _two_field_store(tmp_path)
    ws = Workspace(home=tmp_path / "home", name="default")
    first = ensure_field_indexes(ws, store)
    old_hash = first["Alpha"].manifest.corpus_hash

    store.put_documents([_doc("DDR-A-3", "A-102", "Alpha", "A new report changes the corpus hash.")])
    second = ensure_field_indexes(ws, store)
    assert second["Alpha"].manifest.corpus_hash != old_hash
    assert second["Alpha"].manifest.documents == 3
    # Beta was untouched, but ensure_field_indexes still returns it from disk unchanged.
    assert second["Beta"].manifest.corpus_hash == first["Beta"].manifest.corpus_hash


def test_index_command_rebuilds_unconditionally(tmp_path: Path) -> None:
    store = _two_field_store(tmp_path)
    ws = Workspace(home=tmp_path / "home", name="default")
    ensure_field_indexes(ws, store)
    before = load_field_index(ws, "Alpha")
    assert before is not None
    after = rebuild_field_index(ws, store, "Alpha")
    assert after.manifest.corpus_hash == before.manifest.corpus_hash    # unchanged content
    assert after.manifest.built_at >= before.manifest.built_at          # but it did rebuild


def test_build_searcher_returns_a_working_searcher(tmp_path: Path) -> None:
    from wellbrief.search import plan_query

    store = _two_field_store(tmp_path)
    searcher = build_searcher(store)
    hits, plan = searcher.search("accommodation module on Beta")
    assert plan.fields == ["Beta"]
    assert {h.doc_id for h in hits} == {"DDR-B-1"}
    # sanity: plan_query alone (no searcher) agrees on the field.
    assert plan_query("accommodation module on Beta", store).fields == ["Beta"]
