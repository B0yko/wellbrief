"""`evals.adapter`: the llm narrator built for a seed's workspace, and the verifier
fault-injection harness (`Workspace.verify_fault_injection`) against a real generated corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from fake_openai import FakeOpenAIServer, FakeResponse, completion
from wellbrief.corpus import SEED, build_corpus, write_corpus
from wellbrief.evals import adapter
from wellbrief.narrate import OfflineNarrator, verify as narrate_verify


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("adapter-corpus") / "corpus"
    write_corpus(build_corpus(seed=SEED), out)
    return out


@pytest.fixture(scope="module")
def workspace(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[adapter.Workspace]:
    ws = adapter.Workspace(corpus_dir, tmp_path_factory.mktemp("adapter-workspace") / "workspace")
    yield ws
    ws.close()


def test_workspace_defaults_to_the_offline_narrator(workspace: adapter.Workspace) -> None:
    assert isinstance(workspace.narrator, OfflineNarrator)


def test_narrator_for_offline_needs_no_egress_path() -> None:
    assert isinstance(adapter._narrator_for("offline", None), OfflineNarrator)


def test_narrator_for_llm_without_an_egress_path_raises() -> None:
    with pytest.raises(ValueError, match="egress log path"):
        adapter._narrator_for("llm", None)


def test_verify_fault_injection_catches_every_kind_on_a_rich_question(
        workspace: adapter.Workspace) -> None:
    result = workspace.verify_fault_injection("How many hours of stuck pipe on Orrindale?")
    assert result.not_caught == []
    assert set(result.kinds_tested) | set(result.skipped) == set(narrate_verify.FAULT_KINDS)
    assert result.kinds_tested  # at least one kind was actually exercised


def test_verify_fault_injection_never_uses_a_configured_llm_narrator(
        corpus_dir: Path, tmp_path: Path) -> None:
    """The baseline text is always the deterministic offline answer, whatever narrator the
    workspace itself is configured with, so the harness measures the verifier, not a model."""
    ws = adapter.Workspace(corpus_dir, tmp_path / "ws")
    try:
        result = ws.verify_fault_injection("How many hours of stuck pipe on Orrindale?")
        assert result.kinds_tested
    finally:
        ws.close()


def test_narrator_for_llm_is_given_the_workspaces_known_field_names(
        workspace: adapter.Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_narrator_for` mirrors the CLI's own `_build_narrator`: the llm narrator it builds must
    see every field name the workspace holds, not just well ids, or `eval --narrator llm` could
    never catch a hallucinated field name (check (c))."""
    from wellbrief.narrate import llm as narrate_llm

    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("WELLBRIEF_LLM_MODEL", "fake-model")
    narrator = adapter._narrator_for("llm", tmp_path / "unused-egress.jsonl",
                                     known_field_names=frozenset(workspace.store.field_names()))
    assert isinstance(narrator, narrate_llm.LlmNarrator)
    assert narrator._known_field_names == frozenset(workspace.store.field_names())
    assert narrator._known_field_names == frozenset({"Orrindale", "Vessra South"})


def test_eval_workspace_llm_narrator_catches_a_hallucinated_field_name(
        corpus_dir: Path, tmp_path: Path, fake_server: FakeOpenAIServer,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through `adapter.Workspace(narrator_name="llm")`: a reply that correctly cites
    its evidence but also names a field the question never touched (`Vessra South`, on a
    question scoped to `Orrindale`) is rejected, because the workspace's llm narrator was built
    with `known_field_names` covering both of the corpus's fields."""
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", fake_server.base_url)
    monkeypatch.setenv("WELLBRIEF_LLM_MODEL", "fake-model")
    question = "How many hours of stuck pipe on Orrindale?"

    offline_ws = adapter.Workspace(corpus_dir, tmp_path / "offline-ws")
    try:
        offline = offline_ws.ask(question)
    finally:
        offline_ws.close()
    doc_id = offline.citations[0].doc_id

    llm_ws = adapter.Workspace(corpus_dir, tmp_path / "llm-ws", narrator_name="llm",
                               egress_log_path=tmp_path / "egress.jsonl")
    try:
        fake_server.queue(FakeResponse(completion(
            f"Stuck pipe was recorded [{doc_id}]. Vessra South saw a similar issue.")))
        result = llm_ws.ask(question)
    finally:
        llm_ws.close()
    assert result.narrator_rejected is not None
    assert any("Vessra South" in r for r in result.narrator_rejected["reasons"])
