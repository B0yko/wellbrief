"""The `llm` narrator wired into the CLI: `ask`/`brief` calling a fake OpenAI-compatible server,
`status`'s network mode, the audit log recording a narrator rejection, and `eval --narrator llm`
actually reaching the server (never a paid API).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from fake_openai import FakeOpenAIServer, FakeResponse, completion
from wellbrief import audit, cli


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(h))
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    monkeypatch.delenv("WELLBRIEF_NARRATOR", raising=False)
    monkeypatch.delenv("WELLBRIEF_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("WELLBRIEF_ALLOW_REMOTE", raising=False)
    return h


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("llm-cli-corpus") / "corpus"
    assert cli.main(["corpus", "generate", "--out", str(out), "--scale", "1"]) == 0
    return out


@pytest.fixture
def ingested(home: Path, corpus_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("WELLBRIEF_LLM_BASE_URL", raising=False)
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    return home


def _llm_env(monkeypatch: pytest.MonkeyPatch, server: FakeOpenAIServer) -> None:
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", server.base_url)
    monkeypatch.setenv("WELLBRIEF_LLM_MODEL", "fake-model")


def _a_real_citation(ingested: Path, question: str) -> str:
    """A document id this workspace's offline answer to `question` actually cites, so a test can
    build a fake reply that verifies (cite anything else and the verifier would reject it)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["ask", question, "--json"]) == 0
    payload = json.loads(buf.getvalue())
    return str(payload["citations"][0]["doc_id"])


def test_ask_happy_path_prints_the_servers_text(
        ingested: Path, fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    question = "What happened on ORD-101?"
    doc_id = _a_real_citation(ingested, question)
    capsys.readouterr()
    _llm_env(monkeypatch, fake_server)
    fake_server.queue(FakeResponse(completion(f"Stuck pipe was recorded on Orrindale wells [{doc_id}].")))
    assert cli.main(["--narrator", "llm", "ask", question]) == 0
    out = capsys.readouterr().out
    assert f"Stuck pipe was recorded on Orrindale wells [{doc_id}]." in out
    assert len(fake_server.requests) == 1


def test_ask_hallucination_shows_a_banner_and_is_recorded_in_json(
        ingested: Path, fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    _llm_env(monkeypatch, fake_server)
    fake_server.queue(FakeResponse(completion("Everything is fine [DDR-ZZZ-999-001].")))
    assert cli.main(["--narrator", "llm", "ask", "What happened on ORD-101?", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["narrator_rejected"] is not None
    assert payload["narrator_rejected"]["text"] == "Everything is fine [DDR-ZZZ-999-001]."
    assert any("not in the evidence" in r for r in payload["narrator_rejected"]["reasons"])
    assert "language-model narrative was rejected" in payload["text"]


def test_ask_hallucination_is_reported_in_the_audit_log(
        ingested: Path, fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    _llm_env(monkeypatch, fake_server)
    fake_server.queue(FakeResponse(completion("Everything is fine [DDR-ZZZ-999-001].")))
    assert cli.main(["--narrator", "llm", "ask", "What happened on ORD-101?"]) == 0
    from wellbrief.workspace import Workspace
    ws = Workspace.resolve(None)
    (record,) = audit.read_lines(ws.audit_path)
    assert record["verification"]["ok"] is False
    assert record["verification"]["reasons"] >= 1


def test_status_reports_local_llm_when_the_llm_narrator_is_configured(
        ingested: Path, fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["network_mode"] == "offline"

    _llm_env(monkeypatch, fake_server)
    assert cli.main(["--narrator", "llm", "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["network_mode"] == "local-llm"


def test_eval_narrator_llm_actually_calls_the_fake_server(
        ingested: Path, fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """`eval --narrator llm` is exercised only against a fake server in tests: this proves the
    wiring reaches it (never a paid API), whatever each case then makes of the reply."""
    _llm_env(monkeypatch, fake_server)
    assert cli.main(["--narrator", "llm", "eval", "--suite", "original"]) in (0, 1)
    assert len(fake_server.requests) > 0
