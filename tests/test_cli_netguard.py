"""The process-level network guard as wired into the CLI.

`netguard.py` itself (the guard's own policy and edge cases) is covered by
`test_netguard.py`. This module covers the integration: `main()` installs the
guard on every invocation, the configured LLM host is allowed only while the
`llm` narrator is active, and `status` / `eval` report the blocked count.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from wellbrief import cli, netguard


@pytest.fixture(autouse=True)
def _reset_stats() -> None:
    """Each test starts from a zero blocked count, regardless of what an earlier test in
    this file (or a preceding module) left behind: the counter survives `uninstall()` and
    is cleared only by `reset_stats()` (see `netguard.py`). The repository's `conftest.py`
    already uninstalls the guard after every test; this file additionally wants the
    count itself at zero at the start of each of its own tests."""
    netguard.reset_stats()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(h))
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    monkeypatch.delenv("WELLBRIEF_NARRATOR", raising=False)
    monkeypatch.delenv("WELLBRIEF_LLM_BASE_URL", raising=False)
    return h


def test_main_installs_the_guard(home: Path) -> None:
    assert netguard.is_installed() is False
    assert cli.main(["status"]) == 0
    assert netguard.is_installed() is True


def test_offline_narrator_allows_no_extra_host(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", "http://reports.example.test:8080")
    assert cli.main(["status"]) == 0
    assert netguard.stats()["allowed_hosts"] == []


def test_llm_narrator_allows_only_its_configured_host(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", "http://reports.example.test:8080/v1")
    assert cli.main(["--narrator", "llm", "status"]) == 0
    assert netguard.stats()["allowed_hosts"] == ["reports.example.test"]


def test_llm_narrator_via_env_var_allows_its_host(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_NARRATOR", "llm")
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", "http://192.0.2.9:9000")
    assert cli.main(["status"]) == 0
    assert netguard.stats()["allowed_hosts"] == ["192.0.2.9"]


def test_llm_narrator_without_a_configured_host_allows_nothing_extra(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELLBRIEF_LLM_BASE_URL", raising=False)
    assert cli.main(["--narrator", "llm", "status"]) == 0
    assert netguard.stats()["allowed_hosts"] == []


def test_status_reports_the_blocked_count(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status", "--json"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["outbound_connection_attempts"] == 0

    with pytest.raises(netguard.EgressBlocked):
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    assert cli.main(["status", "--json"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert after["outbound_connection_attempts"] == 1

    assert cli.main(["status"]) == 0
    assert "outbound connection attempts: 1" in capsys.readouterr().out


def test_eval_prints_the_blocked_count_at_the_end(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--suite", "original"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1] == "outbound connection attempts: 0"


def test_eval_json_output_stays_pure_json(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--suite", "original", "--json"]) == 0
    out = capsys.readouterr().out
    json.loads(out)  # a stray "outbound connection attempts" line would break this


@pytest.mark.parametrize("bad_url", [
    "http://[::1:8080",  # unbalanced IPv6 bracket: urlsplit itself raises ValueError
    "http://exam ple.com",  # a space: passes urlsplit, rejected by netguard's own host pattern
    "http://%zz.com",  # a stray '%': same, rejected by netguard's own host pattern
])
def test_a_malformed_llm_base_url_exits_cleanly_with_no_traceback(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        bad_url: str) -> None:
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", bad_url)
    assert cli.main(["--narrator", "llm", "status"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("wellbrief: ")
    assert "WELLBRIEF_LLM_BASE_URL" in err
    assert "Traceback" not in err
    assert netguard.is_installed() is False


def test_a_schemeless_llm_base_url_exits_cleanly_instead_of_silently_allowing_nothing(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", "api.example.test:8080")
    assert cli.main(["--narrator", "llm", "status"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("wellbrief: ")
    assert "WELLBRIEF_LLM_BASE_URL" in err
    assert "Traceback" not in err
