"""`selfcheck.run()`: the offline end-to-end check plus the network guard's positive control.

One test (`test_run_passes_end_to_end`) exercises the real pipeline (corpus generation,
ingestion, indexing, every eval suite and both field briefs) against the default seed; it is
slow (the eval suites dominate) but is the one test that actually proves the command works.
Every other test monkeypatches one step to hit a specific failure path cheaply, without paying
for a full corpus and eval run each time.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from wellbrief import netguard, riskbrief, selfcheck
from wellbrief.ingest import Coverage, SkippedFile


@pytest.fixture(autouse=True)
def _reset_guard_stats() -> None:
    """Independent of the repository's autouse `netguard.uninstall()` (which keeps the
    counters): every test here starts from a zero blocked count."""
    netguard.reset_stats()


def _fake_eval_ok(ok: bool) -> Any:
    def _run(*_args: Any, **_kwargs: Any) -> Any:
        class _Report:
            pass

        report = _Report()
        report.ok = ok  # type: ignore[attr-defined]
        return report

    return _run


def test_run_passes_end_to_end() -> None:
    netguard.install()
    result = selfcheck.run()
    assert result.failures == []
    assert result.ok is True
    assert [s.name for s in result.steps] == [
        "corpus generate", "ingest", "index", "eval", "brief verify",
    ]
    assert all(s.seconds >= 0 for s in result.steps)
    assert result.outbound_connection_attempts == 0
    assert result.guard_probe == {"attempted": 1, "blocked": 1}
    assert result.guard_error is None


def test_run_restores_a_wellbrief_home_that_was_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELLBRIEF_HOME", raising=False)
    monkeypatch.setattr(selfcheck, "_offline_checks", lambda *_a, **_k: None)
    netguard.install()
    selfcheck.run()
    assert "WELLBRIEF_HOME" not in os.environ


def test_run_restores_a_wellbrief_home_that_was_set(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ambient = tmp_path / "ambient-home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(ambient))
    monkeypatch.setattr(selfcheck, "_offline_checks", lambda *_a, **_k: None)
    netguard.install()
    selfcheck.run()
    assert os.environ["WELLBRIEF_HOME"] == str(ambient)


def test_run_uses_its_own_disposable_home_not_the_ambient_one(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ambient = tmp_path / "ambient-home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(ambient))
    seen: list[str] = []

    def _record(*_a: Any, **_k: Any) -> None:
        seen.append(os.environ["WELLBRIEF_HOME"])

    monkeypatch.setattr(selfcheck, "_offline_checks", _record)
    netguard.install()
    selfcheck.run()
    assert len(seen) == 1
    assert seen[0] != str(ambient)
    assert not ambient.exists()  # nothing was ever written to the caller's own workspace


def test_an_unexpected_outbound_attempt_during_the_offline_checks_fails_the_check(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def _connect_out(*_a: Any, **_k: Any) -> None:
        with pytest.raises(netguard.EgressBlocked):
            socket.create_connection(("192.0.2.1", 80), timeout=1)

    monkeypatch.setattr(selfcheck, "_offline_checks", _connect_out)
    netguard.install()
    result = selfcheck.run()
    assert result.outbound_connection_attempts == 1
    assert any("unexpected outbound connection attempt" in f for f in result.failures)
    assert result.ok is False
    # the guard's own positive control still runs, and still passes, despite the failure above
    assert result.guard_probe == {"attempted": 1, "blocked": 1}


def test_guard_not_installed_is_reported_as_a_failure_not_a_crash(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selfcheck, "_offline_checks", lambda *_a, **_k: None)
    assert netguard.is_installed() is False
    result = selfcheck.run()
    assert result.guard_error is not None
    assert result.guard_probe is None
    assert result.ok is False
    assert any("guard self-test" in f for f in result.failures)


def test_guard_probe_that_fails_to_block_is_reported_as_a_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`netguard.self_test()` can return without raising and still report that its own probe
    was not blocked (a guard regression, per its own docstring); `selfcheck.run()` must treat
    that the same as any other failure instead of reporting a pass."""
    monkeypatch.setattr(selfcheck, "_offline_checks", lambda *_a, **_k: None)
    monkeypatch.setattr(netguard, "self_test", lambda: {"attempted": 1, "blocked": 0})
    netguard.install()
    result = selfcheck.run()
    assert result.guard_probe == {"attempted": 1, "blocked": 0}
    assert result.guard_error is None
    assert "guard self-test did not block its own probe" in result.failures
    assert result.ok is False


def test_an_unexpected_exception_is_recorded_and_the_guard_probe_still_runs(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(selfcheck, "build_corpus", _boom)
    netguard.install()
    result = selfcheck.run()
    assert any("RuntimeError: synthetic failure" in f for f in result.failures)
    assert result.ok is False
    assert [s.name for s in result.steps] == ["corpus generate"]
    # the guard's positive control ran despite the crash in the offline checks
    assert result.guard_probe == {"attempted": 1, "blocked": 1}


def test_eval_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wellbrief.evals.runner.run", _fake_eval_ok(False))
    netguard.install()
    result = selfcheck.run()
    assert "eval: one or more gated cases failed" in result.failures
    assert result.ok is False


def test_a_field_with_nothing_ingested_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wellbrief.evals.runner.run", _fake_eval_ok(True))
    monkeypatch.setattr(selfcheck, "BRIEF_CHECKS", (("Nowhere Field", "NWF-NEXT", 1000.0),))
    netguard.install()
    result = selfcheck.run()
    assert "brief Nowhere Field: no documents ingested for this field" in result.failures
    assert result.ok is False


def test_a_brief_that_fails_verification_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wellbrief.evals.runner.run", _fake_eval_ok(True))
    monkeypatch.setattr(
        selfcheck, "verify_brief",
        lambda *_a, **_k: {"ok": False, "problems": ["p1", "p2"], "citations_checked": 3})
    netguard.install()
    result = selfcheck.run()
    assert any(
        f.startswith("brief ") and "verification failed (2 problem(s))" in f
        for f in result.failures
    )
    assert result.ok is False


def test_missing_ddr_data_is_reported_for_each_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wellbrief.evals.runner.run", _fake_eval_ok(True))

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise riskbrief.MissingDdrDataError("no daily reports for this field")

    monkeypatch.setattr(selfcheck, "build_brief", _raise)
    netguard.install()
    result = selfcheck.run()
    assert sum("no daily reports for this field" in f for f in result.failures) == len(
        selfcheck.BRIEF_CHECKS)
    assert result.ok is False


def test_ingest_row_errors_and_skips_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_coverage = Coverage(
        files_seen=1, skipped=[SkippedFile("bad.csv", "unreadable")],
        csv_row_errors=["bad.csv row 2: not a number"],
    )
    monkeypatch.setattr(selfcheck, "ingest_folder", lambda *_a, **_k: fake_coverage)
    monkeypatch.setattr("wellbrief.evals.runner.run", _fake_eval_ok(True))
    netguard.install()
    result = selfcheck.run()
    assert "ingest: bad.csv row 2: not a number" in result.failures
    assert "ingest: 1 file(s) skipped" in result.failures
    assert result.ok is False
