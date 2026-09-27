"""`wellbrief selfcheck`: the CLI command wired to `selfcheck.run()`.

`selfcheck.py` itself is covered by `test_selfcheck.py`; this module covers the CLI's own
formatting and exit code, with `selfcheck.run()` stubbed for every test but the last, which
runs the real, slow, end-to-end command exactly as a caller would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief import cli, netguard, selfcheck


@pytest.fixture(autouse=True)
def _reset_guard_stats() -> None:
    netguard.reset_stats()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(h))
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    monkeypatch.delenv("WELLBRIEF_NARRATOR", raising=False)
    monkeypatch.delenv("WELLBRIEF_LLM_BASE_URL", raising=False)
    return h


def test_selfcheck_is_a_registered_subcommand_with_no_extra_flags() -> None:
    args = cli.build_parser().parse_args(["selfcheck"])
    assert args.command == "selfcheck"
    assert args.func is cli.cmd_selfcheck


def _passing_result() -> selfcheck.SelfCheckResult:
    return selfcheck.SelfCheckResult(
        steps=[
            selfcheck.StepTiming("corpus generate", 0.1),
            selfcheck.StepTiming("ingest", 0.2),
            selfcheck.StepTiming("index", 0.05),
            selfcheck.StepTiming("eval", 3.4),
            selfcheck.StepTiming("brief verify", 0.01),
        ],
        failures=[],
        outbound_connection_attempts=0,
        guard_probe={"attempted": 1, "blocked": 1},
    )


def test_cmd_selfcheck_prints_pass_and_exits_0(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli.selfcheck_mod, "run", _passing_result)
    assert cli.main(["selfcheck"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert "outbound connection attempts: 0" in lines
    assert "guard self-test: blocked 1/1" in lines
    assert lines[-1] == "selfcheck: PASS"
    for step in ("corpus generate", "ingest", "index", "eval", "brief verify"):
        assert any(step in line for line in lines)


def test_cmd_selfcheck_prints_fail_and_exits_1_on_a_failure(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    result = _passing_result()
    result.failures.append("eval: one or more gated cases failed")
    monkeypatch.setattr(cli.selfcheck_mod, "run", lambda: result)
    assert cli.main(["selfcheck"]) == 1
    captured = capsys.readouterr()
    assert "[failure] eval: one or more gated cases failed" in captured.err.splitlines()
    assert captured.out.splitlines()[-1] == "selfcheck: FAIL"


def test_cmd_selfcheck_reports_a_guard_error_cleanly(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    result = selfcheck.SelfCheckResult(
        failures=["guard self-test did not run: netguard is not installed"],
        guard_probe=None, guard_error="netguard is not installed",
    )
    monkeypatch.setattr(cli.selfcheck_mod, "run", lambda: result)
    assert cli.main(["selfcheck"]) == 1
    captured = capsys.readouterr()
    assert "guard self-test: error: netguard is not installed" in captured.err
    assert "Traceback" not in captured.err
    assert not any(line.startswith("guard self-test: blocked") for line in captured.out.splitlines())


def test_selfcheck_end_to_end_through_the_cli(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The real command: no stubs. Slow (dominated by the eval suites), but the one test that
    actually proves `wellbrief selfcheck` passes end to end."""
    assert cli.main(["selfcheck"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "outbound connection attempts: 0" in lines
    assert "guard self-test: blocked 1/1" in lines
    assert lines[-1] == "selfcheck: PASS"
