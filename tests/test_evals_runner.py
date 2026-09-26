"""The runner: a broken case is a failed case, the exit status follows the gated cases only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import evals_mini as mini
from wellbrief import cli
from wellbrief.evals import adapter, runner, suites, truth
from wellbrief.evals.cases import Case

MINI_TRUTH: dict[str, Any] = {**mini.sidecar(),
                              "wells": [mini.well("ORD-101", "Orrindale", "Orrin-1", G1="clean")]}


def _case(category: str = "arithmetic", gate: bool = True, **params: Any) -> Case:
    return Case("extended", f"case-{category}", category, gate, None if gate else "known gap", params)


@pytest.fixture
def ctx(tmp_path: Path) -> suites.Context:
    return suites.Context(1, tmp_path / "corpus", tmp_path, truth.load(MINI_TRUTH))


def test_an_exception_becomes_a_failure_with_its_text(ctx: suites.Context,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(case: Case, ctx: suites.Context) -> suites.Outcome:
        raise RuntimeError("the index is on fire")

    monkeypatch.setitem(suites.CHECKS, "arithmetic", boom)
    result = runner.run_case(_case(), ctx)
    assert not result.passed
    assert result.detail == "error: RuntimeError: the index is on fire"


def test_a_missing_feature_is_a_failure_that_says_so(ctx: suites.Context,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(case: Case, ctx: suites.Context) -> suites.Outcome:
        raise adapter.NotSupported("no classifier")

    monkeypatch.setitem(suites.CHECKS, "classifier-accuracy", missing)
    result = runner.run_case(_case("classifier-accuracy"), ctx)
    assert not result.passed and result.detail == "not supported yet: no classifier"


def test_a_failed_precondition_fails_the_case_without_running_it(ctx: suites.Context,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(suites.CHECKS, "abstention", lambda case, ctx: pytest.fail("the check ran"))
    case = _case("abstention", precondition_sql="SELECT COUNT(*) = 0 FROM truth_wells WHERE well = 'ORD-101'")
    result = runner.run_case(case, ctx)
    assert not result.passed and result.detail.startswith("precondition failed")


def test_a_workspace_that_cannot_be_built_fails_each_case_without_a_rebuild(
        ctx: suites.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def broken(corpus_dir: Path, work_dir: Path) -> adapter.Workspace:
        calls.append(work_dir)
        raise ValueError("unreadable store")

    monkeypatch.setattr(adapter, "Workspace", broken)
    case = _case("discovery", field="Orrindale", code="STUCK_PIPE", formation="Keldra Salt")
    results = [runner.run_case(case, ctx) for _ in range(2)]
    assert [r.detail for r in results] == ["error: ValueError: unreadable store"] * 2
    assert len(calls) == 1


def test_corpus_setup_errors_fail_every_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_corpus(out_dir: Path, seed: int, formats: str = "txt", ledger_csv: bool = False) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(adapter, "generate_corpus", no_corpus)
    lines: list[str] = []
    rows = runner.run_seed(3, {"extended": [_case(), _case("abstention")]}, True, lines.append)
    assert [r.passed for r in rows] == [False, False]
    assert all("corpus setup failed: OSError: disk full" in r.detail for r in rows)
    assert len(lines) == 2


def _row(passed: bool | None, gate: bool, category: str = "retrieval-precision",
         **metrics: Any) -> runner.CaseResult:
    return runner.CaseResult("extended", 1, "x", category, gate, None if gate else "r", passed, "", metrics)


def test_only_gated_failures_fail_the_run() -> None:
    assert runner.Report({}, [_row(True, True), _row(False, False)]).ok
    assert not runner.Report({}, [_row(True, True), _row(False, True)]).ok
    assert runner.Report({}, [_row(True, True), _row(None, True, "brief-precision")]).ok


def test_summary_counts_and_retrieval_means() -> None:
    rows = [_row(True, True, p_at_8=1.0, mrr_at_8=1.0), _row(False, True, p_at_8=0.25, mrr_at_8=0.5),
            _row(False, False)]
    (summary,) = runner.summarise(rows)
    assert (summary["cases"], summary["passed"], summary["gated"], summary["gated_passed"]) == (3, 1, 2, 1)
    # the case that could not be scored counts as 0 in both means
    assert (summary["mean_p_at_8"], summary["mean_mrr_at_8"]) == (0.4167, 0.5)
    assert (summary["retrieval_cases"], summary["retrieval_unscored"]) == (3, 1)
    line = runner.summary_line(summary)
    assert "mean P@8 0.417" in line and "(1 unscored, counted as 0)" in line


def test_informational_rows_are_reported_but_not_counted() -> None:
    rows = [_row(True, True, "classifier-accuracy"), _row(None, True, "brief-precision")]
    (summary,) = runner.summarise(rows)
    assert (summary["cases"], summary["passed"], summary["gated"], summary["informational"]) == (1, 1, 1, 1)
    assert "1 reported without a score" in runner.summary_line(summary)
    assert " INFO " in runner.format_row(rows[1])


def test_no_risk_filters_reaches_brief_precision_only(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[bool] = []

    class Stub:
        def brief(self, field_name: str, well: str, td_m: float, risk_filters: bool = True) -> adapter.Brief:
            built.append(risk_filters)
            return adapter.Brief([], True, [])

    ctx = suites.Context(1, tmp_path, tmp_path, truth.load(MINI_TRUTH), risk_filters=False)
    ctx._workspace = Stub()  # type: ignore[assignment]
    brief = {"field": "Orrindale", "well": "ORD-NEXT", "td_m": 3100}
    precision = Case("brief-precision", "bp", "brief-precision", True, None, {**brief, "min_precision": 0.75})
    mitigations = Case("original", "bm", "brief-mitigations", True, None, brief)
    info, kept = runner.run_case(precision, ctx), runner.run_case(mitigations, ctx)
    assert built == [False, True]
    assert info.passed is None and "filters off" in info.detail
    assert kept.passed is False


def test_cli_rejects_options_that_are_not_wired_yet(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--ablation"]) == 2
    assert "not supported yet" in capsys.readouterr().err
    assert cli.main(["eval", "--narrator", "llm"]) == 2
    assert cli.main(["eval", "--repeats", "3"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["eval", "--seeds", "seven"])


def test_end_to_end_on_the_default_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate, load the truth, build the workspace, check, write: one arithmetic and one abstention case."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "original.toml").write_text(
        'suite = "original"\n'
        '[[case]]\nid = "hours"\ncategory = "arithmetic"\n'
        'question = "How many hours of stuck pipe on Orrindale?"\nfigure = "total_hours"\n'
        'gold_sql = "SELECT ROUND(SUM(hours), 1) FROM truth_events '
        'WHERE field = \'Orrindale\' AND code = \'STUCK_PIPE\'"\n'
        '[[case]]\nid = "broken"\ncategory = "arithmetic"\ngate = false\n'
        'reason = "exercises the error path"\n'
        'question = "How many hours of stuck pipe on Orrindale?"\nfigure = "total_hours"\n'
        'gold_sql = "SELECT no_such_column FROM truth_events"\n',
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    lines: list[str] = []
    report = runner.run(["original"], [20260731], ["eval", "--suite", "original"], cases_dir=cases_dir,
                        emit=lines.append)
    hours, broken = report.rows
    assert hours.passed, hours.detail
    assert not broken.passed and broken.detail.startswith("error: OperationalError")
    assert report.ok
    out = tmp_path / "r.json"
    from wellbrief.evals import results
    results.write(report.to_dict(), out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["passed_all_gates"] is True
    assert written["summary"][0]["gated_passed"] == 1
    assert "original seed 20260731: 1/2 passed (gated 1/1)" in lines
