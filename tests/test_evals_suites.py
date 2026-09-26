"""The checks: clean wells, drivers, abstention, format parity, brief precision and the classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import evals_mini as mini
import pytest

from wellbrief.evals import adapter, suites, truth
from wellbrief.evals.cases import Case

VSS = "Vessra South"
BRIEF = {"field": VSS, "well": "VSS-NEXT", "td_m": 3000}


def _vessra() -> dict[str, Any]:
    """Two wells on the mud pump rig (both with pump repairs) and two on the other rig: one with a
    drawworks repair (same code, no pattern) and one with no rig repair at all."""
    data = mini.sidecar(fields=[mini.field(VSS, "VSS", ["Vessra-3", "Vessra-5"], ["PJ-3"], ["G4"])])
    data["planted_patterns"] = [{"id": "G4", "field": VSS, "scope": "equipment", "driver": "rig",
                                 "keys": [{"code": "RIG_REPAIR", "rig": "Vessra-3"}]}]
    wells = [mini.well("VSS-201", VSS, "Vessra-3", G4="affected"), mini.well("VSS-203", VSS, "Vessra-3", G4="affected"),
             mini.well("VSS-202", VSS, "Vessra-5", G4="clean"), mini.well("VSS-204", VSS, "Vessra-5", G4="clean")]
    ddrs = [mini.ddr(f"DDR-{w['well']}-001", w, '8 1/2"', "Vessra Carbonate") for w in wells]
    eowrs = [mini.eowr(f"EOWR-{w['well']}", w) for w in wells]
    data["wells"], data["documents"] = wells, ddrs + eowrs
    data["npt_events"] = [
        mini.event("E1", ddrs[0], "RIG_REPAIR", 10.0, "G4", "g4_pump"),
        mini.event("E2", ddrs[1], "RIG_REPAIR", 12.0, "G4", "g4_pump"),
        mini.event("E3", ddrs[2], "RIG_REPAIR", 2.0, None, "drawworks"),
    ]
    practice = "Mud pump fluid ends on Vessra-5 were inspected and changed between wells; no pump NPT."
    data["sentences"] = [
        mini.sentence(eowrs[2], practice, "practice", "RIG_REPAIR", ["G4"]),
        mini.sentence(eowrs[3], practice, "practice", "RIG_REPAIR", ["G4"]),
        mini.sentence(eowrs[0], "Mud pump fluid end failed on Vessra-3.", "failure", "RIG_REPAIR", ["G4"]),
    ]
    return data


class Stub:
    """A workspace that returns canned answers."""

    def __init__(self, **answers: Any) -> None:
        self.answers = answers

    def brief(self, field_name: str, well: str, td_m: float, risk_filters: bool = True) -> adapter.Brief:
        return self.answers["brief"]  # type: ignore[no-any-return]

    def ask(self, question: str, top_k: int = 8, spread_rate: float | None = None) -> adapter.AskResult:
        return self.answers["ask"]  # type: ignore[no-any-return]

    def ledger(self, field_name: str | None) -> list[adapter.LedgerRow]:
        return self.answers.get("ledger", [])  # type: ignore[no-any-return]

    def classify(self, sentences: list[str]) -> list[str]:
        return self.answers["classify"](sentences)  # type: ignore[no-any-return]

    def close(self) -> None:
        pass


def _ctx(tmp_path: Path, data: dict[str, Any], **answers: Any) -> suites.Context:
    ctx = suites.Context(1, tmp_path / "corpus", tmp_path, truth.load(data))
    ctx._workspace = Stub(**answers)  # type: ignore[assignment]
    return ctx


def _risk(code: str = "RIG_REPAIR", rig: str | None = "Vessra-3", driver_kind: str | None = "rig",
          driver_category: str | None = "Vessra-3", mitigations: list[adapter.Mitigation] | None = None,
          scope: str = "equipment", section: str | None = None, formation: str | None = None) -> adapter.Risk:
    return adapter.Risk(code=code, scope=scope, section=section, formation=formation, rig=rig, mwd=None, driver="",
                        driver_kind=driver_kind, driver_category=driver_category, mitigations=mitigations or [],
                        citations=[])


def test_same_code_noise_on_the_other_rig_does_not_disqualify_a_clean_well() -> None:
    db = truth.load(_vessra())
    risk = _risk()
    assert suites.clean_wells(db, VSS, risk, {"G4"}) == {"VSS-202", "VSS-204"}
    # an equipment risk that matches no planted pattern keeps the per-code rule
    assert suites.clean_wells(db, VSS, risk, set()) == {"VSS-204"}
    practice = "Mud pump fluid ends on Vessra-5 were inspected and changed between wells; no pump NPT."
    for source in ("EOWR-VSS-202", "EOWR-VSS-204"):
        assert suites.mitigation_problem(db, VSS, risk, adapter.Mitigation(practice, source)) is None
    failure = adapter.Mitigation("Mud pump fluid end failed on Vessra-3.", "EOWR-VSS-201")
    assert "labelled failure" in str(suites.mitigation_problem(db, VSS, risk, failure))


@pytest.mark.parametrize("category, passed", [("Vessra-3", True), ("Vessra-5", False), (None, False)])
def test_a_rig_driver_must_blame_the_planted_rig(tmp_path: Path, category: str | None, passed: bool) -> None:
    ctx = _ctx(tmp_path, _vessra(), brief=adapter.Brief([_risk(driver_category=category)], True, []))
    outcome = suites.check_brief_driver(Case("extended", "d", "brief-driver", True, None, {**BRIEF, "pattern": "G4"}),
                                        ctx)
    assert outcome.passed is passed, outcome.detail


def _answer(text: str, abstained: bool | None, cited: bool = False) -> adapter.AskResult:
    citations = [adapter.Cited("DDR-VSS-201-001", "x")] if cited else []
    return adapter.AskResult(text=text, ranking=[], citations=citations, figures={}, abstained=abstained)


@pytest.mark.parametrize("answer, passed", [
    (_answer(suites.NO_MATCH, None), True),
    (_answer(suites.NO_MATCH, True), True),
    (_answer(suites.NO_MATCH, False), False),
    (_answer(suites.NO_MATCH, True, cited=True), False),
    (_answer("Stuck pipe cost 12 h.", None), False),
])
def test_abstention(tmp_path: Path, answer: adapter.AskResult, passed: bool) -> None:
    ctx = _ctx(tmp_path, _vessra(), ask=answer)
    case = Case("extended", "a", "abstention", True, None, {"question": "q", "precondition_sql": "SELECT 1"})
    assert suites.check_abstention(case, ctx).passed is passed


def test_brief_precision_counts_split_risks_and_reports_the_patterns(tmp_path: Path) -> None:
    risks = [_risk(), _risk(), _risk(code="WAIT_ON_MATERIALS", rig=None, scope="interval", section='8 1/2"',
                                     formation="Vessra Carbonate")]
    ctx = _ctx(tmp_path, _vessra(), brief=adapter.Brief(risks, True, []))
    case = Case("brief-precision", "bp", "brief-precision", True, None, {**BRIEF, "min_precision": 0.75})
    outcome = suites.check_brief_precision(case, ctx)
    assert not outcome.passed
    assert (outcome.metrics["planted"], outcome.metrics["listed"], outcome.metrics["patterns_covered"]) == (2, 3,
                                                                                                           ["G4"])


def test_classifier_metrics_and_practice_precision_target(tmp_path: Path) -> None:
    case = Case("brief-precision", "c", "classifier-accuracy", True, None,
                {"gold_sql": "SELECT text, label FROM truth_sentences ORDER BY id", "min_accuracy": 0.6,
                 "min_practice_precision": 0.95})
    everything_practice = _ctx(tmp_path, _vessra(), classify=lambda texts: ["practice"] * len(texts))
    outcome = suites.check_classifier_accuracy(case, everything_practice)
    assert not outcome.passed  # accuracy 2/3 is enough, but a failure sentence was called practice
    assert outcome.metrics["failure_called_practice"] == 1
    assert (outcome.metrics["distinct_sentences"], outcome.metrics["distinct_accuracy"]) == (2, 0.5)
    right = _ctx(tmp_path, _vessra(), classify=lambda texts: ["failure" if "failed" in t else "practice"
                                                                   for t in texts])
    assert suites.check_classifier_accuracy(case, right).passed


def _row(doc_id: str, field_name: str | None = VSS) -> adapter.LedgerRow:
    return adapter.LedgerRow(doc_id, field_name, "VSS-201", "2022-01-02", "RIG_REPAIR", 10.0, 50.0, '8 1/2"',
                             "Vessra Carbonate")


def _parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: str, rendered: Stub,
            base: list[adapter.LedgerRow]) -> suites.Outcome:
    folder = tmp_path / "rendered"
    folder.mkdir(exist_ok=True)
    (folder / "npt-ledger.csv").write_text("well,date,code,hours\nVSS-201,2022-01-02,RIG_REPAIR,10.0\n",
                                           encoding="utf-8")
    monkeypatch.setattr(adapter, "render", lambda root, seed, f: folder)
    monkeypatch.setattr(adapter, "Workspace", lambda corpus_dir, work_dir: rendered)
    ctx = _ctx(tmp_path, _vessra(), ledger=base)
    return suites.check_format_parity(Case("extended", "fp", "format-parity", True, None,
                                           {"format": fmt, "questions": ["q"]}), ctx)


def test_format_parity_needs_citations_to_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row("DDR-VSS-201-001")]
    silent = Stub(ledger=rows, ask=_answer("12 h", None))
    outcome = _parity(tmp_path, monkeypatch, "pdf", silent, rows)
    assert not outcome.passed and "no citations" in outcome.detail


def test_format_parity_counts_rows_filed_under_no_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = [_row("DDR-VSS-201-001")]
    rendered = Stub(ledger=base + [_row("DDR-VSS-201-001", None)],
                    ask=adapter.AskResult("", [], [adapter.Cited("row-1", "VSS-201,2022-01-02,RIG_REPAIR,10.0")], {}))
    outcome = _parity(tmp_path, monkeypatch, "csv", rendered, base)
    assert not outcome.passed and "no field: 0 of 0" in outcome.detail


@pytest.mark.parametrize("quote, passed", [("VSS-201,2022-01-02,RIG_REPAIR,10.0", True), ("RIG_REPAIR,10.0", False)])
def test_a_csv_quote_is_a_whole_ledger_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quote: str,
                                           passed: bool) -> None:
    base = [_row("DDR-VSS-201-001")]
    rendered = Stub(ledger=[_row("row-1")], ask=adapter.AskResult("", [], [adapter.Cited("row-1", quote)], {}))
    assert _parity(tmp_path, monkeypatch, "csv", rendered, base).passed is passed


@pytest.mark.parametrize("product, truth_name, same", [
    ("PJ-3", "PJ-3", True),
    ("Parvane Downhole PJ-3", "PJ-3", True),
    ("Parvane Downhole PJ-5", "PJ-3", False),
    ("PJ-35", "PJ-3", False),
    ("Vessra-3", "Vessra-3", True),
    ("Vessra-5", "Vessra-3", False),
    (None, "Vessra-3", False),
])
def test_a_rig_or_tool_is_named_as_a_whole_token(product: str | None, truth_name: str, same: bool) -> None:
    assert suites.names(product, truth_name) is same
