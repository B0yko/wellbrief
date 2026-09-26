"""The case loader: valid cases parse, malformed ones are rejected with a clear message."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from wellbrief.evals.cases import CaseError, load_suite, parse_suite


def _suite(*cases: dict[str, Any], name: str = "extended") -> dict[str, Any]:
    return {"suite": name, "case": list(cases)}


ARITHMETIC = {"id": "a", "category": "arithmetic", "question": "How many hours?", "figure": "total_hours",
              "gold_sql": "SELECT 1"}


def test_a_valid_case_parses() -> None:
    (case,) = parse_suite(_suite(ARITHMETIC), "extended", "t")
    assert (case.id, case.category, case.gate, case.reason) == ("a", "arithmetic", True, None)
    assert case["figure"] == "total_hours"


@pytest.mark.parametrize("change, message", [
    ({"tolerance": 1.0}, "unknown key"),
    ({"figure": "hours"}, "must be one of"),
    ({"question": 3}, "wrong type"),
    ({"question": "  "}, "empty"),
    ({"gate": False}, "needs a reason"),
    ({"gate": "no", "reason": "x"}, "gate must be"),
    ({"reason": "because"}, "only allowed with gate = false"),
    ({"category": "vibes"}, "unknown category"),
    ({"spread_rate": True}, "wrong type"),
    ({"spread_rate": -60000}, "must be positive"),
    ({"spread_rate": 0}, "must be positive"),
    ({"category": "brief-precision", "field": "F", "well": "W", "td_m": 3000, "min_precision": 0.5},
     "does not belong in the extended suite"),
])
def test_malformed_cases_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(CaseError, match=message):
        parse_suite(_suite({**ARITHMETIC, **change}), "extended", "t")


BRIEF_PRECISION = {"id": "b", "category": "brief-precision", "field": "Orrindale", "well": "ORD-NEXT",
                   "td_m": 3100, "min_precision": 0.75}


@pytest.mark.parametrize("change, message", [
    ({"min_precision": 5.0}, "between 0 and 1"),
    ({"min_precision": -0.1}, "between 0 and 1"),
    ({"td_m": -3100}, "must be positive"),
    ({"risk_filters": False}, "unknown key"),
])
def test_out_of_range_brief_precision_cases_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(CaseError, match=message):
        parse_suite(_suite({**BRIEF_PRECISION, **change}, name="brief-precision"), "brief-precision", "t")


def test_targets_may_be_written_as_integers() -> None:
    (case,) = parse_suite(_suite({**BRIEF_PRECISION, "min_precision": 1}, name="brief-precision"),
                          "brief-precision", "t")
    assert case["min_precision"] == 1


def test_retrieval_precision_is_always_cut_at_8() -> None:
    case = {"id": "r", "category": "retrieval-precision", "question": "q", "field": "F",
            "relevant_sql": "SELECT 1"}
    parse_suite(_suite(case), "extended", "t")
    with pytest.raises(CaseError, match="unknown key"):
        parse_suite(_suite({**case, "top_k": 5}), "extended", "t")


def test_a_case_that_is_not_a_table_is_rejected() -> None:
    with pytest.raises(CaseError, match="must be a table"):
        parse_suite({"suite": "extended", "case": [1, 2]}, "extended", "t")


def test_missing_keys_are_rejected() -> None:
    case = {k: v for k, v in ARITHMETIC.items() if k != "gold_sql"}
    with pytest.raises(CaseError, match="missing key"):
        parse_suite(_suite(case), "extended", "t")
    with pytest.raises(CaseError, match="string id"):
        parse_suite(_suite({k: v for k, v in ARITHMETIC.items() if k != "id"}), "extended", "t")


def test_gate_false_with_a_reason_is_accepted() -> None:
    (case,) = parse_suite(_suite({**ARITHMETIC, "gate": False, "reason": "known gap"}), "extended", "t")
    assert (case.gate, case.reason) == (False, "known gap")


def test_file_level_errors() -> None:
    with pytest.raises(CaseError, match="unknown top-level"):
        parse_suite({**_suite(ARITHMETIC), "notes": "x"}, "extended", "t")
    with pytest.raises(CaseError, match="suite must be"):
        parse_suite(_suite(ARITHMETIC, name="original"), "extended", "t")
    with pytest.raises(CaseError, match="duplicate"):
        parse_suite(_suite(ARITHMETIC, ARITHMETIC), "extended", "t")
    with pytest.raises(CaseError, match="no \\[\\[case\\]\\]"):
        parse_suite({"suite": "extended"}, "extended", "t")
    grounding = {"id": "g", "category": "grounding", "from_cases": ["a"]}
    with pytest.raises(CaseError, match="not a retrieval case"):
        parse_suite(_suite(ARITHMETIC, grounding, name="original"), "original", "t")


def test_load_suite_reports_the_file(tmp_path: Path) -> None:
    (tmp_path / "extended.toml").write_text('suite = "extended"\n[[case]]\nid = "x"\n', encoding="utf-8")
    with pytest.raises(CaseError, match=r"extended\.toml"):
        load_suite(tmp_path, "extended")
    with pytest.raises(CaseError, match="unknown suite"):
        load_suite(tmp_path, "nightly")
