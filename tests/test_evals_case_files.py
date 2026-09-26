"""The committed case files: counts per category, and every query runs on the default corpus's truth."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from wellbrief.corpus import SEED, build_corpus, write_corpus
from wellbrief.evals import truth
from wellbrief.evals.cases import SUITES, Case, default_dir, load_suite

CASES = Path(__file__).resolve().parents[1] / "evals" / "cases"


def test_the_default_directory_is_the_checkout() -> None:
    assert default_dir() == CASES


def test_original_suite_has_the_17_prototype_cases() -> None:
    cases = load_suite(CASES, "original")
    assert len(cases) == 17
    counts = Counter(c.category for c in cases)
    assert counts == {"retrieval": 8, "arithmetic": 2, "grounding": 1, "discovery": 2,
                      "brief-citations": 2, "brief-mitigations": 2}
    assert all(c.gate for c in cases)


def test_extended_suite_meets_the_minimum_counts() -> None:
    cases = load_suite(CASES, "extended")
    counts = Counter(c.category for c in cases)
    assert counts["retrieval-precision"] >= 16
    assert counts["arithmetic"] >= 8
    assert counts["abstention"] >= 5
    assert counts["brief-recall"] == 4 and counts["brief-driver"] == 3
    assert counts["mitigation-precision"] == 2 and counts["mitigation-recall"] == 4
    assert counts["parser-fidelity"] == 2 and counts["format-parity"] == 3
    assert len(cases) >= 46
    # a case the final run still fails may be ungated, but only with its reason
    assert all(c.reason for c in cases if not c.gate)
    original = {c["question"] for c in load_suite(CASES, "original") if c.category == "retrieval"}
    assert original <= {c["question"] for c in cases if c.category == "retrieval-precision"}


def test_brief_precision_suite() -> None:
    cases = load_suite(CASES, "brief-precision")
    assert Counter(c.category for c in cases) == {"brief-precision": 2, "classifier-accuracy": 1}
    assert all(c.gate for c in cases)


@pytest.fixture(scope="module")
def default_truth(tmp_path_factory: pytest.TempPathFactory) -> truth.Truth:
    out = tmp_path_factory.mktemp("case-files-corpus")
    write_corpus(build_corpus(seed=SEED), out)
    return truth.load_dir(out)


ALL_CASES = [c for name in SUITES for c in load_suite(CASES, name)]


@pytest.mark.parametrize("case", ALL_CASES, ids=[f"{c.suite}:{c.id}" for c in ALL_CASES])
def test_every_query_holds_on_the_default_corpus(case: Case, default_truth: truth.Truth) -> None:
    """Preconditions hold, relevant sets are not empty, scalar golds return one value (truth queries only)."""
    if case.get("precondition_sql"):
        assert default_truth.scalar(case["precondition_sql"])
    if case.get("relevant_sql"):
        assert default_truth.column(case["relevant_sql"])
    if case.get("gold_sql"):
        params = {"spread_rate": case["spread_rate"]} if case.get("spread_rate") is not None else ()
        if case.category == "arithmetic":
            assert default_truth.scalar(case["gold_sql"], params) is not None
        else:
            assert default_truth.rows(case["gold_sql"], params)
