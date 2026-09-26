"""Ranking metrics, tolerances and verbatim matching."""

from __future__ import annotations

import pytest

from wellbrief.evals.metrics import TOLERANCES, match_counts, mrr_at_k, precision_at_k, verbatim, within


def test_precision_uses_min_k_and_relevant_as_denominator() -> None:
    ranking = ["a", "x", "b", "y", "c", "z", "w", "v"]
    assert precision_at_k(ranking, {"a", "b", "c"}, 8) == 1.0          # 3 of min(8, 3)
    many = {f"r{i}" for i in range(20)}
    assert precision_at_k(["r1", "x", "r2", "y", "r3", "z", "r4", "w"], many, 8) == 0.5   # 4 of 8
    assert precision_at_k(["x", "a"], {"a", "b"}, 8) == 0.5             # short ranking: 1 of min(8, 2)


def test_precision_counts_each_document_once_and_only_the_top_k() -> None:
    assert precision_at_k(["a", "a", "a"], {"a", "b"}, 8) == 0.5
    assert precision_at_k(["x", "y", "z", "w", "v", "u", "t", "a"], {"a"}, 8) == 1.0
    assert precision_at_k(["x", "y", "z", "w", "v", "u", "t", "s", "a"], {"a"}, 8) == 0.0


def test_precision_needs_relevant_documents() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], set(), 8)


def test_mrr() -> None:
    assert mrr_at_k(["x", "y", "a", "b"], {"a", "b"}, 8) == pytest.approx(1 / 3)
    assert mrr_at_k(["a"], {"a"}, 8) == 1.0
    assert mrr_at_k(["x", "y"], {"a"}, 8) == 0.0
    assert mrr_at_k(["x", "y", "z", "w", "v", "u", "t", "s", "a"], {"a"}, 8) == 0.0
    assert mrr_at_k(["x", "x", "a"], {"a"}, 8) == 0.5     # duplicates collapse before ranking


def test_tolerances_are_inclusive() -> None:
    assert TOLERANCES == {"total_hours": 0.05, "total_cost_usd": 1.0, "avoidable_share": 0.001}
    assert within(280.55, 280.5, 0.05)
    assert not within(280.56, 280.5, 0.05)
    assert within(0.864, 0.863, 0.001)


def test_verbatim_ignores_whitespace_runs_only() -> None:
    source = "Total losses on entering\n   Vessra Carbonate at 2,660 m."
    assert verbatim("losses on entering Vessra Carbonate", source)
    assert not verbatim("losses on entering Vessra carbonate", source)
    assert not verbatim("   ", source)


def test_match_counts_is_a_multiset_match() -> None:
    assert match_counts(["a", "a", "b"], ["a", "b", "b", "c"]) == (2, 3, 4)
