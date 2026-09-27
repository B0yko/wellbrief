"""The `narrator` suite: one measured `ask`/`brief` call per case, checked against this
seed's own egress log, for comparing narrator backends (`eval --suite narrator`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import evals_mini as mini
from fake_openai import FakeOpenAIServer, FakeResponse, completion
from wellbrief import cli
from wellbrief.corpus import SEED, build_corpus, write_corpus
from wellbrief.evals import adapter, runner, suites
from wellbrief.evals.cases import Case, default_dir, load_suite
from wellbrief.evals.truth import load_dir

MINI_TRUTH: dict[str, Any] = {**mini.sidecar(), "wells": [mini.well("ORD-101", "Orrindale", "Orrin-1")]}


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("narrator-corpus") / "corpus"
    write_corpus(build_corpus(seed=SEED), out)
    return out


def _ctx(corpus_dir: Path, work_dir: Path, narrator: str = "offline") -> suites.Context:
    return suites.Context(SEED, corpus_dir, work_dir, load_dir(corpus_dir), narrator=narrator)


def _ask_case(question: str) -> Case:
    return Case("narrator", "c", "narrator-ask", True, None, {"question": question})


def _brief_case(field: str, well: str, td_m: float) -> Case:
    return Case("narrator", "c", "narrator-brief", True, None,
                {"field": field, "well": well, "td_m": td_m})


# ---------------------------------------------------------------------------
# the case file itself
# ---------------------------------------------------------------------------

def test_narrator_case_file_has_twenty_asks_and_two_briefs() -> None:
    cases = load_suite(default_dir(), "narrator")
    by_category: dict[str, int] = {}
    for c in cases:
        by_category[c.category] = by_category.get(c.category, 0) + 1
    assert by_category == {"narrator-ask": 20, "narrator-brief": 2}
    assert all(c.gate for c in cases)
    briefs = {(c["field"], c["well"]) for c in cases if c.category == "narrator-brief"}
    assert briefs == {("Orrindale", "ORD-NEXT"), ("Vessra South", "VSS-NEXT")}


# ---------------------------------------------------------------------------
# offline narrator: never falls back, always verifies, never spends
# ---------------------------------------------------------------------------

def test_check_narrator_ask_offline(tmp_path: Path, corpus_dir: Path) -> None:
    ctx = _ctx(corpus_dir, tmp_path / "ws")
    try:
        outcome = suites.check_narrator_ask(
            _ask_case("How many hours of wellbore instability on Orrindale?"), ctx)
        assert outcome.passed
        assert outcome.metrics["fallback"] is False
        assert outcome.metrics["verified"] is True
        assert outcome.metrics["cost_usd"] == 0.0
        assert outcome.metrics["latency_ms"] >= 0.0
    finally:
        ctx.close()


def test_check_narrator_brief_offline(tmp_path: Path, corpus_dir: Path) -> None:
    ctx = _ctx(corpus_dir, tmp_path / "ws")
    try:
        outcome = suites.check_narrator_brief(_brief_case("Orrindale", "ORD-NEXT", 3100), ctx)
        assert outcome.passed
        assert outcome.metrics["fallback"] is False
        assert outcome.metrics["verified"] is True
        assert outcome.metrics["cost_usd"] == 0.0
    finally:
        ctx.close()


def test_check_narrator_ask_records_citations_for_the_stats(tmp_path: Path, corpus_dir: Path) -> None:
    ctx = _ctx(corpus_dir, tmp_path / "ws")
    ctx.current_suite = "narrator"
    try:
        suites.check_narrator_ask(_ask_case("How many hours of wellbore instability on Orrindale?"), ctx)
        stats = ctx.citation_stats()["narrator"]
        assert stats["checked"] > 0
        assert stats["checked"] == stats["verbatim_and_resolving"]
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# llm narrator: cost and fallback come from this seed's own egress log, and
# a repeated question is a genuine repeated call, never a cache hit
# ---------------------------------------------------------------------------

def test_check_narrator_ask_llm_narrates_records_cost_and_can_fall_back(
        tmp_path: Path, corpus_dir: Path, fake_server: FakeOpenAIServer,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_LLM_BASE_URL", fake_server.base_url)
    monkeypatch.setenv("WELLBRIEF_LLM_MODEL", "fake-model")
    question = "How many hours of wellbore instability on Orrindale?"
    case = _ask_case(question)

    offline_ctx = _ctx(corpus_dir, tmp_path / "offline-ws")
    try:
        doc_id = offline_ctx.workspace.ask(question).citations[0].doc_id
    finally:
        offline_ctx.close()

    ctx = _ctx(corpus_dir, tmp_path / "ws", narrator="llm")
    try:
        fake_server.queue(FakeResponse(completion(f"Recorded in [{doc_id}].", cost=0.0002)))
        good = suites.check_narrator_ask(case, ctx)
        assert good.passed
        assert good.metrics["fallback"] is False
        assert good.metrics["verified"] is True
        assert good.metrics["cost_usd"] == pytest.approx(0.0002)

        fake_server.queue(FakeResponse(completion("A further 999999.9 h were also recorded.")))
        bad = suites.check_narrator_ask(case, ctx)
        assert bad.passed  # the guard caught it: still a passing case, not a broken one
        assert bad.metrics["fallback"] is True
        assert bad.metrics["verified"] is True  # the delivered (fallback) text still verifies
    finally:
        ctx.close()
    # both calls really reached the server: the second one was not a cached repeat of the first
    assert len(fake_server.requests) == 2


# ---------------------------------------------------------------------------
# runner: --repeats is wired all the way through to a genuinely repeated check
# ---------------------------------------------------------------------------

def test_run_seed_with_repeats_calls_the_workspace_that_many_times(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def no_corpus(out_dir: Path, seed: int, formats: str = "txt", ledger_csv: bool = False) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

    asked: list[str] = []

    class StubWorkspace:
        def __init__(self, corpus_dir: Path, work_dir: Path, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

        def ask(self, question: str, top_k: int = 8, spread_rate: float | None = None) -> adapter.AskResult:
            asked.append(question)
            return adapter.AskResult(text="ok", ranking=[], citations=[], figures={})

        def egress_calls(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(adapter, "generate_corpus", no_corpus)
    monkeypatch.setattr(adapter, "Workspace", StubWorkspace)
    monkeypatch.setattr(runner, "load_dir", lambda corpus_dir: load_dir_stub())
    case = _ask_case("q")
    rows, _ = runner.run_seed(1, {"narrator": [case]}, True, "offline", lambda line: None, repeats=3)
    assert asked == ["q", "q", "q"]
    assert [r.metrics["repeat"] for r in rows] == [1, 2, 3]
    assert all(r.passed for r in rows)


def load_dir_stub() -> Any:
    from wellbrief.evals import truth
    return truth.load(MINI_TRUTH)


# ---------------------------------------------------------------------------
# summarise: the narrator comparison's own columns
# ---------------------------------------------------------------------------

def _row(case_id: str, latency: float, cost: float, fallback: bool,
        verified: bool = True) -> runner.CaseResult:
    return runner.CaseResult("narrator", 1, case_id, "narrator-ask", True, None, True, "ok",
                             {"latency_ms": latency, "cost_usd": cost, "fallback": fallback,
                              "verified": verified})


def test_summarise_adds_a_narrator_block() -> None:
    rows = [_row("a", 10.0, 0.001, False), _row("b", 30.0, 0.002, True), _row("c", 20.0, 0.0, False)]
    (summary,) = runner.summarise(rows)
    n = summary["narrator"]
    assert n["answers"] == 3
    assert n["fallback_rate"] == round(1 / 3, 4)
    assert n["verified_rate"] == 1.0
    assert n["p50_latency_ms"] == 20.0
    assert n["cost_per_answer_usd"] == pytest.approx(0.001)
    line = runner.summary_line(summary)
    assert "narrator:" in line and "fallback 33%" in line and "20 ms" in line


def test_summarise_leaves_out_the_narrator_block_for_other_suites() -> None:
    row = runner.CaseResult("extended", 1, "x", "arithmetic", True, None, True, "ok", {})
    (summary,) = runner.summarise([row])
    assert "narrator" not in summary
    assert "narrator" not in runner.summary_line(summary)


# ---------------------------------------------------------------------------
# CLI: --repeats reaches the runner only for --suite narrator
# ---------------------------------------------------------------------------

def test_cli_eval_narrator_suite_passes_repeats_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeReport:
        ok = True

        def to_dict(self) -> dict[str, Any]:
            return {"summary": []}

    def stub_run(suite_names: list[str], seeds: list[int], args: list[str],
                cases_dir: Path | None = None, risk_filters: bool = True, narrator: str = "offline",
                repeats: int = 1, emit: Any = print) -> FakeReport:
        seen["suites"], seen["repeats"], seen["narrator"] = suite_names, repeats, narrator
        return FakeReport()

    monkeypatch.setattr(runner, "run", stub_run)
    assert cli.main(["eval", "--suite", "narrator", "--repeats", "2", "--json"]) == 0
    assert seen == {"suites": ["narrator"], "repeats": 2, "narrator": "offline"}


def test_cli_eval_all_never_includes_the_narrator_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeReport:
        ok = True

        def to_dict(self) -> dict[str, Any]:
            return {"summary": []}

    def stub_run(suite_names: list[str], *args: Any, **kwargs: Any) -> FakeReport:
        seen["suites"] = suite_names
        return FakeReport()

    monkeypatch.setattr(runner, "run", stub_run)
    assert cli.main(["eval", "--suite", "all", "--json"]) == 0
    assert "narrator" not in seen["suites"]
