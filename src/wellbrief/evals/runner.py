"""Run evaluation suites: one generated corpus, truth database and workspace per seed.

For each seed the runner writes the synthetic corpus into a temporary
directory, loads its ground truth, builds a product workspace from it on first
use, and runs every case of the requested suites. A case that raises is a
failure that carries the exception text; a case that needs a product feature
that does not exist yet fails with "not supported yet". A case run outside the
conditions its target was set for (brief precision under `--no-risk-filters`)
is reported with `passed = None`: its figures are recorded, it is neither a
pass nor a failure, and it is left out of the pass counts. The run passes when
every gated, scored case passes on every seed.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from . import adapter, results, suites
from .cases import Case, default_dir, load_suite
from .truth import load_dir

Emit = Callable[[str], None]


@dataclass
class CaseResult:
    suite: str
    seed: int
    id: str
    category: str
    gate: bool
    reason: str | None
    passed: bool | None                 # None: reported, not scored
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


def run_case(case: Case, ctx: suites.Context) -> CaseResult:
    ctx.current_suite = case.suite
    try:
        pre = case.get("precondition_sql")
        if pre is not None and not ctx.truth.scalar(pre):
            outcome = suites.Outcome(False, f"precondition failed on this corpus: {pre}")
        else:
            outcome = suites.CHECKS[case.category](case, ctx)
    except adapter.NotSupported as exc:
        outcome = suites.Outcome(False, f"not supported yet: {exc}")
    except Exception as exc:  # noqa: BLE001 - a broken case is a failed case, never a crashed run
        outcome = suites.Outcome(False, f"error: {type(exc).__name__}: {exc}")
    passed: bool | None = outcome.passed if ctx.filters_for(case) else None
    return CaseResult(case.suite, ctx.seed, case.id, case.category, case.gate, case.reason,
                      passed, outcome.detail, outcome.metrics)


def _failed(case: Case, seed: int, detail: str) -> CaseResult:
    return CaseResult(case.suite, seed, case.id, case.category, case.gate, case.reason, False, detail)


def run_seed(seed: int, suite_cases: dict[str, list[Case]], risk_filters: bool, narrator: str,
             emit: Emit) -> tuple[list[CaseResult], dict[str, dict[str, int]]]:
    """The case results of this seed, and its per-suite citation stats (`suites.Context.
    citation_stats`; empty when the corpus or workspace could not even be built)."""
    out: list[CaseResult] = []
    citations: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="wellbrief-eval-") as tmp:
        work = Path(tmp)
        ctx: suites.Context | None = None
        setup_error = ""
        try:
            adapter.generate_corpus(work / "corpus", seed)
            ctx = suites.Context(seed, work / "corpus", work, load_dir(work / "corpus"),
                                 risk_filters=risk_filters, narrator=narrator)
        except Exception as exc:  # noqa: BLE001 - reported on every case of the seed
            setup_error = f"error: corpus setup failed: {type(exc).__name__}: {exc}"
        try:
            for cases in suite_cases.values():
                if ctx is not None:
                    ctx.cases = {c.id: c for c in cases}
                for case in cases:
                    result = run_case(case, ctx) if ctx is not None else _failed(case, seed, setup_error)
                    out.append(result)
                    emit(format_row(result))
            if ctx is not None:
                citations = ctx.citation_stats()
        finally:
            if ctx is not None:
                ctx.close()
    return out, citations


def format_row(r: CaseResult) -> str:
    status = "INFO" if r.passed is None else "PASS" if r.passed else "FAIL" if r.gate else "fail"
    detail = r.detail if len(r.detail) <= 140 else r.detail[:137] + "..."
    return f"  {r.id:<42} {r.category:<21} {status}  {detail}"


def summarise(rows: list[CaseResult],
             citations: dict[tuple[str, int], dict[str, int]] | None = None) -> list[dict[str, Any]]:
    """Pass counts per suite and seed, the mean P@8 and MRR@8 over every retrieval precision
    case, and (when `citations` has an entry for it) that suite and seed's citation stats.

    Cases reported without a score are counted apart (`informational`). A
    retrieval precision case that could not be scored (the product raised, or
    the relevant set was empty) counts as P@8 = MRR@8 = 0 in the means.
    """
    citations = citations or {}
    out = []
    for suite, seed in dict.fromkeys((r.suite, r.seed) for r in rows):
        group = [r for r in rows if r.suite == suite and r.seed == seed]
        scored = [r for r in group if r.passed is not None]
        gated = [r for r in scored if r.gate]
        entry: dict[str, Any] = {
            "suite": suite, "seed": seed, "cases": len(scored), "passed": sum(bool(r.passed) for r in scored),
            "gated": len(gated), "gated_passed": sum(bool(r.passed) for r in gated),
            "informational": len(group) - len(scored),
        }
        retrieval = [r for r in group if r.category == "retrieval-precision"]
        if retrieval:
            entry["retrieval_cases"] = len(retrieval)
            entry["retrieval_unscored"] = sum(1 for r in retrieval if "p_at_8" not in r.metrics)
            entry["mean_p_at_8"] = round(fmean(r.metrics.get("p_at_8", 0.0) for r in retrieval), 4)
            entry["mean_mrr_at_8"] = round(fmean(r.metrics.get("mrr_at_8", 0.0) for r in retrieval), 4)
        stats = citations.get((suite, seed))
        if stats is not None:
            checked, verbatim_and_resolving = stats["checked"], stats["verbatim_and_resolving"]
            percent = round(100 * verbatim_and_resolving / checked, 1) if checked else 100.0
            entry["citations"] = {"checked": checked, "verbatim_and_resolving": verbatim_and_resolving,
                                  "percent": percent}
        out.append(entry)
    return out


def summary_line(s: dict[str, Any]) -> str:
    line = (f"{s['suite']} seed {s['seed']}: {s['passed']}/{s['cases']} passed"
            f" (gated {s['gated_passed']}/{s['gated']})")
    if s.get("informational"):
        line += f", {s['informational']} reported without a score"
    if "retrieval_cases" in s:
        line += (f"; retrieval: mean P@8 {s['mean_p_at_8']:.3f}, mean MRR@8 {s['mean_mrr_at_8']:.3f}"
                 f" over {s['retrieval_cases']} cases")
        if s["retrieval_unscored"]:
            line += f" ({s['retrieval_unscored']} unscored, counted as 0)"
    if "citations" in s:
        c = s["citations"]
        line += (f"; citations: {c['verbatim_and_resolving']}/{c['checked']} verbatim and resolving "
                 f"({c['percent']:.1f}%)")
    return line


@dataclass
class Report:
    metadata: dict[str, Any]
    rows: list[CaseResult]
    citations: dict[tuple[str, int], dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.rows if r.gate and r.passed is not None)

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "passed_all_gates": self.ok,
                "summary": summarise(self.rows, self.citations), "results": [asdict(r) for r in self.rows]}


def run(suite_names: list[str], seeds: list[int], args: list[str], cases_dir: Path | None = None,
        risk_filters: bool = True, narrator: str = "offline", emit: Emit = print) -> Report:
    """Run the suites on every seed. Case files are validated before anything is generated."""
    cases_dir = cases_dir or default_dir()
    suite_cases = {name: load_suite(cases_dir, name) for name in suite_names}
    meta = results.metadata(args, suite_names, seeds, {"risk_filters": risk_filters, "narrator": narrator},
                            adapter.version())
    rows: list[CaseResult] = []
    citations: dict[tuple[str, int], dict[str, int]] = {}
    emit("status: PASS, FAIL, fail for a failed case that is not gated, "
         "INFO for a case reported without a score")
    for seed in seeds:
        emit(f"seed {seed}")
        seed_rows, seed_citations = run_seed(seed, suite_cases, risk_filters, narrator, emit)
        rows.extend(seed_rows)
        for suite, stats in seed_citations.items():
            citations[(suite, seed)] = stats
    report = Report(meta, rows, citations)
    emit("")
    for s in summarise(rows, citations):
        emit(summary_line(s))
    emit("all gated cases passed" if report.ok else "gated cases failed")
    return report
