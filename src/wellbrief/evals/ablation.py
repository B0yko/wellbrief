"""Retrieval ablation: mean P@8 and MRR@8 of the extended suite's retrieval-precision cases,
under every mode `search.ABLATION_MODES` names (`wellbrief eval --ablation`).

Corpus generation and ground truth loading follow the same pattern `runner.run_seed` uses,
but only the search ranking is exercised (through `adapter.Workspace.search_ranking`), not a
full `ask`, since the four modes differ only in how documents are retrieved and ranked.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any

from . import adapter, results
from .adapter import ABLATION_MODES, MODE_LABELS
from .cases import Case, default_dir, load_suite
from .metrics import mrr_at_k, precision_at_k
from .truth import load_dir

RETRIEVAL_K = 8


def _seed_scores(seed: int, cases: list[Case]) -> dict[str, list[dict[str, float]]]:
    """Per mode, the P@8/MRR@8 of every case whose `relevant_sql` names a document on
    this seed's corpus (a case it names none for is skipped, exactly as
    `suites.check_retrieval_precision` treats it)."""
    per_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode in ABLATION_MODES}
    with tempfile.TemporaryDirectory(prefix="wellbrief-ablation-") as tmp:
        work = Path(tmp)
        corpus_dir = work / "corpus"
        adapter.generate_corpus(corpus_dir, seed)
        truth = load_dir(corpus_dir)
        workspace = adapter.Workspace(corpus_dir, work / "workspace")
        try:
            for case in cases:
                relevant = set(truth.column(case["relevant_sql"]))
                if not relevant:
                    continue
                for mode in ABLATION_MODES:
                    ranking = workspace.search_ranking(case["question"], mode, top_k=RETRIEVAL_K)
                    per_mode[mode].append({
                        "p_at_8": precision_at_k(ranking, relevant, RETRIEVAL_K),
                        "mrr_at_8": mrr_at_k(ranking, relevant, RETRIEVAL_K),
                    })
        finally:
            workspace.close()
    return per_mode


def run(seeds: list[int], args: list[str], cases_dir: Path | None = None) -> dict[str, Any]:
    """Run every retrieval-precision case of the extended suite, on each of `seeds`, under
    every ablation mode, and return the JSON-ready payload `wellbrief eval --ablation` writes
    and tabulates."""
    cases_dir = cases_dir or default_dir()
    cases = [c for c in load_suite(cases_dir, "extended") if c.category == "retrieval-precision"]
    combined: dict[str, list[dict[str, float]]] = {mode: [] for mode in ABLATION_MODES}
    for seed in seeds:
        for mode, scores in _seed_scores(seed, cases).items():
            combined[mode].extend(scores)
    modes_out = []
    for mode in ABLATION_MODES:
        scores = combined[mode]
        modes_out.append({
            "mode": mode,
            "label": MODE_LABELS[mode],
            "cases": len(scores),
            "mean_p_at_8": round(fmean(s["p_at_8"] for s in scores), 4) if scores else 0.0,
            "mean_mrr_at_8": round(fmean(s["mrr_at_8"] for s in scores), 4) if scores else 0.0,
        })
    meta = results.metadata(args, ["extended"], seeds, {"cases": "retrieval-precision"}, adapter.version())
    return {"metadata": meta, "modes": modes_out}


def table(payload: dict[str, Any]) -> str:
    """The stdout table `wellbrief eval --ablation` prints alongside the JSON it can write."""
    header = f"{'mode':<48} {'cases':>5} {'mean P@8':>9} {'mean MRR@8':>10}"
    lines = [header, "-" * len(header)]
    for m in payload["modes"]:
        lines.append(f"{m['label']:<48} {m['cases']:>5} {m['mean_p_at_8']:>9.3f} {m['mean_mrr_at_8']:>10.3f}")
    return "\n".join(lines)
