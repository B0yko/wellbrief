"""End-to-end offline self-check: everything the product does with no network access, run once
against a freshly generated corpus, plus the network guard's own positive control.

`run()` generates the demo corpus (mixed formats: `.txt` daily reports, `.pdf` end-of-well
reports, `.docx` incident reports), ingests it, builds the per-field indexes, runs every eval
suite on the default seed, and verifies a brief for each demo field, all inside a disposable
`WELLBRIEF_HOME` this module creates and removes itself; the caller's real workspace is never
touched. It then reads how many outbound connection attempts the process-level network guard
(`netguard.py`) has blocked so far, and finally runs the guard's own positive control
(`netguard.self_test`): a connection attempt to a documentation address that must be blocked and
counted. The result is `ok` only when every step above succeeded, every gated eval case passed,
both briefs verified, no unexpected outbound attempt occurred, and the guard's own probe was
blocked exactly once. This is the check a network-disabled container runs to prove the product
needs no network access for any of its offline commands (`docker run --network none ... selfcheck`).
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import netguard
from .corpus import SEED, build_corpus, write_corpus
from .evals.cases import SUITES
from .ingest import ingest_folder
from .narrate import OfflineNarrator
from .riskbrief import MissingDdrDataError, build_brief, verify_brief
from .settings import resolve_settings
from .store import Store
from .workspace import Workspace, check_field_slugs, rebuild_field_index

# The two demo fields' brief parameters, matching `evals/cases/brief-precision.toml`, so this
# check builds the same briefs the eval suite's brief-precision cases already score.
BRIEF_CHECKS: tuple[tuple[str, str, float], ...] = (
    ("Orrindale", "ORD-NEXT", 3100.0),
    ("Vessra South", "VSS-NEXT", 3000.0),
)


@dataclass
class StepTiming:
    name: str
    seconds: float


@dataclass
class SelfCheckResult:
    steps: list[StepTiming] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    outbound_connection_attempts: int = 0
    guard_probe: dict[str, int] | None = None      # netguard.self_test()'s own result
    guard_error: str | None = None                 # a RuntimeError from self_test(), as text

    @property
    def ok(self) -> bool:
        return not self.failures and self.guard_probe == {"attempted": 1, "blocked": 1}


@contextmanager
def _step(result: SelfCheckResult, name: str) -> Iterator[None]:
    start = time.monotonic()
    try:
        yield
    finally:
        result.steps.append(StepTiming(name, time.monotonic() - start))


def _offline_checks(result: SelfCheckResult, tmp_path: Path) -> None:
    """Corpus, ingest, index, eval and brief steps. Any failure is recorded on `result` (never
    raised): a broken step still lets the guard's positive control run afterwards, and the caller
    always gets a full report instead of a traceback."""
    ws = Workspace.resolve("selfcheck")
    store: Store | None = None
    try:
        with _step(result, "corpus generate"):
            corpus_dir = tmp_path / "corpus"
            write_corpus(build_corpus(seed=SEED), corpus_dir, "mixed")

        with _step(result, "ingest"):
            store = ws.open_store()
            coverage = ingest_folder(store, corpus_dir, workspace_root=ws.root)
        for message in coverage.csv_row_errors:
            result.failures.append(f"ingest: {message}")
        if coverage.skipped:
            result.failures.append(f"ingest: {len(coverage.skipped)} file(s) skipped")

        fields = store.field_names()
        with _step(result, "index"):
            check_field_slugs(fields)
            st = resolve_settings(workspace_root=ws.root)
            for f in fields:
                rebuild_field_index(ws, store, f, bm25_k1=st.retrieval.k1, bm25_b=st.retrieval.b)

        with _step(result, "eval"):
            from .evals import runner as evals_runner

            report = evals_runner.run(list(SUITES), [SEED], [], narrator="offline",
                                      emit=lambda _line: None)
        if not report.ok:
            result.failures.append("eval: one or more gated cases failed")

        with _step(result, "brief verify"):
            for field_name, well_name, td_m in BRIEF_CHECKS:
                if field_name not in fields:
                    result.failures.append(f"brief {field_name}: no documents ingested for this field")
                    continue
                try:
                    brief = build_brief(store, well_name, field_name, td_m, narrator=OfflineNarrator())
                except MissingDdrDataError as exc:
                    result.failures.append(f"brief {field_name}: {exc}")
                    continue
                check = verify_brief(brief, store)
                if not check["ok"]:
                    result.failures.append(
                        f"brief {field_name}: verification failed ({len(check['problems'])} problem(s))")
    except Exception as exc:  # noqa: BLE001 - every failure is reported on the result, never raised
        result.failures.append(f"error: {type(exc).__name__}: {exc}")
    finally:
        if store is not None:
            store.close()


def run() -> SelfCheckResult:
    """Run every offline check in a temporary `WELLBRIEF_HOME`, then the guard's positive
    control. `WELLBRIEF_HOME` (and only that variable) is restored to whatever it was before
    this call returns, whether or not it was set."""
    result = SelfCheckResult()
    prior_home = os.environ.get("WELLBRIEF_HOME")
    with tempfile.TemporaryDirectory(prefix="wellbrief-selfcheck-") as tmp:
        tmp_path = Path(tmp)
        os.environ["WELLBRIEF_HOME"] = str(tmp_path / "home")
        try:
            _offline_checks(result, tmp_path)
        finally:
            if prior_home is None:
                os.environ.pop("WELLBRIEF_HOME", None)
            else:
                os.environ["WELLBRIEF_HOME"] = prior_home

    # Counted from here, before the probe below adds one more: any attempt during the offline
    # checks above would be a product bug (none of them should ever reach the network), so it
    # fails the check even though the guard correctly blocked it.
    result.outbound_connection_attempts = netguard.stats()["blocked"]
    if result.outbound_connection_attempts:
        result.failures.append(
            f"{result.outbound_connection_attempts} unexpected outbound connection attempt(s) "
            "during the offline checks")

    try:
        probe = netguard.self_test()
    except RuntimeError as exc:
        result.guard_error = str(exc)
        result.failures.append(f"guard self-test did not run: {exc}")
    else:
        result.guard_probe = {"attempted": probe["attempted"], "blocked": probe["blocked"]}
        if result.guard_probe != {"attempted": 1, "blocked": 1}:
            result.failures.append("guard self-test did not block its own probe")
    return result
