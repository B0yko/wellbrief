"""`wellbrief bench`: the performance numbers the README reports.

Every measurement below runs against a corpus this process builds and tears down itself
(never a fixture left over from another run), so the numbers describe the code as it stands
right now. `run()` ties them together into the JSON payload `wellbrief bench --out` writes;
installing the package (cold, into a throwaway `uv` cache) is timed separately, by whatever
command the README documents, since it has nothing to do with this module's own import path.

One function per measurement:

- `demo_startup_seconds`: wall-clock time from launching `wellbrief demo --no-browser` against
  an empty `$WELLBRIEF_HOME` until `GET /api/status` first answers.
- `run_scale`: corpus generation, ingest and index-build time at one scale, the resulting
  index size on disk, a cold-start reload of that index, and `ask`/`brief` latency against it.
- `ask_latency` / `brief_latency`: p50/p95 milliseconds over `WARM_REPEATS` timed repeats of a
  fixed question or brief set, after one untimed warm-up call each (so the first call's Python
  import and bytecode-compile cost is never charged to the retrieval or narration path this
  measures).
- `peak_rss_mib`: this process's peak resident set size so far (`resource.getrusage`),
  normalised: macOS reports `ru_maxrss` in bytes, Linux in KiB.

`default_ask_questions`/`default_briefs` read the committed case files rather than naming a
field or a well here directly, so this module tracks the case files if they ever change.
"""

from __future__ import annotations

import os
import platform
import queue
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from . import __version__
from .config import DEFAULT_SPREAD_RATE_USD_PER_DAY, EMBED_BACKEND, RRF_K
from .corpus import MAX_SCALE, SEED, build_corpus, write_corpus
from .embed import get_embedder
from .evals.cases import default_dir, load_suite
from .evals.results import command_line, cpu_model, git_state, os_name, ram_gb
from .ingest import ingest_folder
from .narrate import OfflineNarrator
from .qa import ask as product_ask
from .riskbrief import build_brief
from .search import Searcher
from .store import Store
from .workspace import Workspace, check_field_slugs, load_field_index, rebuild_field_index

WARM_REPEATS = 20
DEMO_TARGET_SECONDS = 30.0
ASK_P95_TARGET_MS = 500.0    # target at scale 1 only; report scale 10 as measured


# ---------------------------------------------------------------------------
# Pure helpers: percentiles, timed repeats, peak memory
# ---------------------------------------------------------------------------


def peak_rss_mib() -> float:
    """This process's peak resident set size so far, in MiB.

    `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a high-water mark (it never falls, even
    after memory is freed), reported in bytes on macOS and in KiB on Linux; this normalises
    both to MiB so a report is comparable across platforms.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile (`fraction` in `[0, 1]`) over `values`; `0.0` on an
    empty list, so a caller with no samples gets a clean zero rather than an exception."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


@dataclass(frozen=True)
class Latency:
    """p50/p95 milliseconds over one pool of timed calls, and how many samples fed it."""

    p50_ms: float
    p95_ms: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timed_repeats(calls: list[Callable[..., Any]], repeats: int = WARM_REPEATS) -> Latency:
    """Call every entry of `calls` once, untimed (a warm-up: the first call against a fresh
    index pays for lazy setup a later call does not), then `repeats` more times, timed; return
    the p50/p95 milliseconds pooled over every timed call, of every entry, together.

    An empty `calls` list returns zero samples rather than dividing by one.
    """
    for call in calls:
        call()
    samples: list[float] = []
    for _ in range(max(repeats, 0)):
        for call in calls:
            started = time.perf_counter()
            call()
            samples.append((time.perf_counter() - started) * 1000)
    return Latency(p50_ms=round(_percentile(samples, 0.50), 2),
                    p95_ms=round(_percentile(samples, 0.95), 2), samples=len(samples))


# ---------------------------------------------------------------------------
# The fixed question / brief sets: read from the committed case files, so
# this module has no field or well name of its own to fall out of step.
# ---------------------------------------------------------------------------


def default_ask_questions(cases_dir: Path | None = None) -> list[str]:
    """Every plain-English question in the extended suite (retrieval, arithmetic and
    abstention cases): the fixed set `wellbrief bench` times `ask` over."""
    cases = load_suite(cases_dir or default_dir(), "extended")
    return [c["question"] for c in cases if "question" in c.params]


def default_briefs(cases_dir: Path | None = None) -> list[tuple[str, str, float]]:
    """`(well, field, planned_td_m)` of the two field briefs the brief-precision suite scores,
    in file order and without duplicates: the fixed set `wellbrief bench` times `brief` over."""
    cases = load_suite(cases_dir or default_dir(), "brief-precision")
    seen: dict[tuple[str, str, float], None] = {}
    for c in cases:
        if c.category == "brief-precision":
            seen.setdefault((c["well"], c["field"], float(c["td_m"])), None)
    return list(seen)


# ---------------------------------------------------------------------------
# ask / brief latency
# ---------------------------------------------------------------------------


def ask_latency(store: Store, searcher: Searcher, questions: Iterable[str],
                repeats: int = WARM_REPEATS, top_k: int = 8,
                spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY) -> Latency:
    """p50/p95 milliseconds of `ask`, over `questions`, offline-narrated (the deterministic
    path every `ask` falls back to; no network, so a repeat's cost is retrieval and analytics
    alone, not a narrator round trip)."""
    narrator = OfflineNarrator()
    calls = [(lambda q=question: product_ask(
        q, store, searcher, narrator=narrator, top_k=top_k, spread_rate=spread_rate))
        for question in questions]
    return timed_repeats(calls, repeats)


def brief_latency(store: Store, briefs: Iterable[tuple[str, str, float]], repeats: int = WARM_REPEATS,
                  spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY) -> Latency:
    """p50/p95 milliseconds of `brief` (offline-narrated, see `ask_latency`), over `briefs`."""
    narrator = OfflineNarrator()
    calls = [(lambda w=well, f=field_name, td=td_m: build_brief(
        store, w, f, td, spread_rate=spread_rate, narrator=narrator))
        for well, field_name, td_m in briefs]
    return timed_repeats(calls, repeats)


# ---------------------------------------------------------------------------
# One workspace: index build + size + cold-start reload + ask/brief latency
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def measure_workspace(ws: Workspace, store: Store, questions: Iterable[str],
                      briefs: Iterable[tuple[str, str, float]], repeats: int = WARM_REPEATS,
                      spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY,
                      embed_backend: str = EMBED_BACKEND) -> dict[str, Any]:
    """Build and persist every field's index already ingested into `store`, then measure: the
    index-build time, its size on disk, a cold-start reload from disk, and `ask`/`brief`
    latency against the reloaded index -- the state a served process actually answers from,
    not the in-memory index the build step already holds.

    The one measurement helper `tests/test_bench.py` exercises directly, over a small,
    already-ingested store (never the full synthetic corpus, so the test stays fast).
    """
    known = store.field_names()
    check_field_slugs(known)
    fields = sorted(known)

    started = time.monotonic()
    for field_name in fields:
        rebuild_field_index(ws, store, field_name, embed_backend=embed_backend)
    index_build_seconds = time.monotonic() - started
    index_bytes = _dir_size_bytes(ws.fields_dir) if ws.fields_dir.exists() else 0

    started = time.monotonic()
    reloaded = {field_name: load_field_index(ws, field_name) for field_name in fields}
    cold_start_load_seconds = time.monotonic() - started
    missing = [f for f, idx in reloaded.items() if idx is None]
    if missing:
        raise RuntimeError(f"index reload failed right after it was built, for field(s): "
                           f"{', '.join(missing)}")
    indexes = {f: idx for f, idx in reloaded.items() if idx is not None}

    searcher = Searcher(store, indexes, get_embedder(embed_backend), rrf_k=RRF_K)
    ask = ask_latency(store, searcher, questions, repeats=repeats, spread_rate=spread_rate)
    brief = brief_latency(store, briefs, repeats=repeats, spread_rate=spread_rate)

    return {
        "fields": fields,
        "index_build_seconds": round(index_build_seconds, 4),
        "index_bytes": index_bytes,
        "cold_start_load_seconds": round(cold_start_load_seconds, 4),
        "ask": ask.to_dict(),
        "brief": brief.to_dict(),
        "peak_rss_mib": round(peak_rss_mib(), 1),
    }


def ingest_and_measure(corpus_dir: Path, ws: Workspace, questions: Iterable[str],
                       briefs: Iterable[tuple[str, str, float]],
                       repeats: int = WARM_REPEATS) -> dict[str, Any]:
    """Ingest `corpus_dir` into a fresh store under `ws`, timed, then `measure_workspace` on
    top of it. The other measurement helper `tests/test_bench.py` exercises directly, over
    `examples/alt-template` (three daily reports and one end-of-well report: fast)."""
    store = ws.open_store()
    try:
        started = time.monotonic()
        coverage = ingest_folder(store, corpus_dir, workspace_root=ws.root)
        ingest_seconds = time.monotonic() - started
        report = measure_workspace(ws, store, questions, briefs, repeats=repeats)
    finally:
        store.close()
    report["ingest_seconds"] = round(ingest_seconds, 4)
    report["documents"] = coverage.documents
    report["npt_events"] = coverage.npt_events
    return report


def run_scale(scale: int, work_root: Path, seed: int = SEED, questions: Iterable[str] | None = None,
             briefs: Iterable[tuple[str, str, float]] | None = None,
             repeats: int = WARM_REPEATS) -> dict[str, Any]:
    """Generate the synthetic corpus at `scale` under `work_root`, ingest and index it into a
    fresh workspace there, and return every timing/size measurement for that scale.
    `questions`/`briefs` default to `default_ask_questions`/`default_briefs`."""
    questions = list(default_ask_questions() if questions is None else questions)
    briefs = list(default_briefs() if briefs is None else briefs)
    corpus_dir = work_root / "corpus"
    started = time.monotonic()
    generated = build_corpus(seed=seed, scale=scale)
    write_corpus(generated, corpus_dir, "txt")
    corpus_generation_seconds = time.monotonic() - started

    ws = Workspace(home=work_root / "home", name="bench")
    report = ingest_and_measure(corpus_dir, ws, questions, briefs, repeats=repeats)
    report["scale"] = scale
    report["corpus_generation_seconds"] = round(corpus_generation_seconds, 4)
    return report


# ---------------------------------------------------------------------------
# Demo end to end: generate + ingest + index, through the real `demo` command
# ---------------------------------------------------------------------------

_LISTENING_RE = re.compile(r"listening on (http://\S+/)")


def _pump_lines(stream: Any, sink: queue.Queue[str | None]) -> None:
    for line in stream:
        sink.put(line)
    sink.put(None)


def _wait_for_listening_url(proc: subprocess.Popen[str], deadline: float) -> str:
    assert proc.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()
    thread = threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True)
    thread.start()
    seen: list[str] = []
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=max(deadline - time.monotonic(), 0.01))
        except queue.Empty:
            break
        if line is None:
            break
        seen.append(line)
        match = _LISTENING_RE.search(line)
        if match:
            return match.group(1)
    proc.kill()
    proc.wait(timeout=5)
    raise TimeoutError("wellbrief demo never reported a listening URL in time; output so far:\n"
                       + "".join(seen))


def _wait_for_status_ok(base_url: str, deadline: float) -> None:
    url = base_url.rstrip("/") + "/api/status"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:   # loopback only: this process's own subprocess
                if resp.status == 200:
                    return
        except (OSError, URLError, HTTPError) as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"GET {url} never answered in time: {last_error}")


def demo_startup_seconds(home: Path, formats: str, timeout: float = DEMO_TARGET_SECONDS) -> float:
    """Wall-clock seconds from launching `wellbrief demo --no-browser --formats <formats>`
    against the empty `home` (`$WELLBRIEF_HOME`) until `GET /api/status` first answers 200;
    the process is killed either way before this returns.

    A fresh, ephemeral port (`--port 0`) avoids colliding with another `serve`/`demo` already
    bound to the default one, on this machine or in a parallel test run.
    """
    env = {**os.environ, "WELLBRIEF_HOME": str(home), "PYTHONUNBUFFERED": "1"}
    env.pop("WELLBRIEF_WORKSPACE", None)
    started = time.monotonic()
    deadline = started + timeout
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "wellbrief", "demo", "--no-browser", "--port", "0",
         "--formats", formats],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    try:
        base_url = _wait_for_listening_url(proc, deadline)
        _wait_for_status_ok(base_url, deadline)
        return time.monotonic() - started
    finally:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Metadata and the full run
# ---------------------------------------------------------------------------


def metadata(args: list[str], scales: list[int], options: dict[str, Any]) -> dict[str, Any]:
    """The same shape `evals.results.metadata` writes for an eval run (date, hardware, Python,
    git state, command line -- never a hostname), with `scales` in place of suites/seeds."""
    return {
        "wellbrief_version": __version__,
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command_line(args),
        "scales": scales,
        "options": options,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "hardware": {"cpu": cpu_model(), "ram_gb": ram_gb(), "os": os_name(), "arch": platform.machine()},
        "git": git_state(),
    }


def run(scales: list[int], args: list[str], repeats: int = WARM_REPEATS, seed: int = SEED,
       demo_timeout: float = DEMO_TARGET_SECONDS,
       emit: Callable[[str], None] = print) -> dict[str, Any]:
    """Run every section-9d measurement: `run_scale` at each of `scales`, then the demo
    end-to-end timing for the mixed and the txt renderings. Returns the JSON payload
    `wellbrief bench --out` writes; `emit` is the progress line callback (`print`, or a no-op
    for `--json`, exactly like `wellbrief eval`)."""
    for scale in scales:
        if not 1 <= scale <= MAX_SCALE:
            raise ValueError(f"--scale must be between 1 and {MAX_SCALE}, not {scale}")

    questions = default_ask_questions()
    briefs = default_briefs()
    scale_reports = []
    for scale in scales:
        emit(f"scale {scale}: generating, ingesting, indexing ...")
        with tempfile.TemporaryDirectory(prefix=f"wellbrief-bench-scale{scale}-") as tmp:
            report = run_scale(scale, Path(tmp), seed=seed, questions=questions, briefs=briefs,
                               repeats=repeats)
        emit(f"  corpus {report['corpus_generation_seconds']:.2f}s  ingest {report['ingest_seconds']:.2f}s"
             f"  index {report['index_build_seconds']:.2f}s ({report['index_bytes']:,} bytes)"
             f"  cold-start load {report['cold_start_load_seconds']:.3f}s")
        emit(f"  ask p50 {report['ask']['p50_ms']:.1f}ms p95 {report['ask']['p95_ms']:.1f}ms"
             f"  brief p50 {report['brief']['p50_ms']:.1f}ms p95 {report['brief']['p95_ms']:.1f}ms")
        scale_reports.append(report)

    demo_seconds: dict[str, float] = {}
    for formats in ("mixed", "txt"):
        emit(f"demo --no-browser --formats {formats}: waiting for /api/status ...")
        with tempfile.TemporaryDirectory(prefix=f"wellbrief-bench-demo-{formats}-") as tmp:
            seconds = demo_startup_seconds(Path(tmp) / "home", formats, timeout=demo_timeout)
        emit(f"  {seconds:.2f}s")
        demo_seconds[formats] = round(seconds, 2)

    return {
        "metadata": metadata(args, scales, {"warm_repeats": repeats, "seed": seed}),
        "demo_startup_seconds": demo_seconds,
        "demo_target_seconds": DEMO_TARGET_SECONDS,
        "ask_p95_target_ms": ASK_P95_TARGET_MS,
        "scales": scale_reports,
        "peak_rss_mib": round(peak_rss_mib(), 1),
    }
