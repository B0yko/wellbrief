"""`wellbrief bench`'s measurement helpers, exercised over a tiny, fast corpus:
`examples/alt-template` (three daily reports and one end-of-well report) for everything that
needs an ingested workspace, and hand-controlled inputs for the pure timing/percentile helpers.
The full generator at scale, and the `demo --no-browser` subprocess timing, are what `wellbrief
bench` itself runs on a clean tree; neither belongs in a fast test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief import bench
from wellbrief.corpus import SEED
from wellbrief.evals.cases import default_dir, load_suite
from wellbrief.workspace import Workspace

ALT_TEMPLATE = Path(__file__).resolve().parents[1] / "examples" / "alt-template"


# ---------------------------------------------------------------------------
# timed_repeats / percentiles
# ---------------------------------------------------------------------------


def test_timed_repeats_warms_up_once_then_times_every_repeat() -> None:
    calls_made: list[int] = []

    def call() -> None:
        calls_made.append(1)

    result = bench.timed_repeats([call], repeats=3)
    assert len(calls_made) == 1 + 3   # one untimed warm-up, then the timed repeats
    assert result.samples == 3


def test_timed_repeats_pools_samples_across_every_call_in_the_list() -> None:
    calls_made: list[str] = []
    result = bench.timed_repeats(
        [lambda: calls_made.append("a"), lambda: calls_made.append("b")], repeats=4)
    assert calls_made.count("a") == 1 + 4
    assert calls_made.count("b") == 1 + 4
    assert result.samples == 2 * 4   # both calls' timed repeats, pooled


def test_timed_repeats_percentiles_over_controlled_durations(monkeypatch: pytest.MonkeyPatch) -> None:
    # Five timed calls of 10/20/30/40/50 ms each (perf_counter is read at start and end of
    # every timed call; the untimed warm-up never touches it).
    ticks = iter([0.000, 0.010, 0.000, 0.020, 0.000, 0.030, 0.000, 0.040, 0.000, 0.050])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    result = bench.timed_repeats([lambda: None], repeats=5)
    assert result.samples == 5
    assert result.p50_ms == pytest.approx(30.0)
    assert result.p95_ms == pytest.approx(48.0)


def test_timed_repeats_on_an_empty_call_list_is_zero_not_a_crash() -> None:
    result = bench.timed_repeats([], repeats=5)
    assert result == bench.Latency(p50_ms=0.0, p95_ms=0.0, samples=0)


# ---------------------------------------------------------------------------
# peak_rss_mib
# ---------------------------------------------------------------------------


def test_peak_rss_mib_normalises_macos_bytes_vs_linux_kib(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeUsage:
        ru_maxrss = 2 * 1024 * 1024   # 2 MiB in bytes (macOS), or 2 GiB in KiB (Linux)

    monkeypatch.setattr(bench.resource, "getrusage", lambda who: FakeUsage())
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    assert bench.peak_rss_mib() == pytest.approx(2.0)
    monkeypatch.setattr(bench.platform, "system", lambda: "Linux")
    assert bench.peak_rss_mib() == pytest.approx(2.0 * 1024)


# ---------------------------------------------------------------------------
# The fixed question / brief sets, read from the committed case files
# ---------------------------------------------------------------------------


def test_default_ask_questions_reads_every_question_in_the_extended_suite() -> None:
    questions = bench.default_ask_questions()
    expected = [c["question"] for c in load_suite(default_dir(), "extended") if "question" in c.params]
    assert questions == expected
    assert len(questions) >= 30
    assert 'What caused stuck pipe on Orrindale wells in the 17 1/2" section?' in questions


def test_default_briefs_reads_the_two_field_briefs_without_duplicates() -> None:
    briefs = bench.default_briefs()
    assert briefs == [("ORD-NEXT", "Orrindale", 3100.0), ("VSS-NEXT", "Vessra South", 3000.0)]


# ---------------------------------------------------------------------------
# measure_workspace / ingest_and_measure, over the tiny alt-template corpus
# ---------------------------------------------------------------------------


def test_ingest_and_measure_over_a_tiny_corpus(tmp_path: Path) -> None:
    ws = Workspace(home=tmp_path / "home", name="bench")
    questions = ["What non-productive time was recorded on Marrow Deep?"]
    briefs = [("MRD-NEXT", "Marrow Deep", 2200.0)]

    report = bench.ingest_and_measure(ALT_TEMPLATE, ws, questions, briefs, repeats=2)

    assert report["fields"] == ["Marrow Deep"]
    assert report["documents"] == 4
    assert report["npt_events"] == 3
    assert report["ingest_seconds"] >= 0.0
    assert report["index_build_seconds"] >= 0.0
    assert report["index_bytes"] > 0
    assert report["cold_start_load_seconds"] >= 0.0
    assert report["ask"]["samples"] == 2 * len(questions)
    assert report["brief"]["samples"] == 2 * len(briefs)
    assert report["ask"]["p50_ms"] >= 0.0
    assert report["ask"]["p95_ms"] >= report["ask"]["p50_ms"]
    assert report["brief"]["p50_ms"] >= 0.0
    assert report["peak_rss_mib"] > 0.0

    # The on-disk index this measured is real: every field's three index files exist.
    field_dir = ws.field_dir("Marrow Deep")
    assert {p.name for p in field_dir.iterdir()} == {"bm25.json", "vectors.f32", "manifest.json"}


def test_measure_workspace_reload_matches_the_freshly_built_index(tmp_path: Path) -> None:
    from wellbrief.ingest import ingest_folder

    ws = Workspace(home=tmp_path / "home", name="bench")
    store = ws.open_store()
    try:
        ingest_folder(store, ALT_TEMPLATE, workspace_root=ws.root)
        report = bench.measure_workspace(
            ws, store, questions=["Stuck pipe on Marrow Deep"],
            briefs=[("MRD-NEXT", "Marrow Deep", 2200.0)], repeats=1)
    finally:
        store.close()
    assert report["fields"] == ["Marrow Deep"]
    assert report["ask"]["samples"] == 1
    assert report["brief"]["samples"] == 1


def test_measure_workspace_raises_a_clear_error_when_the_reload_comes_back_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field this call just built and saved an index for, but whose reload somehow finds
    nothing (a corrupted or concurrently-removed index file, say), is a clear internal error,
    not a silently empty index handed to `Searcher`."""
    from wellbrief.ingest import ingest_folder

    ws = Workspace(home=tmp_path / "home", name="bench")
    store = ws.open_store()
    try:
        ingest_folder(store, ALT_TEMPLATE, workspace_root=ws.root)
        monkeypatch.setattr(bench, "load_field_index", lambda ws, field: None)
        with pytest.raises(RuntimeError, match="Marrow Deep"):
            bench.measure_workspace(ws, store, questions=[], briefs=[], repeats=1)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# ask_latency / brief_latency directly
# ---------------------------------------------------------------------------


def test_ask_latency_and_brief_latency_are_offline_and_deterministic_in_shape(tmp_path: Path) -> None:
    from wellbrief.config import RRF_K
    from wellbrief.embed import get_embedder
    from wellbrief.ingest import ingest_folder
    from wellbrief.search import Searcher
    from wellbrief.workspace import load_field_index, rebuild_field_index

    ws = Workspace(home=tmp_path / "home", name="bench")
    store = ws.open_store()
    try:
        ingest_folder(store, ALT_TEMPLATE, workspace_root=ws.root)
        rebuild_field_index(ws, store, "Marrow Deep")
        index = load_field_index(ws, "Marrow Deep")
        assert index is not None
        searcher = Searcher(store, {"Marrow Deep": index}, get_embedder(), rrf_k=RRF_K)

        ask = bench.ask_latency(store, searcher, ["Stuck pipe on Marrow Deep"], repeats=3)
        assert ask.samples == 3
        assert ask.p50_ms >= 0.0

        brief = bench.brief_latency(store, [("MRD-NEXT", "Marrow Deep", 2200.0)], repeats=3)
        assert brief.samples == 3
        assert brief.p50_ms >= 0.0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# One real (but short) end-to-end scale run, over the actual generator: proves
# run_scale's own wiring (corpus generation -> ingest -> index -> ask/brief),
# not just the tiny-fixture path above. Kept small: 2 questions, 2 repeats.
# ---------------------------------------------------------------------------


def test_run_scale_end_to_end_on_the_smallest_scale(tmp_path: Path) -> None:
    report = bench.run_scale(
        1, tmp_path, seed=SEED,
        questions=["What caused stuck pipe on Orrindale wells in the 17 1/2\" section?",
                  "How many hours of fishing on Orrindale?"],
        briefs=[("ORD-NEXT", "Orrindale", 3100.0)],
        repeats=2,
    )
    assert report["scale"] == 1
    assert report["corpus_generation_seconds"] >= 0.0
    assert set(report["fields"]) == {"Orrindale", "Vessra South"}
    assert report["documents"] > 0
    assert report["npt_events"] > 0
    assert report["ask"]["samples"] == 2 * 2
    assert report["brief"]["samples"] == 2 * 1
