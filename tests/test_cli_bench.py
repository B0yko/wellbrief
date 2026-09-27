"""`wellbrief bench`'s CLI wiring: `--scale` parsing and `cmd_bench` itself, with
`bench.run` replaced by a stub (the real run is exercised in `tests/test_bench.py` and by
the recorded measurement on a clean tree, never inside a fast unit test)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wellbrief import bench, cli


def test_scale_defaults_to_one() -> None:
    assert cli.build_parser().parse_args(["bench"]).scale == [1]


def test_scale_accepts_a_comma_separated_list() -> None:
    assert cli.build_parser().parse_args(["bench", "--scale", "1,10"]).scale == [1, 10]


def test_scale_rejects_a_value_out_of_range(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["bench", "--scale", "11"])
    assert "between 1 and" in capsys.readouterr().err


def test_scale_rejects_garbage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["bench", "--scale", "abc"])
    assert "comma-separated integers" in capsys.readouterr().err


def test_repeats_defaults_to_bench_warm_repeats() -> None:
    assert cli.build_parser().parse_args(["bench"]).repeats == bench.WARM_REPEATS


_FAKE_PAYLOAD: dict[str, Any] = {
    "metadata": {"scales": [1]},
    "demo_startup_seconds": {"mixed": 1.0, "txt": 1.0},
    "scales": [{"scale": 1}],
    "peak_rss_mib": 42.0,
}


def test_cmd_bench_calls_run_with_the_parsed_scale_and_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(scales: list[int], args: list[str], repeats: int, emit: Any) -> dict[str, Any]:
        seen["scales"] = scales
        seen["repeats"] = repeats
        return _FAKE_PAYLOAD

    monkeypatch.setattr(bench, "run", fake_run)
    args = cli.build_parser().parse_args(["bench", "--scale", "1,10", "--repeats", "3"])
    args.argv = ["bench", "--scale", "1,10", "--repeats", "3"]
    assert cli.cmd_bench(args) == 0
    assert seen == {"scales": [1, 10], "repeats": 3}


def test_cmd_bench_writes_the_out_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "run", lambda scales, args, repeats, emit: _FAKE_PAYLOAD)
    out = tmp_path / "bench.json"
    args = cli.build_parser().parse_args(["bench", "--out", str(out)])
    args.argv = ["bench", "--out", str(out)]
    assert cli.cmd_bench(args) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["peak_rss_mib"] == 42.0


def test_cmd_bench_reports_a_clean_error_instead_of_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_run(scales: list[int], args: list[str], repeats: int, emit: Any) -> dict[str, Any]:
        raise TimeoutError("demo never answered")

    monkeypatch.setattr(bench, "run", failing_run)
    args = cli.build_parser().parse_args(["bench"])
    args.argv = ["bench"]
    assert cli.cmd_bench(args) == 2
