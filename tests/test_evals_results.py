"""Result metadata: where the numbers came from, never which machine or whose home directory."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from wellbrief.evals import results

SENTINEL = "sentinel-host-7f3a"
ABSOLUTE = re.compile(r"(^|[\s\"'=])/(Users|home|private|var|tmp)\b")


def _strings(obj: object) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in _strings(k) + _strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


def test_metadata_never_contains_the_host_name(monkeypatch: pytest.MonkeyPatch) -> None:
    real = platform.node()
    monkeypatch.setattr(socket, "gethostname", lambda: SENTINEL)
    monkeypatch.setattr(platform, "node", lambda: SENTINEL)
    meta = results.metadata(["eval", "--suite", "original"], ["original"], [20260731], {"risk_filters": True},
                            "0.1.0")
    text = json.dumps(meta)
    assert SENTINEL not in text
    if len(real) >= 6:
        assert real not in text
    for key in ("wellbrief_version", "date_utc", "command", "seeds", "suites", "python", "hardware", "git"):
        assert key in meta
    assert meta["hardware"]["cpu"]
    assert meta["command"] == "wellbrief eval --suite original"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["date_utc"])


def test_command_line_paths_are_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    inside = tmp_path / "results" / "run.json"
    outside = Path(tempfile.gettempdir()).parent / "elsewhere" / "run.json"
    line = results.command_line(["eval", "--out", str(inside), f"--out={inside}", "--out", str(outside),
                                 "--seeds", "7"])
    assert line == ("wellbrief eval --out results/run.json --out=results/run.json --out <outside>/run.json "
                    "--seeds 7")


def test_git_state_outside_a_checkout(tmp_path: Path) -> None:
    assert results.git_state(tmp_path) == {"sha": "unknown", "dirty": "unknown"}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_dirty_flag_ignores_untracked_results_only(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    (repo / "src" / "wellbrief").mkdir(parents=True)
    (repo / "src" / "wellbrief" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=test", "-c", "user.email=", "-c", "commit.gpgsign=false",
         "commit", "-qm", "init")
    assert results.git_state(repo)["dirty"] is False

    # a first result file, untracked, does not make the next run dirty
    (repo / "results" / "v0.1.0").mkdir(parents=True)
    (repo / "results" / "v0.1.0" / "original-first-run.json").write_text("{}", encoding="utf-8")
    state = results.git_state(repo)
    assert state["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", state["sha"])

    (repo / "src" / "wellbrief" / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert results.git_state(repo)["dirty"] is True
    (repo / "src" / "wellbrief" / "new.py").unlink()
    (repo / "src" / "wellbrief" / "mod.py").write_text("x = 3\n", encoding="utf-8")
    assert results.git_state(repo)["dirty"] is True


def test_git_state_of_this_checkout() -> None:
    state = results.git_state()
    if state["sha"] != "unknown":
        assert re.fullmatch(r"[0-9a-f]{40}", state["sha"])
        assert isinstance(state["dirty"], bool)


def test_written_results_carry_no_absolute_paths(tmp_path: Path) -> None:
    home, tmp = str(Path.home()), tempfile.gettempdir()
    report = {
        "metadata": results.metadata(["eval", "--out", str(tmp_path / "r.json")], ["extended"], [1], {},
                                     "0.1.0"),
        "results": [
            {"detail": f"error: OSError: cannot read {tmp}/wellbrief-eval-x/corpus/_truth.json"},
            {"detail": f"error: FileNotFoundError: {home}/Documents/somewhere/file.txt"},
            {"detail": f"error: {os.path.realpath(tmp)}/abc and /home/someone/y and /private/etc/z"},
            {"detail": 'section 12 1/4" at 2,660 m and 17 1/2"'},
        ],
    }
    path = tmp_path / "out" / "r.json"
    results.write(report, path)
    written = json.loads(path.read_text(encoding="utf-8"))
    for s in _strings(written):
        assert not ABSOLUTE.search(s), s
        assert home not in s
    assert written["results"][3]["detail"] == 'section 12 1/4" at 2,660 m and 17 1/2"'
    assert "<tmp>/wellbrief-eval-x/corpus" in written["results"][0]["detail"]


@pytest.mark.parametrize("text", [
    "error: /usr/local/lib/python3.12/site-packages/wellbrief/store.py line 3",
    "error: cannot open /app/data/x.db",
    "error: /github/workspace/results/r.json",
    "error: /srv/a",
    "error: ../../somewhere/else/file.txt",
    f"error: {sys.prefix}/lib/python3.12/x.py",
    f"error: {sys.base_prefix}/lib/python3.12/x.py",
    f"error: {Path(results.__file__).resolve().parents[1]}/store.py",
])
def test_scrub_rewrites_any_absolute_path(text: str) -> None:
    out = results.scrub(text)
    assert not re.search(r"(^|[\s\"'=:])(/[A-Za-z_]|\.\./)", out), out
    assert out.startswith("error: ")


@pytest.mark.parametrize("text", [
    'hole 12 1/4" and 13 3/8" casing, 9 5/8" liner',
    "cost $60,000 /day and hours/day, and/or",
    "wellbrief eval --out <outside>/run.json",
    "<tmp>/wellbrief-eval-x and ~/Documents",
])
def test_scrub_leaves_ordinary_text_alone(text: str) -> None:
    assert results.scrub(text) == text
