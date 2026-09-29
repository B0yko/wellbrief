"""`wellbrief demo`: corpus generate + ingest + index + serve as one command.

`WELLBRIEF_DEMO_WELLS_PER_FIELD` is a test-only environment variable (see
`demo.wells_per_field_from_env`), read only by `demo.prepare()`: it shrinks the generated corpus
to that many wells per field before it is written, hashed or ingested, so these tests build and
ingest a handful of documents instead of the full ~1,450-document corpus. No public CLI flag sets
it, and a real `wellbrief demo` invocation never does either: an ordinary run always demonstrates
the full corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from wellbrief import cli, demo
from wellbrief.selfcheck import BRIEF_CHECKS
from wellbrief.workspace import Workspace

# ---------------------------------------------------------------------------
# prepare(): reuse, rebuild, and the parameters that must invalidate reuse
# ---------------------------------------------------------------------------


@pytest.fixture
def small_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two wells per field (see the module docstring): fast enough for every test below."""
    monkeypatch.setenv("WELLBRIEF_DEMO_WELLS_PER_FIELD", "2")


def test_wells_per_field_from_env_reads_a_positive_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELLBRIEF_DEMO_WELLS_PER_FIELD", raising=False)
    assert demo.wells_per_field_from_env() is None
    monkeypatch.setenv("WELLBRIEF_DEMO_WELLS_PER_FIELD", "3")
    assert demo.wells_per_field_from_env() == 3
    monkeypatch.setenv("WELLBRIEF_DEMO_WELLS_PER_FIELD", "0")
    assert demo.wells_per_field_from_env() is None
    monkeypatch.setenv("WELLBRIEF_DEMO_WELLS_PER_FIELD", "not-a-number")
    assert demo.wells_per_field_from_env() is None


def test_prepare_builds_a_fresh_workspace_with_both_fields(tmp_path: Path, small_corpus: None) -> None:
    ws = Workspace(home=tmp_path / "home", name=demo.WORKSPACE_NAME)
    build = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    assert build.reused is False
    assert build.fields == ("Orrindale", "Vessra South")
    assert build.documents > 0
    assert ws.db_path.is_file()
    assert (ws.root / "demo.json").is_file()
    store = ws.open_store()
    try:
        assert set(store.field_names()) == {"Orrindale", "Vessra South"}
    finally:
        store.close()


def test_prepare_reuses_a_matching_workspace(tmp_path: Path, small_corpus: None) -> None:
    ws = Workspace(home=tmp_path / "home", name=demo.WORKSPACE_NAME)
    first = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    built_at = (ws.root / "demo.json").read_text(encoding="utf-8")
    second = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    assert second.reused is True
    assert second.corpus_hash == first.corpus_hash
    assert second.documents == first.documents
    # nothing was rewritten: the marker on disk is byte-identical
    assert (ws.root / "demo.json").read_text(encoding="utf-8") == built_at


def test_prepare_rebuild_forces_a_fresh_build_even_when_unchanged(tmp_path: Path,
                                                                   small_corpus: None) -> None:
    ws = Workspace(home=tmp_path / "home", name=demo.WORKSPACE_NAME)
    first = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    again = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=True)
    assert again.reused is False
    assert again.corpus_hash == first.corpus_hash


@pytest.mark.parametrize(("seed", "formats"), [(1, "mixed"), (20260731, "txt")])
def test_prepare_does_not_reuse_across_a_different_seed_or_format(tmp_path: Path, small_corpus: None,
                                                                  seed: int, formats: str) -> None:
    ws = Workspace(home=tmp_path / "home", name=demo.WORKSPACE_NAME)
    demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    changed = demo.prepare(ws, seed=seed, formats=formats, rebuild=False)
    assert changed.reused is False


def test_prepare_does_not_reuse_across_a_different_wellbrief_version(tmp_path: Path, small_corpus: None,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    ws = Workspace(home=tmp_path / "home", name=demo.WORKSPACE_NAME)
    demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    monkeypatch.setattr(demo, "__version__", "9.9.9")
    changed = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    assert changed.reused is False


def test_prepare_ignores_a_corrupt_marker_and_rebuilds(tmp_path: Path, small_corpus: None) -> None:
    ws = Workspace(home=tmp_path / "home", name=demo.WORKSPACE_NAME)
    demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    (ws.root / "demo.json").write_text("not json", encoding="utf-8")
    rebuilt = demo.prepare(ws, seed=20260731, formats="mixed", rebuild=False)
    assert rebuilt.reused is False


# ---------------------------------------------------------------------------
# CLI wiring (argument parsing only -- `demo` blocks in `serve_forever`,
# exactly like `serve`, so its actual run is covered by the subprocess
# integration test below, not by calling `cli.main` in-process)
# ---------------------------------------------------------------------------


def test_the_demo_subcommand_defaults() -> None:
    args = cli.build_parser().parse_args(["demo"])
    assert args.func is cli.cmd_demo
    assert args.no_browser is False
    assert args.rebuild is False
    assert args.formats == "mixed"
    assert args.seed == 20260731
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_the_demo_subcommand_accepts_its_flags() -> None:
    args = cli.build_parser().parse_args([
        "demo", "--no-browser", "--rebuild", "--formats", "txt", "--seed", "7",
        "--host", "0.0.0.0", "--port", "9001",
    ])
    assert args.no_browser is True
    assert args.rebuild is True
    assert args.formats == "txt"
    assert args.seed == 7
    assert args.host == "0.0.0.0"
    assert args.port == 9001


def test_the_demo_subcommand_rejects_an_unknown_format() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["demo", "--formats", "pdf"])


# ---------------------------------------------------------------------------
# cmd_demo's browser-opening step, in process: prepare() and create_server()
# are stubbed out so these run in milliseconds and never bind a real socket,
# leaving only the behaviour under test -- what happens to webbrowser.open()'s
# result and its failures.
# ---------------------------------------------------------------------------


def _stub_build(ws: Workspace) -> demo.DemoBuild:
    return demo.DemoBuild(ws, True, 20260731, "mixed", "deadbeef", ("Orrindale",), 3, 0.01)


class _StubHTTPD:
    """A fake `create_server()` result: reports a fixed address, and its `serve_forever`
    raises `KeyboardInterrupt` immediately, exactly like a real one does on Ctrl-C, so
    `cmd_demo` runs to completion instead of blocking."""

    server_name = "127.0.0.1"
    server_port = 0

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def serve_forever(self, poll_interval: float = 0.2) -> None:
        self._calls.append("serve_forever")
        raise KeyboardInterrupt

    def server_close(self) -> None:
        self._calls.append("server_close")


def _demo_args(*, no_browser: bool) -> argparse.Namespace:
    return argparse.Namespace(seed=20260731, formats="mixed", rebuild=False, narrator="offline",
                              host="127.0.0.1", port=0, no_browser=no_browser)


def _stub_out_prepare_and_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                 calls: list[str]) -> None:
    monkeypatch.setenv("WELLBRIEF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    monkeypatch.setattr(cli.demo_mod, "prepare", lambda ws, **kw: _stub_build(ws))
    monkeypatch.setattr(cli.server_mod, "create_server",
                       lambda ws, narrator, host, port: _StubHTTPD(calls))


@pytest.mark.parametrize("raised", [OSError("no runnable browser"), webbrowser.Error("no browser")])
def test_cmd_demo_survives_a_browser_that_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    raised: Exception,
) -> None:
    """A server already up and listening must not be torn down by a browser-launch failure,
    whether webbrowser.open() reports it the documented way (webbrowser.Error) or the way the
    stdlib's own browser controllers sometimes do instead, letting a plain OSError propagate."""
    calls: list[str] = []
    _stub_out_prepare_and_server(monkeypatch, tmp_path, calls)

    def _raise(url: str, new: int = 0, autoraise: bool = True) -> bool:
        raise raised
    monkeypatch.setattr(cli.webbrowser, "open", _raise)

    exit_code = cli.cmd_demo(_demo_args(no_browser=False))

    assert exit_code == 0
    assert calls == ["serve_forever", "server_close"]
    assert "could not open a browser" in capsys.readouterr().err


def test_cmd_demo_opens_the_browser_on_the_reported_url(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _stub_out_prepare_and_server(monkeypatch, tmp_path, calls)
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

    exit_code = cli.cmd_demo(_demo_args(no_browser=False))

    assert exit_code == 0
    assert opened == ["http://127.0.0.1:0/"]


def test_cmd_demo_no_browser_never_calls_webbrowser_open(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _stub_out_prepare_and_server(monkeypatch, tmp_path, calls)
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

    exit_code = cli.cmd_demo(_demo_args(no_browser=True))

    assert exit_code == 0
    assert opened == []


# ---------------------------------------------------------------------------
# Integration: the real `wellbrief demo --no-browser` command, end to end
# ---------------------------------------------------------------------------


_LISTENING_RE = re.compile(r"listening on (http://\S+/)")


def _pump_lines(stream: Any, sink: queue.Queue[str | None]) -> None:
    for line in stream:
        sink.put(line)
    sink.put(None)


def _wait_for_listening_url(proc: subprocess.Popen[str], timeout: float) -> str:
    """Read `proc`'s stdout until it reports the URL it bound, or raise after `timeout` seconds
    (including if the process exits first). Read from a background thread so a stuck process
    cannot hang this call past `timeout`."""
    assert proc.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()
    thread = threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
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
    raise AssertionError("wellbrief demo never reported a listening URL; output so far:\n"
                        + "".join(seen))


def _json(base: str, path: str, *, method: str = "GET", body: dict[str, Any] | None = None,
          expect: int = 200) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(base.rstrip("/") + path, data=data, method=method,
                 headers={"Content-Type": "application/json", "Origin": base.rstrip("/")}
                 if data is not None else {})
    try:
        with urlopen(req, timeout=10) as resp:
            status, raw = resp.status, resp.read()
    except HTTPError as exc:
        status, raw = exc.code, exc.read()
    assert status == expect, f"{method} {path} -> {status}: {raw.decode('utf-8', 'replace')}"
    return json.loads(raw.decode("utf-8"))


def _download(base: str, path: str) -> tuple[int, dict[str, str]]:
    with urlopen(base.rstrip("/") + path, timeout=10) as resp:
        return resp.status, dict(resp.headers)


def test_demo_no_browser_end_to_end_on_a_reduced_corpus(tmp_path: Path) -> None:
    env = {**os.environ, "WELLBRIEF_HOME": str(tmp_path / "home"),
          "WELLBRIEF_DEMO_WELLS_PER_FIELD": "2", "PYTHONUNBUFFERED": "1"}
    env.pop("WELLBRIEF_WORKSPACE", None)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "wellbrief", "demo", "--no-browser", "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    try:
        base = _wait_for_listening_url(proc, timeout=30)

        status = _json(base, "/api/status")
        assert status["workspace"] == "demo"
        assert {f["field"] for f in status["fields"]} == {"Orrindale", "Vessra South"}

        ask = _json(base, "/api/ask", method="POST",
                    body={"question": "What non-productive time was recorded on Orrindale?"})
        assert ask["abstained"] is False
        assert ask["citations"], "expected at least one citation on a field-scoped NPT question"
        doc_id = ask["citations"][0]["doc_id"]

        for field_name, well, td in BRIEF_CHECKS:
            brief = _json(base, "/api/brief", method="POST",
                          body={"field": field_name, "well": well, "td": td})
            assert brief["field_name"] == field_name
            assert brief["well_name"] == well

        doc = _json(base, f"/api/doc/{doc_id}")
        assert doc["doc_id"] == doc_id
        assert doc["text"]

        field_name, well, td = BRIEF_CHECKS[0]
        query = f"?field={field_name}&well={well}&td={td}"
        md_status, md_headers = _download(base, "/api/brief.md" + query)
        assert md_status == 200
        assert "attachment" in md_headers["Content-Disposition"]
        json_status, json_headers = _download(base, "/api/brief.json" + query)
        assert json_status == 200
        assert "attachment" in json_headers["Content-Disposition"]
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            returncode = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("wellbrief demo did not shut down after SIGINT")
    assert returncode == 0
