"""The final CLI surface: the global `--narrator` flag, per-command `--json`, `npt --since`
in place of the old `digest` command, the removed prototype-era flags, and the audit log
every `ask` and `brief` appends to (`audit.jsonl` inside the workspace).

`test_cli_workspace.py` covers `build`/`--db` (already removed before this module existed)
and the day-to-day ingest/status/ask/brief flow; this module covers what changed to reach
the final command surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wellbrief import audit, cli


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(h))
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    monkeypatch.delenv("WELLBRIEF_NARRATOR", raising=False)
    return h


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generated once (outside any workspace) and only ever read by `ingest`."""
    out = tmp_path_factory.mktemp("final-surface-corpus") / "corpus"
    assert cli.main(["corpus", "generate", "--out", str(out), "--scale", "1"]) == 0
    return out


# ---------------------------------------------------------------------------
# Removed prototype leftovers
# ---------------------------------------------------------------------------

def test_digest_is_no_longer_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["digest", "--since", "2026-01-01"])
    assert "invalid choice" in capsys.readouterr().err


def test_embed_backend_flag_is_gone(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--embed-backend", "offline", "status"])
    err = capsys.readouterr().err
    assert "--embed-backend" not in err  # swallowed as unknown, not accepted


def test_llm_backend_flag_is_gone(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--llm-backend", "offline", "status"])
    err = capsys.readouterr().err
    assert "--llm-backend" not in err


def test_json_is_a_per_command_flag_not_global(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--json` before the subcommand name is no longer recognised; it belongs after it."""
    with pytest.raises(SystemExit):
        cli.main(["--json", "status"])
    err = capsys.readouterr().err
    assert "invalid choice: '--json'" in err or "unrecognized arguments" in err

    assert cli.main(["status", "--json"]) == 0
    json.loads(capsys.readouterr().out)  # parses cleanly as JSON


def test_brief_has_no_json_flag_of_its_own(capsys: pytest.CaptureFixture[str]) -> None:
    """`brief` uses `--format json` instead of a `--json` flag."""
    with pytest.raises(SystemExit):
        cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100", "--json"])
    assert "unrecognized arguments" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Global --narrator / WELLBRIEF_NARRATOR
# ---------------------------------------------------------------------------

def test_narrator_defaults_to_offline() -> None:
    assert cli.build_parser().parse_args(["status"]).narrator == "offline"


def test_narrator_env_var_sets_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_NARRATOR", "llm")
    assert cli.build_parser().parse_args(["status"]).narrator == "llm"
    # an explicit flag still wins over the environment variable
    assert cli.build_parser().parse_args(["--narrator", "offline", "status"]).narrator == "offline"


def test_narrator_llm_without_a_configured_host_exits_cleanly(
        home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The `llm` narrator is implemented, but every command still needs
    `WELLBRIEF_LLM_BASE_URL` to build it; without one, each exits 2 with a clean,
    traceback-free message rather than crashing or silently falling back."""
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    assert cli.main(["--narrator", "llm", "ask", "What happened on ORD-101?"]) == 2
    err = capsys.readouterr().err
    assert "WELLBRIEF_LLM_BASE_URL" in err and "Traceback" not in err

    assert cli.main(["--narrator", "llm", "brief", "--field", "Orrindale",
                     "--well", "ORD-NEXT", "--td", "3100"]) == 2
    err = capsys.readouterr().err
    assert "WELLBRIEF_LLM_BASE_URL" in err and "Traceback" not in err

    assert cli.main(["--narrator", "llm", "eval", "--suite", "original"]) == 2
    err = capsys.readouterr().err
    assert "WELLBRIEF_LLM_BASE_URL" in err and "Traceback" not in err


def test_narrator_choices_reject_anything_else(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--narrator", "remote", "status"])
    assert "invalid choice" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End to end: corpus generate -> ingest -> status -> ask -> brief -> npt --since -> patterns
# ---------------------------------------------------------------------------

def test_final_surface_end_to_end(home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir), "--json"]) == 0
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["documents"] > 0

    assert cli.main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert {f["field"] for f in status["fields"]} == {"Orrindale", "Vessra South"}

    assert cli.main(["ask", "What happened in the 12 1/4 inch section on Orrindale", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is False
    assert answer["citations"]

    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100",
                     "--format", "json"]) == 0
    brief_payload = json.loads(capsys.readouterr().out)
    assert brief_payload["not_verified"] is False

    assert cli.main(["npt", "--since", "2000-01-01", "--json"]) == 0
    npt_payload = json.loads(capsys.readouterr().out)
    assert npt_payload["since"] == "2000-01-01"
    assert npt_payload["event_count"] > 0

    assert cli.main(["patterns", "--field", "Orrindale", "--json"]) == 0
    patterns_payload = json.loads(capsys.readouterr().out)
    assert isinstance(patterns_payload, list)


def test_npt_since_excludes_events_before_that_date(
        home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    assert cli.main(["npt", "--json"]) == 0
    everything = json.loads(capsys.readouterr().out)
    assert everything["event_count"] > 0

    # Far beyond the synthetic corpus's own timeline: --since excludes every event.
    assert cli.main(["npt", "--since", "2999-01-01", "--json"]) == 0
    nothing = json.loads(capsys.readouterr().out)
    assert nothing["event_count"] == 0
    assert nothing["since"] == "2999-01-01"


def test_ask_exits_0_even_when_it_abstains(home: Path, corpus_dir: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()
    assert cli.main(["ask", "What happened on ORD-199?", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is True
    assert answer["citations"] == []


def test_brief_still_exits_2_on_verification_failure_and_3_on_a_ledger_only_field(
        home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit codes unchanged by the CLI consolidation: 3 when `brief` cannot run at all
    (a field with NPT-ledger rows but no daily reports), 0 otherwise even on a verified
    empty register. The verification-failure path (exit 2) is exercised in
    `test_riskbrief.py` at the `verify_brief`/`render_markdown` level; this test only
    confirms the CLI still surfaces exit 3 for the documented ledger-only case."""
    folder = tmp_path / "csv-only"
    folder.mkdir()
    (folder / "ledger.csv").write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,12.5,Orrindale\n", encoding="utf-8")
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()

    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100"]) == 3
    err = capsys.readouterr().err
    assert "no daily drilling reports" in err
    # nothing to audit: the brief was never built
    assert audit.read_lines(home / "default" / "audit.jsonl") == []


def test_brief_exits_2_on_verification_failure(
        home: Path, corpus_dir: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """CLI wiring for the verification-failure exit code: a failing `verify_brief` check
    (forced here rather than reproduced from real store content, which `test_riskbrief.py`
    already covers at the `verify_brief`/`render_markdown` level) must still exit 2, label
    the text output, and land in the audit log as a failed verification."""
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    def fake_verify_brief(brief: object, store: object) -> dict[str, object]:
        return {"citations_checked": 1, "problems": ["forced failure"], "ok": False}

    monkeypatch.setattr(cli.riskbrief, "verify_brief", fake_verify_brief)

    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100"]) == 2
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out

    lines = audit.read_lines(home / "default" / "audit.jsonl")
    assert len(lines) == 1
    assert lines[0]["command"] == "brief"
    assert lines[0]["verification"] == {"ok": False, "reasons": 1}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_ask_appends_an_audit_line(home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    audit_path = home / "default" / "audit.jsonl"
    assert not audit_path.exists()

    question = "What happened in the 12 1/4 inch section on Orrindale"
    assert cli.main(["ask", question]) == 0
    capsys.readouterr()

    lines = audit.read_lines(audit_path)
    assert len(lines) == 1
    record = lines[0]
    assert record["command"] == "ask"
    assert record["parameters"] == {"question": question, "fields": ["Orrindale"]}
    assert record["narrator"] == "offline"
    assert len(record["corpus_hash"]) == 64
    assert record["cited_document_ids"]
    assert record["verification"] == {"ok": True, "reasons": 0}
    assert record["abstained"] is False
    assert record["ts"].endswith("Z")


def test_ask_audit_line_records_an_abstention(home: Path, corpus_dir: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()
    assert cli.main(["ask", "What happened on ORD-199?"]) == 0
    capsys.readouterr()

    lines = audit.read_lines(home / "default" / "audit.jsonl")
    assert len(lines) == 1
    assert lines[0]["abstained"] is True
    assert lines[0]["cited_document_ids"] == []


def test_brief_appends_an_audit_line(home: Path, corpus_dir: Path,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100"]) == 0
    capsys.readouterr()

    lines = audit.read_lines(home / "default" / "audit.jsonl")
    assert len(lines) == 1
    record = lines[0]
    assert record["command"] == "brief"
    assert record["parameters"] == {
        "field": "Orrindale", "well": "ORD-NEXT", "td_m": 3100.0,
        "rig": None, "mwd": None, "spread_rate_usd_per_day": 48_000.0, "format": "text",
    }
    assert record["narrator"] == "offline"
    assert len(record["corpus_hash"]) == 64
    assert record["cited_document_ids"]
    assert record["verification"] == {"ok": True, "reasons": 0}
    assert record["ts"].endswith("Z")


def test_ask_and_brief_append_to_the_same_audit_log_in_order(
        home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()
    assert cli.main(["ask", "What happened on ORD-101?"]) == 0
    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100"]) == 0
    capsys.readouterr()

    lines = audit.read_lines(home / "default" / "audit.jsonl")
    assert [line["command"] for line in lines] == ["ask", "brief"]


def test_audit_log_lives_at_the_documented_workspace_path(
        home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--workspace", "review", "ingest", str(corpus_dir)]) == 0
    capsys.readouterr()
    assert cli.main(["--workspace", "review", "ask", "What happened on ORD-101?"]) == 0
    capsys.readouterr()
    assert (home / "review" / "audit.jsonl").exists()
    assert not (home / "default" / "audit.jsonl").exists()
