"""The CLI's workspace flow: `corpus generate` -> `ingest` -> `index` -> `status` -> `ask` / `brief`.

There is no `build` command or `--db` flag: the
workspace is chosen with `--workspace` / `WELLBRIEF_WORKSPACE`, under
`$WELLBRIEF_HOME` / `WELLBRIEF_HOME` (default `~/.wellbrief`, never touched by
these tests -- `WELLBRIEF_HOME` is always pointed at a tmp dir).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wellbrief import cli


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("WELLBRIEF_HOME", str(h))
    monkeypatch.delenv("WELLBRIEF_WORKSPACE", raising=False)
    return h


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generated once (outside any workspace) and only ever read by `ingest`, so every
    test in this module can safely share it."""
    out = tmp_path_factory.mktemp("cli-corpus") / "corpus"
    assert cli.main(["corpus", "generate", "--out", str(out)]) == 0
    return out


def test_build_is_no_longer_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["build"])
    assert "invalid choice" in capsys.readouterr().err


def test_the_db_flag_is_gone(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--db", "x.db", "status"])
    # argparse no longer recognises --db at all: it is swallowed as an unknown
    # option and "x.db" is left over, tripping the subcommand choice instead.
    err = capsys.readouterr().err
    assert "--db" not in err
    assert "invalid choice: 'x.db'" in err or "unrecognized arguments" in err


def test_ingest_status_ask_brief_end_to_end(
        home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    ingested = capsys.readouterr().out
    assert "1446 file(s) seen, 1446 ingested" in ingested
    assert "documents: 1446" in ingested
    assert "Indexed:" in ingested

    # The workspace's on-disk layout: one db file, plus a bm25 index per field directory.
    ws_root = home / "default"
    assert (ws_root / "wellbrief.db").exists()
    for field_dir_name in ("orrindale", "vessra-south"):
        field_dir = ws_root / "fields" / field_dir_name
        assert (field_dir / "bm25.json").exists()
        assert (field_dir / "vectors.f32").exists()
        assert (field_dir / "manifest.json").exists()

    assert cli.main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["workspace"] == "default"
    assert status["network_mode"] == "offline"
    fields = {f["field"]: f for f in status["fields"]}
    assert set(fields) == {"Orrindale", "Vessra South"}
    assert fields["Orrindale"]["documents"] > 0
    assert fields["Orrindale"]["chunks"] > 0
    assert fields["Orrindale"]["npt_events"] > 0
    assert fields["Orrindale"]["index_stale"] is False
    assert fields["Orrindale"]["embedder"] == "offline"

    assert cli.main(["index", "--field", "Orrindale", "--json"]) == 0
    reindexed = json.loads(capsys.readouterr().out)
    assert reindexed["fields"][0]["field"] == "Orrindale"

    assert cli.main(["ask", "What happened in the 12 1/4 inch section on Orrindale", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is False
    assert answer["citations"]
    assert all("page" in c for c in answer["citations"])

    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100"]) == 0
    brief_out = capsys.readouterr().out
    assert "NOT VERIFIED" not in brief_out
    assert "Decision support built from the offset archive" in brief_out


def test_index_rejects_an_unknown_field(home: Path, corpus_dir: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["ingest", str(corpus_dir)])
    capsys.readouterr()
    assert cli.main(["index", "--field", "Nowhereland"]) == 2
    assert "unknown field" in capsys.readouterr().err


def test_status_on_an_empty_workspace_does_not_crash(home: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "none ingested yet" in out


def test_the_workspace_flag_and_env_var_select_different_workspaces(
        home: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.main(["--workspace", "field-review", "ingest", str(corpus_dir)]) == 0
    capsys.readouterr()
    assert (home / "field-review" / "wellbrief.db").exists()
    assert not (home / "default" / "wellbrief.db").exists()

    monkeypatch.setenv("WELLBRIEF_WORKSPACE", "from-env")
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()
    assert (home / "from-env" / "wellbrief.db").exists()

    # The flag still wins over the environment variable.
    assert cli.main(["--workspace", "flag-wins", "status"]) == 0
    status = capsys.readouterr().out
    assert "flag-wins" in status


def test_ingest_dry_run_writes_nothing(home: Path, corpus_dir: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Would ingest" in out
    assert cli.main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["fields"] == []


def test_ingest_prune_removes_a_deleted_file(home: Path, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "prune-corpus"
    assert cli.main(["corpus", "generate", "--out", str(folder), "--scale", "1"]) == 0
    capsys.readouterr()
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    removed = next(folder.glob("DDR-ORD-101-*.txt"))
    removed.unlink()

    assert cli.main(["ingest", str(folder), "--prune"]) == 0
    out = capsys.readouterr().out
    assert "1 pruned" in out


def test_corpus_generate_formats_and_ledger_csv(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "mixed"
    assert cli.main(["corpus", "generate", "--out", str(out), "--scale", "1",
                     "--formats", "mixed", "--ledger-csv"]) == 0
    printed = capsys.readouterr().out
    assert "format mixed" in printed
    assert "NPT ledger CSV" in printed
    assert (out / "npt-ledger.csv").exists()
    assert any(out.glob("EOWR-*.pdf"))
    assert any(out.glob("INC-*.docx"))
    assert any(out.glob("DDR-*.txt"))


def test_brief_exits_3_when_a_field_has_ledger_rows_but_no_ddrs(
        home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "csv-only"
    folder.mkdir()
    (folder / "ledger.csv").write_text(
        "well,date,code,hours,field\nORD-501,2026-01-01,STUCK_PIPE,12.5,Orrindale\n", encoding="utf-8")
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()

    assert cli.main(["brief", "--field", "Orrindale", "--well", "ORD-NEXT", "--td", "3100"]) == 3
    err = capsys.readouterr().err
    assert "no daily drilling reports" in err


# ---------------------------------------------------------------------------
# Configuration: workspace wellbrief.toml, ingest --config, and the settings
# CLI/env precedence.
# ---------------------------------------------------------------------------


def test_ingest_with_a_missing_config_file_exits_cleanly_with_no_traceback(
        home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    assert cli.main(["ingest", str(folder), "--config", str(tmp_path / "no-such.toml")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("wellbrief: ")
    assert "Traceback" not in err


def test_a_malformed_workspace_toml_exits_cleanly_on_every_command_that_reads_settings(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = home / "default"
    ws.mkdir(parents=True)
    (ws / "wellbrief.toml").write_text("[risk]\nmin_lft = 2.0\n", encoding="utf-8")

    assert cli.main(["npt"]) == 2
    err = capsys.readouterr().err
    assert "wellbrief.toml" in err and "min_lft" in err and "Traceback" not in err


def test_ingest_examples_alt_template_finds_its_own_wellbrief_toml(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The example used by `test_examples.py`'s equivalence test also works end to end
    through the CLI, with no `--config` flag: `ingest` reads a folder's own `wellbrief.toml`
    on its own."""
    alt_template = Path(__file__).resolve().parents[1] / "examples" / "alt-template"
    assert cli.main(["ingest", str(alt_template)]) == 0
    out = capsys.readouterr().out
    assert "4 file(s) seen, 4 ingested" in out
    assert "NPT events: 3" in out

    assert cli.main(["npt", "--field", "Marrow Deep", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event_count"] == 3
    assert {row["code"] for row in payload["by_code"]} == {"STUCK_PIPE"}


def test_ask_finds_mitigations_on_a_field_ingested_with_its_own_toml(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`ask` quotes mitigations for a field ingested through its own `wellbrief.toml` (an
    `ingest --config` file, or here the ingested folder's own file, per
    `test_ingest_examples_alt_template_finds_its_own_wellbrief_toml`) even though the bare
    workspace this test ingests into has no `wellbrief.toml` of its own at all: the miner reads
    each document's lessons, recommendations and corrective actions from the settings they were
    actually parsed with at ingest time, not from whatever the workspace's own configuration
    (here, none) currently says. Regression test: before this was fixed, the miner re-parsed
    `doc.text` at query time with the workspace's settings, which do not know the alt template's
    `LESSONS` / `RECOMMENDATIONS FOR NEXT WELL` headings, and silently returned no mitigations."""
    alt_template = Path(__file__).resolve().parents[1] / "examples" / "alt-template"
    assert cli.main(["ingest", str(alt_template)]) == 0
    capsys.readouterr()

    assert not (home / "default" / "wellbrief.toml").exists()
    assert cli.main(["ask", "What went wrong with stuck pipe on MRD-201 in the Marrow Deep field?",
                     "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["mitigations"], "expected at least one mitigation quoted from the alt template"
    assert any("Hold at least" in m["text"] for m in answer["mitigations"])


def test_workspace_toml_retrieval_settings_flow_through_ask(
        home: Path, corpus_dir: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """`[retrieval] top_k` in the workspace's own `wellbrief.toml` reaches `qa.ask` when `ask`
    is not given its own `--top-k`, and an explicit `--top-k` still wins over it."""
    ws = home / "default"
    ws.mkdir(parents=True)
    (ws / "wellbrief.toml").write_text("[retrieval]\ntop_k = 3\n", encoding="utf-8")
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    captured: dict[str, object] = {}
    real_ask = cli.ask_qa

    def spy(question: str, store: object, searcher: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_ask(question, store, searcher, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "ask_qa", spy)

    assert cli.main(["ask", "What went wrong on ORD-105?"]) == 0
    assert captured["top_k"] == 3

    assert cli.main(["ask", "What went wrong on ORD-105?", "--top-k", "9"]) == 0
    assert captured["top_k"] == 9


def test_spread_rate_env_var_is_used_when_no_flag_is_given(
        home: Path, monkeypatch: pytest.MonkeyPatch, corpus_dir: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    capsys.readouterr()

    monkeypatch.setenv("WELLBRIEF_SPREAD_RATE_USD_PER_DAY", "60000")
    assert cli.main(["npt", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spread_rate_usd_per_day"] == 60000.0

    # an explicit --spread-rate still wins over the environment variable
    assert cli.main(["--spread-rate", "12000", "npt", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spread_rate_usd_per_day"] == 12000.0
