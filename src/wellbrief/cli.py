"""Command line interface.

Every command needs no network access. `corpus generate` writes the
synthetic corpus and its ground truth to a directory; `ingest` loads a folder
of well files into a workspace's store; every other command works on that
workspace's store and its per-field index files (see `workspace.py`).
`--json` is a per-command flag on `ingest`, `index`, `status`, `ask`, `npt`,
`patterns`, `corpus generate` and `eval`, switching that command's output to
machine-readable JSON; `brief` has its own `--format text|md|json` instead.
Every `ask` and `brief` also appends one line to the workspace's
`audit.jsonl` (see `audit.py`). `main()` installs the process-level network
guard (see `netguard.py`) before running any command: loopback and Unix
sockets are always allowed, and the configured LLM host is allowed as well
while this run's narrator is `llm`. Everything else is blocked and counted;
`status` and `eval` report the count as "outbound connection attempts: N",
and `status` also reports the network mode (`offline`, `local-llm` or
`remote-llm`, from `egress.network_mode`). The `llm` narrator (`ask`, `brief`
and `eval --narrator llm`) is built from `WELLBRIEF_LLM_*` environment
variables (see `narrate.llm.narrator_from_env`); on any failure to build it,
or on a call it cannot complete or whose text fails verification, the
deterministic offline narrator answers instead (see `narrate.llm.LlmNarrator`
for the latter two). `selfcheck` (see `selfcheck.py`) runs every offline step
end to end in a disposable workspace of its own, then the network guard's
own positive control, and exits non-zero if any of it fails; it is the
command a network-disabled run proves against. `serve` (see `server.py`)
runs the same workspace's `ask`/`brief`/`npt`/`status` behind a local JSON
API and a static single-page UI, built from the same functions this module
calls, so its JSON output matches `--json` here; it also appends to the same
`audit.jsonl` for `ask` and `brief`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import (
    __version__,
    analytics,
    audit,
    corpus,
    egress,
    ingest as ingest_mod,
    netguard,
    riskbrief,
    selfcheck as selfcheck_mod,
    server as server_mod,
    workspace as workspace_mod,
)
from .corpus import MAX_SCALE, SEED
from .evals.cases import SUITES as EVAL_SUITES
from .narrate import NARRATORS, Narrator, OfflineNarrator, llm as narrate_llm
from .qa import ask as ask_qa
from .search import Searcher
from .settings import Settings, SettingsError, resolve_settings
from .store import Store
from .workspace import FieldIndex, Workspace


def _default_narrator() -> str:
    """`WELLBRIEF_NARRATOR`, else `"offline"`. Read fresh on every call (not cached at
    import time) so a test or a caller can change the environment between invocations."""
    return os.environ.get("WELLBRIEF_NARRATOR", "offline")


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace.resolve(getattr(args, "workspace", None))


def _settings(ws: Workspace, args: argparse.Namespace) -> Settings:
    """Resolve this command's settings: `<workspace>/wellbrief.toml`, overridden by whichever
    of this command's own flags carry a value (`--spread-rate`, `ask`'s `--top-k`, `brief`'s
    `--max-risks`), overridden in turn by the matching environment variable
    (`WELLBRIEF_SPREAD_RATE_USD_PER_DAY`; `top_k`/`max_risks` have none yet). `ingest` resolves
    its own, separate settings (`cmd_ingest`), since it alone also has `--config` and the
    ingested folder's own `wellbrief.toml` to fold in."""
    return resolve_settings(
        workspace_root=ws.root,
        spread_rate=getattr(args, "spread_rate", None),
        top_k=getattr(args, "top_k", None),
        max_risks=getattr(args, "max_risks", None),
    )


def _wire(args: argparse.Namespace) -> tuple[Workspace, Store, Searcher, Settings]:
    ws = _workspace(args)
    store = ws.open_store()
    st = _settings(ws, args)
    searcher = workspace_mod.wire_searcher(ws, store, st)
    return ws, store, searcher, st


def _emit(payload: Any, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(text)


def _netguard_allow_hosts(narrator: str) -> list[str]:
    """Extra host the process-level network guard should allow, beyond loopback and Unix
    sockets: the configured LLM host, and only while this invocation's narrator is `llm`.
    `WELLBRIEF_LLM_BASE_URL` is read directly rather than through `Settings`, because the
    guard is installed before any workspace or configuration file is resolved.

    Raises:
        ValueError: `WELLBRIEF_LLM_BASE_URL` is set but is not a URL with a host name, with
            no mention of the variable name (the caller adds that, so it reads the same
            whether this function rejected it or `netguard.install` did, for a host name
            this function extracted but that is not one `install` accepts). `main` turns it
            into a clean "wellbrief: ..." message and exit code 2, rather than a crash,
            since a malformed URL should still fail closed but not with a raw traceback.
    """
    if narrator != "llm":
        return []
    base_url = os.environ.get("WELLBRIEF_LLM_BASE_URL")
    if not base_url:
        return []
    try:
        host = urlsplit(base_url).hostname
    except ValueError as exc:
        raise ValueError(f"{base_url!r} is not a valid URL: {exc}") from exc
    if not host:
        raise ValueError(
            f"{base_url!r} has no host name; expected a URL such as 'http://127.0.0.1:8080'"
        )
    return [host]


def _build_narrator(command: str, narrator_name: str, ws: Workspace,
                    store: Store) -> tuple[Narrator | None, int | None]:
    """The narrator for this invocation: `OfflineNarrator` for `offline`, or the `llm` narrator
    built from `WELLBRIEF_LLM_*` (see `narrate.llm.narrator_from_env`) for `llm`.

    Returns:
        `(narrator, None)` on success; `(None, 2)`, having already printed a clean
        "wellbrief <command>: ..." message, when the `llm` narrator's configuration cannot be
        used at all (a missing or malformed `WELLBRIEF_LLM_BASE_URL`/`WELLBRIEF_LLM_MODEL`, or a
        bad budget or price variable). A call the narrator itself cannot complete, or whose text
        fails verification, is not this function's concern: `narrate.llm.LlmNarrator` falls back
        to the offline answer for that on its own.
    """
    if narrator_name == "offline":
        return OfflineNarrator(), None
    try:
        narrator: Narrator = narrate_llm.narrator_from_env(
            os.environ, ws.egress_path, known_field_names=frozenset(store.field_names()))
    except (narrate_llm.LLMConfigError, egress.InvalidBaseURL, ValueError) as exc:
        print(f"wellbrief {command}: {exc}", file=sys.stderr)
        return None, 2
    return narrator, None


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _coverage_lines(coverage: ingest_mod.Coverage, path: Path, dry_run: bool) -> list[str]:
    verb = "Would ingest" if dry_run else "Ingested"
    lines = [
        f"{verb} from {path}: {coverage.files_seen} file(s) seen, {coverage.ingested} ingested, "
        f"{coverage.unchanged} unchanged, {coverage.pruned} pruned",
        f"  documents: {coverage.documents}   chunks: {coverage.chunks}   "
        f"NPT events: {coverage.npt_events}",
    ]
    if coverage.duplicates_merged:
        lines.append(f"  CSV/DDR duplicates merged: {coverage.duplicates_merged}")
    for message in coverage.csv_row_errors:
        lines.append(f"  CSV row error: {message}")
    for message in coverage.csv_row_warnings:
        lines.append(f"  CSV row warning: {message}")
    for skipped in coverage.skipped:
        lines.append(f"  skipped {skipped.path}: {skipped.reason}")
    extracted = {k: v for k, v in coverage.ddr_extraction.items() if v is not None}
    if extracted:
        lines.append("  DDR extraction share: "
                     + ", ".join(f"{k} {v * 100:.0f}%" for k, v in extracted.items()))
    return lines


def cmd_ingest(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    try:
        coverage = ingest_mod.ingest_folder(store, args.path, field=args.field, config=args.config,
                                            workspace_root=ws.root, prune=args.prune, dry_run=args.dry_run)
    except OSError as exc:
        print(f"wellbrief ingest: {exc}", file=sys.stderr)
        store.close()
        return 2
    lines = _coverage_lines(coverage, args.path, args.dry_run)
    if not args.dry_run and coverage.fields:
        st = resolve_settings(workspace_root=ws.root, ingest_config_path=args.config,
                              ingest_folder=args.path)
        workspace_mod.check_field_slugs(store.field_names())
        built = [workspace_mod.rebuild_field_index(ws, store, f, bm25_k1=st.retrieval.k1,
                                                    bm25_b=st.retrieval.b)
                for f in sorted(coverage.fields)]
        if built:
            lines.append("Indexed: " + "; ".join(
                f"{i.field} ({i.manifest.chunks} chunks, {i.manifest.build_seconds:.3f}s)" for i in built))
    payload = {**coverage.to_dict(), "workspace": ws.name, "path": str(args.path), "dry_run": args.dry_run}
    _emit(payload, args.json, "\n".join(lines))
    store.close()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    st = _settings(ws, args)
    known = store.field_names()
    workspace_mod.check_field_slugs(known)
    if args.field and args.field not in known:
        print(f"wellbrief index: unknown field {args.field!r} (known: {', '.join(known) or '-'})",
              file=sys.stderr)
        store.close()
        return 2
    fields = [args.field] if args.field else known
    built: list[FieldIndex] = [workspace_mod.rebuild_field_index(ws, store, f, bm25_k1=st.retrieval.k1,
                                                                 bm25_b=st.retrieval.b)
                               for f in fields]
    payload = {
        "workspace": ws.name,
        "fields": [{"field": i.field, "documents": i.manifest.documents, "chunks": i.manifest.chunks,
                    "corpus_hash": i.manifest.corpus_hash, "embedder": i.manifest.embedder,
                    "build_seconds": i.manifest.build_seconds, "built_at": i.manifest.built_at}
                   for i in built],
    }
    lines = [f"Indexed {len(built)} field(s) in workspace '{ws.name}':"]
    lines += [f"  {i.field:<20} {i.manifest.documents:>5} docs  {i.manifest.chunks:>6} chunks"
             f"  {i.manifest.build_seconds:>7.3f}s  ({i.manifest.embedder})" for i in built]
    _emit(payload, args.json, "\n".join(lines))
    store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    payload = workspace_mod.status_payload(ws, store, args.narrator)
    rows = payload["fields"]
    lines = [f"workspace    : {ws.name} ({ws.root})", f"network mode : {payload['network_mode']}",
             f"outbound connection attempts: {payload['outbound_connection_attempts']}"]
    if not rows:
        lines.append("fields       : none ingested yet")
    for r in rows:
        if r["index_built_at"] is None:
            age = "not built"
        else:
            age = f"{r['index_age_seconds']:,.0f}s old" + (" STALE" if r["index_stale"] else "")
        lines.append(f"  {r['field']:<20} {r['documents']:>5} docs {r['chunks']:>6} chunks "
                     f"{r['npt_events']:>5} events   index: {age}")
    _emit(payload, args.json, "\n".join(lines))
    store.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    ws, store, searcher, st = _wire(args)
    narrator, exit_code = _build_narrator("ask", args.narrator, ws, store)
    if narrator is None:
        store.close()
        return exit_code if exit_code is not None else 2
    answer = ask_qa(args.question, store, searcher, narrator=narrator,
                    top_k=st.retrieval.top_k, spread_rate=st.spread_rate_usd_per_day,
                    keywords=st.taxonomy.keywords)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(answer.text)
        print()
        print(f"[query plan] {answer.query_plan['summary']}")
        if not answer.abstained:
            print(f"[sources]    {', '.join(dict.fromkeys(c.doc_id for c in answer.citations))}")
        if answer.figure_sources:
            print(f"[figures]    computed from {len(answer.figure_sources)} reports "
                  "(all listed under figure_sources in --json)")
        for w in answer.citation_warnings:
            print(f"[warning]    {w}", file=sys.stderr)
        if answer.narrator_rejected:
            for reason in answer.narrator_rejected["reasons"]:
                print(f"[warning]    narrator output rejected: {reason}", file=sys.stderr)

    audit.record_ask(ws.audit_path, store, args.question, answer)
    store.close()
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    ws, store, _, st = _wire(args)
    narrator, exit_code = _build_narrator("brief", args.narrator, ws, store)
    if narrator is None:
        store.close()
        return exit_code if exit_code is not None else 2
    risk_thresholds = asdict(st.risk)
    find_patterns_thresholds = {k: v for k, v in risk_thresholds.items() if k != "max_risks"}
    verification_placeholder: dict[str, Any] = {}
    index_hash = workspace_mod.index_files_hash(ws, args.field)
    provenance = riskbrief.build_provenance(store, args.field, st.spread_rate_usd_per_day, narrator.name,
                                            verification_placeholder, index_manifest_hash=index_hash,
                                            risk_thresholds=risk_thresholds)
    try:
        brief = riskbrief.build_brief(
            store, args.well, args.field, args.td,
            spread_rate=st.spread_rate_usd_per_day, narrator=narrator, max_risks=st.risk.max_risks,
            plan_rig=args.rig, plan_mwd=args.mwd, provenance=provenance,
            risk_thresholds=find_patterns_thresholds, keywords=st.taxonomy.keywords,
        )
    except riskbrief.MissingDdrDataError as exc:
        print(f"wellbrief brief: {exc}", file=sys.stderr)
        store.close()
        return 3
    check = riskbrief.verify_brief(brief, store)
    verification_placeholder.update(check)   # the provenance block carries the same dict

    audit.record_brief(ws.audit_path, brief, check, parameters={
        "field": args.field, "well": args.well, "td_m": args.td,
        "rig": args.rig, "mwd": args.mwd,
        "spread_rate_usd_per_day": st.spread_rate_usd_per_day,
        "format": args.format,
    })

    for p in check["problems"][:5]:
        print(f"[warning] {p}", file=sys.stderr)
    if brief.narrator_rejected:
        for reason in brief.narrator_rejected["reasons"]:
            print(f"[warning] narrator output rejected: {reason}", file=sys.stderr)

    label = "" if check["ok"] else "NOT VERIFIED: "
    if args.format == "json":
        out = json.dumps(riskbrief.brief_json(brief, check), indent=2, ensure_ascii=False)
    elif args.format == "md":
        out = riskbrief.render_markdown(brief, check)
    else:
        lines = [f"{label}{brief.narrative}" if label else brief.narrative, ""]
        lines.append(f"[verification] {check['citations_checked']} citations checked, "
                    f"{'all verbatim' if check['ok'] else str(len(check['problems'])) + ' problems'}")
        lines.append("")
        lines.append(riskbrief.DISCLAIMER)
        out = "\n".join(lines)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out if out.endswith("\n") else out + "\n", encoding="utf-8")
        print(f"brief written to {args.out}")
    else:
        print(out)
    store.close()
    return 0 if check["ok"] else 2


def cmd_npt(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    st = _settings(ws, args)
    payload = analytics.npt_payload(store, st.spread_rate_usd_per_day, field_name=args.field,
                                    well=args.well, code=args.code, since=args.since)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        header = f"NPT since {args.since}: " if args.since else ""
        print(f"{header}{payload['event_count']} events, {payload['well_count']} wells, "
              f"{payload['total_hours']:,.1f} h NPT, ${payload['total_cost_usd']:,.0f}")
        print(f"avoidable: {payload['avoidable_hours']:,.1f} h "
              f"({payload['avoidable_share'] * 100:.0f} %), ${payload['avoidable_cost_usd']:,.0f}")
        print()
        print(f"{'code':<26}{'hours':>10}{'events':>9}{'cost':>14}")
        for row in payload["by_code"]:
            print(f"{row['code']:<26}{row['hours']:>10,.1f}{row['events']:>9}${row['cost_usd']:>13,.0f}")
    store.close()
    return 0


def cmd_patterns(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    st = _settings(ws, args)
    thresholds = {k: v for k, v in asdict(st.risk).items() if k != "max_risks"}
    patterns = analytics.find_patterns(store, args.field, **thresholds)
    if args.json:
        print(json.dumps([p.to_dict() for p in patterns], indent=2, ensure_ascii=False))
    else:
        if not patterns:
            print(f"No repeating pattern clears the threshold on {args.field}.")
        for p in patterns:
            where = f"{p.hole_section} / {p.formation}" if p.scope == "interval" else (p.rig or p.mwd)
            metric = f"lift {p.lift:.2f}" if p.lift is not None else f"ratio {p.ratio:.2f}"
            print(f"{p.code} [{p.scope}] {where}")
            print(f"  {p.wells_affected}/{p.wells_total} wells ({p.probability * 100:.0f} %), "
                  f"{p.hours_total:,.1f} h total, {metric}, "
                  f"mean {p.mean_hours_per_affected_well:.1f} h per affected well")
            if p.driver:
                print(f"  driver: {p.driver}")
            print()
    store.close()
    return 0


def cmd_corpus_generate(args: argparse.Namespace) -> int:
    try:
        generated = corpus.build_corpus(seed=args.seed, scale=args.scale)
        written = corpus.write_corpus(generated, args.out, args.formats, ledger_csv=args.ledger_csv)
    except (ValueError, corpus.OutputDirError) as exc:
        print(f"wellbrief corpus generate: {exc}", file=sys.stderr)
        return 2
    by_type = Counter(d.doc_type for d in generated.documents)
    events: Counter[str] = Counter()
    for w in generated.wells:
        events[w.spec.name] += sum(len(d.entries) for d in w.days)
    payload = {
        "out": str(args.out),
        "seed": args.seed,
        "scale": args.scale,
        "formats": args.formats,
        "ledger_csv": args.ledger_csv,
        "wells": len(generated.wells),
        "documents": len(generated.documents),
        "documents_by_type": dict(sorted(by_type.items())),
        "npt_events": sum(events.values()),
        "npt_events_by_field": dict(sorted(events.items())),
        "files_written": len(written),
        "manifest_hash": corpus.manifest_hash(args.out, written),
    }
    lines = [
        f"Generated the synthetic corpus in {args.out} (seed {args.seed}, scale {args.scale}, "
        f"format {args.formats}{', with an NPT ledger CSV' if args.ledger_csv else ''}).",
        f"  wells        : {payload['wells']}",
        f"  documents    : {payload['documents']:,} ("
        + ", ".join(f"{t} {n:,}" for t, n in payload["documents_by_type"].items()) + ")",
        f"  NPT events   : {payload['npt_events']:,} ("
        + ", ".join(f"{f} {n:,}" for f, n in payload["npt_events_by_field"].items()) + ")",
        f"  files        : {len(written):,} (documents plus the ground-truth sidecar"
        + (" and the NPT ledger CSV)" if args.ledger_csv else ")"),
        f"  manifest     : sha256:{payload['manifest_hash']}",
    ]
    _emit(payload, args.json, "\n".join(lines))
    return 0


def _seeds(raw: str) -> list[int]:
    try:
        seeds = [int(s) for s in raw.split(",") if s.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"seeds must be comma-separated integers, not {raw!r}") from None
    if not seeds:
        raise argparse.ArgumentTypeError("give at least one seed")
    return seeds


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals import cases, results, runner

    unsupported = [flag for flag, used in (("--ablation", args.ablation), ("--repeats", args.repeats != 1))
                  if used]
    if unsupported:
        print(f"wellbrief eval: {', '.join(unsupported)} is not supported yet", file=sys.stderr)
        return 2
    if args.narrator == "llm":
        # A fast, filesystem-free configuration check, so a missing or malformed
        # WELLBRIEF_LLM_* variable is one clean message instead of the same error
        # repeated for every case of every seed (each seed builds its own llm
        # narrator, next to its own temporary workspace, once the run actually starts).
        try:
            narrate_llm.narrator_from_env(os.environ, Path(os.devnull))
        except (narrate_llm.LLMConfigError, egress.InvalidBaseURL, ValueError) as exc:
            print(f"wellbrief eval: {exc}", file=sys.stderr)
            return 2
    suites = list(EVAL_SUITES) if args.suite == "all" else [args.suite]
    emit = (lambda line: None) if args.json else print
    try:
        report = runner.run(suites, args.seeds, args.argv, risk_filters=not args.no_risk_filters,
                           narrator=args.narrator, emit=emit)
    except cases.CaseError as exc:
        print(f"wellbrief eval: invalid case file: {exc}", file=sys.stderr)
        return 2
    emit(f"outbound connection attempts: {netguard.stats()['blocked']}")
    payload = report.to_dict()
    if args.out:
        results.write(payload, args.out)
        emit(f"results written to {args.out}")
    if args.json:
        print(json.dumps(results.scrub(payload), indent=2, ensure_ascii=False))
    return 0 if report.ok else 1


def cmd_selfcheck(args: argparse.Namespace) -> int:
    result = selfcheck_mod.run()
    print(f"wellbrief selfcheck (seed {corpus.SEED})")
    for step in result.steps:
        print(f"  {step.name:<16} {step.seconds:7.3f}s")
    print(f"outbound connection attempts: {result.outbound_connection_attempts}")
    if result.guard_error is not None:
        print(f"guard self-test: error: {result.guard_error}", file=sys.stderr)
    elif result.guard_probe is not None:
        print(f"guard self-test: blocked {result.guard_probe['blocked']}/{result.guard_probe['attempted']}")
    for failure in result.failures:
        print(f"[failure] {failure}", file=sys.stderr)
    print("selfcheck: PASS" if result.ok else "selfcheck: FAIL")
    return 0 if result.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    narrator, exit_code = _build_narrator("serve", args.narrator, ws, store)
    store.close()
    if narrator is None:
        return exit_code if exit_code is not None else 2
    try:
        httpd = server_mod.create_server(ws, narrator, args.host, args.port)
    except OSError as exc:
        print(f"wellbrief serve: {exc}", file=sys.stderr)
        return 2
    print(f"wellbrief serve: workspace '{ws.name}', narrator '{narrator.name}', "
          f"listening on http://{httpd.server_name}:{httpd.server_port}/")
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wellbrief",
        description="Offline retrieval and NPT analytics over drilling well files: cited "
                    "answers and pre-spud offset-well risk briefs, with no network access.",
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--workspace", default=None,
                   help="workspace name inside $WELLBRIEF_HOME (default: $WELLBRIEF_WORKSPACE or 'default')")
    p.add_argument("--narrator", choices=NARRATORS, default=_default_narrator(),
                   help="narrator that phrases `ask` and `brief` answers, and (for `eval`) that "
                        "`eval`'s own ask/brief cases use (default: $WELLBRIEF_NARRATOR, else "
                        "'offline'; 'llm' calls an OpenAI-compatible server configured through "
                        "WELLBRIEF_LLM_* environment variables, behind an egress guard and a "
                        "budget; a call it cannot complete or verify falls back to 'offline')")
    p.add_argument("--spread-rate", type=float, default=None,
                   help="all-in spread rate in USD per day, used for every cost figure "
                        "(default: $WELLBRIEF_SPREAD_RATE_USD_PER_DAY, else 48000)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("ingest", help="load a directory of well files into the workspace")
    sp.add_argument("path", type=Path, help="folder of .txt/.md/.pdf/.docx/.csv well files")
    sp.add_argument("--field", help="field for a document with no field detected in its own text "
                                    "(default: 'unassigned')")
    sp.add_argument("--config", type=Path,
                    help="wellbrief.toml with this ingest's parse settings (default: the ingested "
                         "folder's own wellbrief.toml, if it has one)")
    sp.add_argument("--prune", action="store_true",
                    help="remove documents whose files are no longer in the folder")
    sp.add_argument("--dry-run", action="store_true",
                    help="print the coverage table without writing anything")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("index", help="rebuild the per-field retrieval indexes")
    sp.add_argument("--field", help="rebuild only this field (default: every field in the workspace)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("status", help="what is loaded, and how fresh its indexes are")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("ask", help="ask a question of the archive")
    sp.add_argument("question", help="the question, in plain English, e.g. "
                                     "'what happened in the 12 1/4 inch section on Orrindale'")
    sp.add_argument("--top-k", type=int, default=None,
                    help="results to retrieve (default: [retrieval] top_k in wellbrief.toml, else 8)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("brief", help="build an offset-well risk brief for a planned well")
    sp.add_argument("--well", required=True, help="name of the planned well, e.g. ORD-NEXT")
    sp.add_argument("--field", required=True, help="field whose offset wells are used, e.g. Orrindale")
    sp.add_argument("--td", type=float, required=True, help="planned total depth in metres")
    sp.add_argument("--rig", help="planned rig; marks rig-keyed risks as applies or does-not-apply")
    sp.add_argument("--mwd", help="planned MWD tool; marks tool-keyed risks as applies or does-not-apply")
    sp.add_argument("--max-risks", type=int, default=None,
                    help="risks listed, ranked by expected cost (default: [risk] max_risks in "
                         "wellbrief.toml, else 8)")
    sp.add_argument("--format", choices=["text", "md", "json"], default="text",
                    help="output rendering (default: text)")
    sp.add_argument("--out", type=Path, help="write the brief to this file instead of stdout")
    sp.set_defaults(func=cmd_brief)

    sp = sub.add_parser("npt", help="non-productive time rollup, with cost")
    sp.add_argument("--field", help="only this field")
    sp.add_argument("--well", help="only this well")
    sp.add_argument("--code", help="only this NPT code, e.g. STUCK_PIPE")
    sp.add_argument("--since", help="only events on or after this date (ISO YYYY-MM-DD)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_npt)

    sp = sub.add_parser("patterns", help="repeating problems in a field")
    sp.add_argument("--field", required=True, help="field whose repeating problems to list, e.g. Orrindale")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_patterns)

    sp = sub.add_parser("eval", help="run the evaluation suites on freshly generated corpora")
    sp.add_argument("--suite", choices=[*EVAL_SUITES, "all"], default="all",
                    help="which case suite to run (default: all); 'verifier-faults' measures the "
                         "narrator verifier's catch rate by injecting a hallucination into a "
                         "real, grounded answer and checking that it is rejected")
    sp.add_argument("--seeds", type=_seeds, default=[SEED],
                    help=f"comma-separated corpus seeds (default {SEED})")
    sp.add_argument("--out", type=Path, help="also write the full results as JSON to this file")
    sp.add_argument("--no-risk-filters", action="store_true",
                    help="brief precision cases: build the briefs without the risk filters (min_lift 0, "
                         "unavoidable codes included) and report them without a score; "
                         "other briefs keep them")
    sp.add_argument("--ablation", action="store_true", help="retrieval ablation (not supported yet)")
    sp.add_argument("--repeats", type=int, default=1, help="repeats per case (not supported yet)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("corpus", help="synthetic well-file corpus")
    corpus_sub = sp.add_subparsers(dest="corpus_command", required=True)
    gp = corpus_sub.add_parser("generate",
                               help="write the synthetic corpus and its ground truth to a directory")
    gp.add_argument("--out", type=Path, required=True, help="output directory (empty, or a previous corpus)")
    gp.add_argument("--seed", type=int, default=SEED, help=f"random seed (default {SEED})")
    gp.add_argument("--scale", type=int, default=1,
                    help=f"multiply the number of wells per field, 1 to {MAX_SCALE} "
                         "(default 1: 28 + 14 wells)")
    gp.add_argument("--formats", choices=sorted(corpus.FORMATS), default="txt",
                    help="txt (default) or pdf or docx for every document, or mixed: daily reports "
                         "as .txt, end-of-well reports as .pdf, incident reports as .docx")
    gp.add_argument("--ledger-csv", action="store_true",
                    help="also write npt-ledger.csv with every DDR NPT row")
    gp.add_argument("--json", action="store_true", help="machine-readable output")
    gp.set_defaults(func=cmd_corpus_generate)

    sp = sub.add_parser("selfcheck", help="offline end-to-end check plus a network-guard positive control")
    sp.set_defaults(func=cmd_selfcheck)

    sp = sub.add_parser("serve", help="run the local JSON API and web UI")
    sp.add_argument("--host", default=server_mod.DEFAULT_HOST,
                    help=f"bind address (default {server_mod.DEFAULT_HOST}); binding to anything "
                         "else prints a warning, since the API has no authentication")
    sp.add_argument("--port", type=int, default=server_mod.DEFAULT_PORT,
                    help=f"bind port (default {server_mod.DEFAULT_PORT})")
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    # Installed here, never on import: a library caller who wants the guard calls
    # `wellbrief.netguard.install()` itself. Loopback and Unix sockets are always
    # allowed; the configured LLM host is allowed only while this run's narrator is `llm`.
    # A malformed WELLBRIEF_LLM_BASE_URL is a configuration problem like any other, so it
    # gets the same clean message and exit code rather than a raw traceback.
    try:
        netguard.install(allow_hosts=_netguard_allow_hosts(args.narrator))
    except ValueError as exc:
        print(f"wellbrief: WELLBRIEF_LLM_BASE_URL: {exc}", file=sys.stderr)
        return 2
    func: Callable[[argparse.Namespace], int] = args.func
    try:
        return func(args)
    except SettingsError as exc:
        print(f"wellbrief: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
