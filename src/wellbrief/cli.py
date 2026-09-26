"""Command line interface.

Every command needs no network access. `corpus generate` writes the
synthetic corpus and its ground truth to a directory; `ingest` loads a folder
of well files into a workspace's store; every other command works on that
workspace's store and its per-field index files (see `workspace.py`).
`--json` switches status, ask, npt, patterns, digest, eval, corpus generate
and index to machine-readable output; `brief` has its own
`--format text|md|json` instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, analytics, corpus, ingest as ingest_mod, riskbrief, workspace as workspace_mod
from .config import DEFAULT_SPREAD_RATE_USD_PER_DAY, EMBED_BACKEND, LLM_BACKEND, RISK_MAX_RISKS
from .corpus import MAX_SCALE, SEED
from .embed import get_embedder
from .evals.cases import SUITES as EVAL_SUITES
from .llm import get_narrator
from .qa import ask as ask_qa
from .search import Searcher
from .store import Store
from .workspace import FieldIndex, Workspace


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace.resolve(getattr(args, "workspace", None))


def _wire(args) -> tuple[Workspace, Store, Searcher]:
    ws = _workspace(args)
    store = ws.open_store()
    indexes = workspace_mod.ensure_field_indexes(ws, store, args.embed_backend)
    return ws, store, Searcher(store, indexes, get_embedder(args.embed_backend))


def _index_age_seconds(built_at: str) -> float:
    built = datetime.strptime(built_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (datetime.now(UTC) - built).total_seconds()


def _emit(payload, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(text)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_ingest(args) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    docs = ingest_mod.load_corpus_dir(args.path)
    if not docs:
        print(f"No .txt files found in {args.path}", file=sys.stderr)
        store.close()
        return 1
    if args.replace:
        store.reset()
    stats = ingest_mod.ingest(store, docs)
    print(f"Ingested {stats['documents']} documents, {stats['chunks']} chunks and "
          f"{stats['npt_events']} NPT events from {args.path} into workspace '{ws.name}'")
    workspace_mod.check_field_slugs(store.field_names())
    touched = sorted({d.field_name for d in docs})
    built = [workspace_mod.rebuild_field_index(ws, store, f, args.embed_backend) for f in touched]
    if built:
        print("Indexed: " + "; ".join(
            f"{i.field} ({i.manifest.chunks} chunks, {i.manifest.build_seconds:.3f}s)" for i in built))
    store.close()
    return 0


def cmd_index(args) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    known = store.field_names()
    workspace_mod.check_field_slugs(known)
    if args.field and args.field not in known:
        print(f"wellbrief index: unknown field {args.field!r} (known: {', '.join(known) or '-'})",
              file=sys.stderr)
        store.close()
        return 2
    fields = [args.field] if args.field else known
    built: list[FieldIndex] = [workspace_mod.rebuild_field_index(ws, store, f, args.embed_backend)
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


def cmd_status(args) -> int:
    ws = _workspace(args)
    store = ws.open_store()
    fields = store.field_names()
    workspace_mod.check_field_slugs(fields)
    rows = []
    for f in fields:
        counts = store.field_counts(f)
        current_hash = store.corpus_hash(f)
        built = workspace_mod.load_field_index(ws, f)
        manifest = built.manifest if built else None
        rows.append({
            "field": f,
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "npt_events": counts["npt_events"],
            "corpus_hash": current_hash,
            "embedder": manifest.embedder if manifest else None,
            "index_built_at": manifest.built_at if manifest else None,
            "index_age_seconds": round(_index_age_seconds(manifest.built_at), 1) if manifest else None,
            "index_stale": manifest is None or manifest.corpus_hash != current_hash,
        })
    payload = {
        "workspace": ws.name,
        "home": str(ws.home),
        "database": str(ws.db_path),
        "network_mode": "offline",
        "fields": rows,
    }
    lines = [f"workspace    : {ws.name} ({ws.root})", "network mode : offline"]
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


def cmd_ask(args) -> int:
    _, store, searcher = _wire(args)
    narrator = get_narrator(args.llm_backend)
    answer = ask_qa(args.question, store, searcher, narrator=narrator,
                    top_k=args.top_k, spread_rate=args.spread_rate)
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
    store.close()
    return 0


def cmd_brief(args) -> int:
    ws, store, _ = _wire(args)
    narrator = get_narrator(args.llm_backend)
    verification_placeholder: dict[str, Any] = {}
    index_hash = workspace_mod.index_files_hash(ws, args.field)
    provenance = riskbrief.build_provenance(store, args.field, args.spread_rate, narrator.name,
                                            verification_placeholder, index_manifest_hash=index_hash)
    brief = riskbrief.build_brief(
        store, args.well, args.field, args.td,
        spread_rate=args.spread_rate, narrator=narrator, max_risks=args.max_risks,
        plan_rig=args.rig, plan_mwd=args.mwd, provenance=provenance,
    )
    check = riskbrief.verify_brief(brief, store)
    verification_placeholder.update(check)   # the provenance block carries the same dict

    for p in check["problems"][:5]:
        print(f"[warning] {p}", file=sys.stderr)

    label = "" if check["ok"] else "NOT VERIFIED: "
    if args.format == "json":
        payload = brief.to_dict()
        payload["verification"] = check
        payload["disclaimer"] = riskbrief.DISCLAIMER
        payload["not_verified"] = not check["ok"]
        out = json.dumps(payload, indent=2, ensure_ascii=False)
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


def cmd_npt(args) -> int:
    store = _workspace(args).open_store()
    filters = {}
    if args.field:
        filters["field_name"] = args.field
    if args.code:
        filters["code"] = args.code
    if args.well:
        filters["well"] = args.well
    roll = analytics.rollup(store.npt(**filters), args.spread_rate)
    if args.json:
        print(json.dumps(roll, indent=2, ensure_ascii=False))
    else:
        print(f"{roll['event_count']} events, {roll['well_count']} wells, "
              f"{roll['total_hours']:,.1f} h NPT, ${roll['total_cost_usd']:,.0f}")
        print(f"avoidable: {roll['avoidable_hours']:,.1f} h "
              f"({roll['avoidable_share'] * 100:.0f} %), ${roll['avoidable_cost_usd']:,.0f}")
        print()
        print(f"{'code':<26}{'hours':>10}{'events':>9}{'cost':>14}")
        for row in roll["by_code"]:
            print(f"{row['code']:<26}{row['hours']:>10,.1f}{row['events']:>9}${row['cost_usd']:>13,.0f}")
    store.close()
    return 0


def cmd_patterns(args) -> int:
    store = _workspace(args).open_store()
    patterns = analytics.find_patterns(store, args.field)
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


def cmd_digest(args) -> int:
    store = _workspace(args).open_store()
    payload = analytics.digest(store, args.since, args.spread_rate)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        s = payload["summary"]
        print(f"NPT since {payload['since']}: {s['npt_hours']:,.1f} h across {s['wells']} wells, "
              f"${s['npt_cost_usd']:,.0f}, {s['avoidable_share'] * 100:.0f} % avoidable")
        for e in payload["biggest_events"]:
            print(f"  {e['date']}  {e['well']:<10}{e['code']:<24}{e['hours']:>6.1f} h  "
                  f"{e['description'][:70]}")
    store.close()
    return 0


def cmd_corpus_generate(args) -> int:
    try:
        generated = corpus.build_corpus(seed=args.seed, scale=args.scale)
        written = corpus.write_corpus(generated, args.out, "txt")
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
        "formats": "txt",
        "wells": len(generated.wells),
        "documents": len(generated.documents),
        "documents_by_type": dict(sorted(by_type.items())),
        "npt_events": sum(events.values()),
        "npt_events_by_field": dict(sorted(events.items())),
        "files_written": len(written),
        "manifest_hash": corpus.manifest_hash(args.out, written),
    }
    lines = [
        f"Generated the synthetic corpus in {args.out} (seed {args.seed}, scale {args.scale}, format txt).",
        f"  wells        : {payload['wells']}",
        f"  documents    : {payload['documents']:,} ("
        + ", ".join(f"{t} {n:,}" for t, n in payload["documents_by_type"].items()) + ")",
        f"  NPT events   : {payload['npt_events']:,} ("
        + ", ".join(f"{f} {n:,}" for f, n in payload["npt_events_by_field"].items()) + ")",
        f"  files        : {len(written):,} (documents plus the ground-truth sidecar)",
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


def cmd_eval(args) -> int:
    from .evals import cases, results, runner

    unsupported = [flag for flag, used in (("--ablation", args.ablation), ("--repeats", args.repeats != 1),
                                           ("--narrator llm", args.narrator != "offline")) if used]
    if unsupported:
        print(f"wellbrief eval: {', '.join(unsupported)} is not supported yet", file=sys.stderr)
        return 2
    suites = list(EVAL_SUITES) if args.suite == "all" else [args.suite]
    emit = (lambda line: None) if args.json else print
    try:
        report = runner.run(suites, args.seeds, args.argv, risk_filters=not args.no_risk_filters, emit=emit)
    except cases.CaseError as exc:
        print(f"wellbrief eval: invalid case file: {exc}", file=sys.stderr)
        return 2
    payload = report.to_dict()
    if args.out:
        results.write(payload, args.out)
        emit(f"results written to {args.out}")
    if args.json:
        print(json.dumps(results.scrub(payload), indent=2, ensure_ascii=False))
    return 0 if report.ok else 1


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wellbrief",
        description="Offline retrieval and NPT analytics over drilling well files.",
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--workspace", default=None,
                   help="workspace name inside $WELLBRIEF_HOME (default: $WELLBRIEF_WORKSPACE or 'default')")
    p.add_argument("--embed-backend", default=EMBED_BACKEND, choices=["offline"])
    p.add_argument("--llm-backend", default=LLM_BACKEND, choices=["offline"])
    p.add_argument("--spread-rate", type=float, default=DEFAULT_SPREAD_RATE_USD_PER_DAY,
                   help="all-in spread rate in USD per day, used for every cost figure")
    p.add_argument("--json", action="store_true", help="machine readable output")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("ingest", help="load a directory of well files into the workspace")
    sp.add_argument("path", type=Path)
    sp.add_argument("--replace", action="store_true", help="clear the store first")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("index", help="rebuild the per-field retrieval indexes")
    sp.add_argument("--field", help="rebuild only this field (default: every field in the workspace)")
    sp.set_defaults(func=cmd_index)

    sub.add_parser("status", help="what is loaded, and how fresh its indexes are").set_defaults(
        func=cmd_status)

    sp = sub.add_parser("ask", help="ask a question of the archive")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=8)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("brief", help="build an offset-well risk brief for a planned well")
    sp.add_argument("--well", required=True, help="name of the planned well, e.g. ORD-NEXT")
    sp.add_argument("--field", required=True, help="field whose offset wells are used, e.g. Orrindale")
    sp.add_argument("--td", type=float, required=True, help="planned total depth in metres")
    sp.add_argument("--rig", help="planned rig; marks rig-keyed risks as applies or does-not-apply")
    sp.add_argument("--mwd", help="planned MWD tool; marks tool-keyed risks as applies or does-not-apply")
    sp.add_argument("--max-risks", type=int, default=RISK_MAX_RISKS)
    sp.add_argument("--format", choices=["text", "md", "json"], default="text")
    sp.add_argument("--out", type=Path, help="write the brief to this file instead of stdout")
    sp.set_defaults(func=cmd_brief)

    sp = sub.add_parser("npt", help="non-productive time rollup")
    sp.add_argument("--field")
    sp.add_argument("--well")
    sp.add_argument("--code")
    sp.set_defaults(func=cmd_npt)

    sp = sub.add_parser("patterns", help="repeating problems in a field")
    sp.add_argument("--field", required=True)
    sp.set_defaults(func=cmd_patterns)

    sp = sub.add_parser("digest", help="summary of NPT since a date")
    sp.add_argument("--since", required=True, help="ISO date")
    sp.set_defaults(func=cmd_digest)

    sp = sub.add_parser("eval", help="run the evaluation suites on freshly generated corpora")
    sp.add_argument("--suite", choices=[*EVAL_SUITES, "all"], default="all")
    sp.add_argument("--seeds", type=_seeds, default=[SEED],
                    help=f"comma-separated corpus seeds (default {SEED})")
    sp.add_argument("--out", type=Path, help="write the results as JSON to this file")
    sp.add_argument("--no-risk-filters", action="store_true",
                    help="brief precision cases: build the briefs without the risk filters (min_lift 0, "
                         "unavoidable codes included) and report them without a score; "
                         "other briefs keep them")
    sp.add_argument("--ablation", action="store_true", help="retrieval ablation (not supported yet)")
    sp.add_argument("--narrator", choices=["offline", "llm"], default="offline",
                    help="narrator for answers and briefs (only offline is supported yet)")
    sp.add_argument("--repeats", type=int, default=1, help="repeats per case (not supported yet)")
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
    gp.set_defaults(func=cmd_corpus_generate)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
