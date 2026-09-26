"""Command line interface.

Every command works on the local SQLite store and index files and needs no
network access. `--json` switches status, ask, risk, npt, patterns, digest
and eval to machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, analytics, ingest as ingest_mod, riskbrief
from .config import DB_PATH, DEFAULT_SPREAD_RATE_USD_PER_DAY, EMBED_BACKEND, LLM_BACKEND
from .embed import get_embedder
from .llm import get_narrator
from .qa import ask as ask_qa
from .search import Searcher
from .store import Store, build_indexes, load_indexes


def _wire(args) -> tuple[Store, Searcher]:
    store = Store(args.db)
    embedder = get_embedder(args.embed_backend)
    bm25, vectors = load_indexes(store)
    if vectors.backend != embedder.name:
        # The index on disk was built with a different embedder. Rebuilding is
        # cheaper than serving mismatched vectors.
        print(f"[wellbrief] index backend '{vectors.backend}' != '{embedder.name}', rebuilding",
              file=sys.stderr)
        bm25, vectors = build_indexes(store, embedder.name)
    return store, Searcher(store, bm25, vectors, embedder)


def _emit(payload, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(text)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_build(args) -> int:
    store = Store(args.db)
    stats = ingest_mod.bootstrap(store)
    counts = store.counts()
    print(f"Generated and indexed the demo field history.")
    print(f"  wells      : {counts['wells']}")
    print(f"  documents  : {counts['documents']}")
    print(f"  NPT events : {counts['npt_events']}")
    print(f"  database   : {store.db_path}")
    store.close()
    return 0


def cmd_ingest(args) -> int:
    store = Store(args.db)
    docs = ingest_mod.load_corpus_dir(args.path)
    if not docs:
        print(f"No .txt files found in {args.path}", file=sys.stderr)
        return 1
    if args.replace:
        store.reset()
    stats = ingest_mod.ingest(store, docs)
    print(f"Ingested {stats['documents']} documents and {stats['npt_events']} NPT events from {args.path}")
    store.close()
    return 0


def cmd_status(args) -> int:
    store = Store(args.db)
    counts = store.counts()
    fields = sorted({w.field_name for w in store.wells()})
    payload = {
        "database": str(store.db_path),
        "counts": counts,
        "fields": fields,
        "index_backend": store.get_kv("index_backend"),
        "indexed_documents": store.get_kv("index_docs"),
        "embed_backend": args.embed_backend,
        "llm_backend": args.llm_backend,
    }
    lines = [
        f"database        : {payload['database']}",
        f"wells           : {counts['wells']} across {len(fields)} field(s): {', '.join(fields) or '-'}",
        f"documents       : {counts['documents']}",
        f"NPT events      : {counts['npt_events']}",
        f"index backend   : {payload['index_backend']}",
        f"embed / narrate : {args.embed_backend} / {args.llm_backend}",
    ]
    _emit(payload, args.json, "\n".join(lines))
    store.close()
    return 0


def cmd_ask(args) -> int:
    store, searcher = _wire(args)
    narrator = get_narrator(args.llm_backend)
    answer = ask_qa(args.question, store, searcher, narrator=narrator,
                    top_k=args.top_k, spread_rate=args.spread_rate)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(answer.text)
        print()
        print(f"[query plan] {answer.structured['query_plan']}")
        print(f"[sources]    {', '.join(dict.fromkeys(c.doc_id for c in answer.citations))}")
        for w in answer.structured.get("citation_warnings", []):
            print(f"[warning]    {w}", file=sys.stderr)
    store.close()
    return 0


def cmd_risk(args) -> int:
    store, _ = _wire(args)
    narrator = get_narrator(args.llm_backend)
    brief = riskbrief.build_brief(
        store, args.well, args.field, args.td,
        spread_rate=args.spread_rate, narrator=narrator, max_risks=args.max_risks,
    )
    check = riskbrief.verify_brief(brief, store)
    if args.json:
        payload = brief.to_dict()
        payload["verification"] = check
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(brief.narrative)
        print()
        print(f"[verification] {check['citations_checked']} citations checked, "
              f"{'all verbatim' if check['ok'] else str(len(check['problems'])) + ' problems'}")
        for p in check["problems"][:5]:
            print(f"[warning] {p}", file=sys.stderr)
    store.close()
    return 0 if check["ok"] else 2


def cmd_npt(args) -> int:
    store = Store(args.db)
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
    store = Store(args.db)
    patterns = analytics.find_patterns(store, args.field)
    if args.json:
        print(json.dumps([p.to_dict() for p in patterns], indent=2, ensure_ascii=False))
    else:
        if not patterns:
            print(f"No repeating pattern clears the threshold on {args.field}.")
        for p in patterns:
            print(f"{p.code} / {p.hole_section} / {p.formation}")
            print(f"  {p.wells_affected}/{p.wells_total} wells ({p.probability * 100:.0f} %), "
                  f"{p.hours_total:,.1f} h total, mean {p.mean_hours_per_affected_well:.1f} h per affected well")
            if p.driver:
                print(f"  driver: {p.driver}")
            print()
    store.close()
    return 0


def cmd_digest(args) -> int:
    store = Store(args.db)
    payload = analytics.digest(store, args.since, args.spread_rate)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        s = payload["summary"]
        print(f"NPT since {payload['since']}: {s['npt_hours']:,.1f} h across {s['wells']} wells, "
              f"${s['npt_cost_usd']:,.0f}, {s['avoidable_share'] * 100:.0f} % avoidable")
        for e in payload["biggest_events"]:
            print(f"  {e['date']}  {e['well']:<10}{e['code']:<24}{e['hours']:>6.1f} h  {e['description'][:70]}")
    store.close()
    return 0


def cmd_eval(args) -> int:
    from . import evaluate
    store, searcher = _wire(args)
    narrator = get_narrator(args.llm_backend)
    report = evaluate.run(store, searcher, narrator)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"narrator: {report['narrator']}")
        for r in report["results"]:
            mark = "pass" if r["passed"] else "FAIL"
            print(f"  [{mark}] {r['case']:<42}{r['detail']}")
        print()
        print(f"{report['passed']}/{report['cases']} passed ({report['pass_rate'] * 100:.0f} %)")
    store.close()
    return 0 if report["failed"] == 0 else 1


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wellbrief",
        description="Offline retrieval and NPT analytics over drilling well files.",
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--db", default=str(DB_PATH), help="path to the SQLite store")
    p.add_argument("--embed-backend", default=EMBED_BACKEND, choices=["offline"])
    p.add_argument("--llm-backend", default=LLM_BACKEND, choices=["offline"])
    p.add_argument("--spread-rate", type=float, default=DEFAULT_SPREAD_RATE_USD_PER_DAY,
                   help="all-in spread rate in USD per day, used for every cost figure")
    p.add_argument("--json", action="store_true", help="machine readable output")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="generate and index the demo field history").set_defaults(func=cmd_build)

    sp = sub.add_parser("ingest", help="load a directory of well files")
    sp.add_argument("path", type=Path)
    sp.add_argument("--replace", action="store_true", help="clear the store first")
    sp.set_defaults(func=cmd_ingest)

    sub.add_parser("status", help="what is loaded").set_defaults(func=cmd_status)

    sp = sub.add_parser("ask", help="ask a question of the archive")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=8)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("risk", help="build an offset risk register for a planned well")
    sp.add_argument("--well", required=True, help="name of the planned well, e.g. ORD-NEXT")
    sp.add_argument("--field", required=True, help="field whose offset wells are used, e.g. Orrindale")
    sp.add_argument("--td", type=float, required=True, help="planned total depth in metres")
    sp.add_argument("--max-risks", type=int, default=8)
    sp.set_defaults(func=cmd_risk)

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

    sub.add_parser("eval", help="run the accuracy harness").set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
