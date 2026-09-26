"""Question answering over the well archive.

The pipeline: plan the question, apply its structured filters, retrieve,
compute figures, quote mitigations, narrate. The narrator only phrases what
the other stages produced.

- A question that names a well, field, hole section, formation or NPT code
  the workspace does not hold, or whose filters match no document and no NPT
  entry, is not answered: the text is exactly `NO_MATCH` plus one line naming
  what did not match, with no figures, no citations and no mitigations.
- Figures are aggregated with SQL over the NPT ledger. The text cites at most
  `FIGURE_SOURCE_LIMIT` reports behind them, the largest contributors first;
  `Answer.figure_sources` lists every contributing report with its hours.
  When the question's documents exist but no ledger entry matches (a well
  with reports and no NPT), the figures are zeros and the text says that no
  NPT is recorded.
- Every document the narrator may cite is put into the evidence pack with a
  verbatim quote: the retrieval hits, the cited figure sources, and the end of
  well reports the mitigations are quoted from. Any document id the narrator
  cites that is not in the pack is reported as a citation warning.
"""

from __future__ import annotations

from typing import Any

from .config import (
    AVOIDABLE_CODES,
    DEFAULT_SPREAD_RATE_USD_PER_DAY,
    DEFAULT_TOP_K,
    FIGURE_SOURCE_LIMIT,
    NPT_CODES,
    hours_to_usd,
)
from .llm import NO_MATCH, Narrator, OfflineNarrator, extract_cited_ids
from .models import Answer, Citation, Document, SearchHit
from .parse import parse_eowr
from .search import QueryPlan, Searcher, plan_query
from .store import Store
from .text import snippet, tokenize

__all__ = ["FIGURE_SOURCE_LIMIT", "NO_MATCH", "abstention_text", "ask", "ledger_figures", "verify_citations"]


def _is_verbatim(quote: str, text: str) -> bool:
    """The same whitespace-insensitive test the grounding checks apply."""
    return " ".join(quote.split()) in " ".join(text.split())


def _pack_entry(doc: Document, quote: str, role: str, score: float = 0.0) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id,
        "doc_type": doc.doc_type,
        "well": doc.well,
        "field": doc.field_name,
        "date": doc.date,
        "quote": quote,
        "score": score,
        "role": role,
    }


def _evidence_pack(hits: list[SearchHit], question: str) -> list[dict[str, Any]]:
    terms = tokenize(question)
    return [
        _pack_entry(hit.document, hit.snippet or snippet(hit.document.text, terms), "retrieval", hit.score)
        for hit in hits
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figure_filters(plan: QueryPlan) -> dict[str, Any]:
    """The ledger filters behind the figures, as they appear in the JSON output."""
    band = plan.depth_band
    return {
        "fields": list(plan.fields),
        "wells": list(plan.wells),
        "codes": list(plan.codes),
        "hole_sections": list(plan.hole_sections),
        "formations": list(plan.formations),
        "depth_band_m": list(band) if plan.depth_exact and band is not None else None,
    }


def ledger_figures(
    store: Store, plan: QueryPlan, spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Totals and breakdowns over the ledger rows the plan selects, aggregated in SQL.

    Returns the figures and every contributing report with its hours, largest
    first. When no row matches, the figures are zeros (the question's scope
    has no NPT recorded) and there are no sources. A breakdown is left out
    when the plan pins that dimension to one value (one code, one field, one
    well, one section). Cost is hours / 24 x spread rate, from the unrounded
    sum.
    """
    filters = plan.ledger_filters()
    summary = store.npt_summary(AVOIDABLE_CODES, **filters)
    total = float(summary["hours"])
    avoidable = float(summary["avoidable_hours"])
    figures: dict[str, Any] = {
        "filters": _figure_filters(plan),
        "spread_rate": spread_rate,
        "event_count": summary["events"],
        "well_count": summary["wells"],
        "report_count": summary["reports"],
        "total_hours": round(total, 1),
        "total_cost_usd": round(hours_to_usd(total, spread_rate), 0),
        "avoidable_hours": round(avoidable, 1),
        "avoidable_share": round(avoidable / total, 3) if total else 0.0,
    }
    if len(plan.codes) != 1:
        figures["by_code"] = [
            {"code": r["key"], "label": NPT_CODES.get(r["key"], {}).get("label", r["key"]),
             "avoidable": r["key"] in AVOIDABLE_CODES, "hours": round(r["hours"], 1),
             "cost_usd": round(hours_to_usd(r["hours"], spread_rate), 0), "events": r["events"],
             "wells": r["wells"]}
            for r in store.npt_breakdown("code", **filters)
        ]
    if len(plan.fields) != 1:
        figures["by_field"] = [{"field": r["key"], "hours": round(r["hours"], 1), "events": r["events"],
                                "wells": r["wells"]}
                               for r in store.npt_breakdown("field_name", **filters)]
    if len(plan.wells) != 1:
        figures["by_well"] = [{"well": r["key"], "hours": round(r["hours"], 1), "events": r["events"]}
                              for r in store.npt_breakdown("well", **filters)]
    if len(plan.hole_sections) != 1:
        figures["by_section"] = [{"section": r["key"] or "unknown", "hours": round(r["hours"], 1),
                                  "events": r["events"]}
                                 for r in store.npt_breakdown("hole_section", **filters)]
    sources = [
        {"doc_id": r["key"], "well": r["well"], "date": r["date"], "hours": round(r["hours"], 1),
         "events": r["events"], "codes": sorted(str(r["codes"]).split(","))}
        for r in store.npt_breakdown("doc_id", **filters)
    ]
    return figures, sources


def _figure_source_pack(store: Store, plan: QueryPlan, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack entries for the cited figure sources, each quoted by the description of its largest entry."""
    out: list[dict[str, Any]] = []
    filters = plan.ledger_filters()
    for source in sources:
        doc = store.get_document(source["doc_id"])
        if doc is None:
            continue
        events = store.npt(**filters, doc_id=doc.doc_id)
        description = max(events, key=lambda e: e.hours).description if events else ""
        if description and _is_verbatim(description, doc.text):
            quote = description
        else:
            quote = snippet(doc.text, tokenize(description))
        out.append(_pack_entry(doc, quote, "figure_source"))
    return out


# ---------------------------------------------------------------------------
# Mitigations and the evidence pack
# ---------------------------------------------------------------------------


def _mitigation_sources(store: Store, mitigations: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The end of well reports the mitigations are quoted from, one entry per sentence."""
    out: list[dict[str, Any]] = []
    for m in mitigations:
        doc = store.get_document(m["doc_id"])
        if doc is not None:
            out.append(_pack_entry(doc, m["text"], "mitigation"))
    return out


def _merge(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate pack entries, keeping the first of any repeated (document, quote) pair."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for group in groups:
        for entry in group:
            key = (entry["doc_id"], " ".join(entry["quote"].split()))
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
    return out


def _mitigations_for(store: Store, plan: QueryPlan, limit: int = 4) -> list[dict[str, str]]:
    """Lessons written in end of well reports that match the question's scope."""
    if not plan.scoped:
        return []
    needles = [n.lower() for n in [*plan.formations, *plan.hole_sections]]
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in store.documents(doc_type="eowr", field_name=plan.fields or None):
        parsed = parse_eowr(doc.text)
        for sentence in list(parsed.get("recommendations", [])) + list(parsed.get("lessons", [])):
            low = sentence.lower()
            if needles and not any(n in low for n in needles):
                continue
            # A mitigation is shown as a quote, so it has to be one. parse_eowr
            # normalises the text (NFKC, unicode fractions, dashes, apostrophes),
            # so a sentence that contained one of those characters is not in the
            # raw report word for word and is skipped here.
            if not _is_verbatim(sentence, doc.text):
                continue
            key = low[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": sentence, "doc_id": doc.doc_id, "well": doc.well})
            if len(out) >= limit:
                return out
    return out


def verify_citations(text: str, pack: list[dict[str, Any]], store: Store) -> list[str]:
    """Return any document id the narrator cited that was not in the pack."""
    allowed = {e["doc_id"] for e in pack}
    warnings: list[str] = []
    for cited in sorted(extract_cited_ids(text)):
        if cited in allowed:
            continue
        if store.get_document(cited) is None:
            warnings.append(f"cited document does not exist: {cited}")
        else:
            warnings.append(f"cited document was not in the evidence pack: {cited}")
    return warnings


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def abstention_text(unmatched: dict[str, Any], plan: QueryPlan) -> str:
    """`NO_MATCH` and one line naming what did not match."""
    if "filters" in unmatched:
        line = f"Unmatched: no document or NPT event matches {plan.filters_text()}"
    else:
        line = "Unmatched: " + "; ".join(f"{kind} {', '.join(values)}" for kind, values in unmatched.items())
    return f"{NO_MATCH}\n{line}"


def _unmatched_entities(plan: QueryPlan) -> dict[str, Any]:
    out: dict[str, list[str]] = {}
    for u in plan.unmatched:
        out.setdefault(u.kind, []).append(u.value)
    return dict(out)


def _abstain(question: str, plan: QueryPlan, unmatched: dict[str, Any], narrator: Narrator) -> Answer:
    return Answer(
        question=question,
        text=abstention_text(unmatched, plan),
        query_plan=plan.to_dict(),
        abstained=True,
        unmatched=unmatched,
        narrator=narrator.name,
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def ask(
    question: str,
    store: Store,
    searcher: Searcher,
    narrator: Narrator | None = None,
    top_k: int = DEFAULT_TOP_K,
    spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY,
) -> Answer:
    narrator = narrator or OfflineNarrator()
    plan = plan_query(question, store)
    if plan.unmatched:
        return _abstain(question, plan, _unmatched_entities(plan), narrator)

    figures, sources = ledger_figures(store, plan, spread_rate) if plan.countable else (None, [])
    hits, _ = searcher.search(question, top_k=top_k, plan=plan)
    if not hits and not sources:
        applied = {k: v for k, v in (_figure_filters(plan) | {"doc_types": plan.doc_types}).items() if v}
        return _abstain(question, plan, {"filters": applied}, narrator)

    cited_sources = sources[:FIGURE_SOURCE_LIMIT]
    mitigations = _mitigations_for(store, plan)
    pack = _merge(_evidence_pack(hits, question), _figure_source_pack(store, plan, cited_sources),
                  _mitigation_sources(store, mitigations))

    summary = {
        "figures": figures,
        "figure_sources": [{"doc_id": s["doc_id"], "hours": s["hours"]} for s in cited_sources],
        "report_count": len(sources),
        "mitigations": mitigations,
        "plan": plan.describe(),
        "scope": plan.filters_text(types=False),
    }
    text = narrator.answer(question, pack, summary)
    citations = [
        Citation(doc_id=e["doc_id"], doc_type=e["doc_type"], well=e["well"], date=e["date"], quote=e["quote"])
        for e in pack
    ]
    return Answer(
        question=question,
        text=text,
        citations=citations,
        hits=hits,
        query_plan=plan.to_dict(),
        figures=figures,
        figure_sources=sources,
        mitigations=mitigations,
        citation_warnings=verify_citations(text, pack, store),
        narrator=narrator.name,
    )
