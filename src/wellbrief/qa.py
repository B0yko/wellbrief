"""Question answering over the well archive.

The shape of an answer is fixed: computed figures first, document evidence
second, written mitigations third. The narrator only phrases what these three
stages produced. Every document the narrator may cite is put into the
evidence pack with a verbatim quote: the retrieval hits, the reports that
contribute most to the computed figures, and the end of well reports the
mitigations are quoted from. When retrieval finds nothing, the answer says
so and carries no figures, no mitigations and no citations. Any document id
the narrator cites that is not in the evidence pack is reported as a citation
warning rather than passed through silently.
"""

from __future__ import annotations

from typing import Any

from .analytics import rollup
from .config import DEFAULT_SPREAD_RATE_USD_PER_DAY, DEFAULT_TOP_K
from .llm import Narrator, OfflineNarrator, extract_cited_ids
from .models import Answer, Citation, Document, NptEvent
from .parse import parse_eowr
from .search import QueryPlan, Searcher
from .store import Store
from .text import snippet, tokenize

# Reports cited behind the computed figures, largest contributors first.
FIGURE_SOURCE_LIMIT = 5


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


def _evidence_pack(hits, question: str) -> list[dict[str, Any]]:
    terms = tokenize(question)
    return [
        _pack_entry(hit.document, hit.snippet or snippet(hit.document.text, terms), "retrieval", hit.score)
        for hit in hits
    ]


def _figure_sources(
    store: Store, events: list[NptEvent], limit: int = FIGURE_SOURCE_LIMIT,
) -> list[dict[str, Any]]:
    """The daily reports that contribute the most hours to the computed figures.

    Each is quoted by the NPT description of its largest matching entry.
    """
    by_doc: dict[str, list[NptEvent]] = {}
    for e in events:
        by_doc.setdefault(e.doc_id, []).append(e)
    ranked = sorted(by_doc.items(), key=lambda kv: (-sum(e.hours for e in kv[1]), kv[0]))
    out: list[dict[str, Any]] = []
    for doc_id, doc_events in ranked[:limit]:
        doc = store.get_document(doc_id)
        if doc is None:
            continue
        description = max(doc_events, key=lambda e: e.hours).description
        if _is_verbatim(description, doc.text):
            quote = description
        else:
            quote = snippet(doc.text, tokenize(description))
        out.append(_pack_entry(doc, quote, "figure_source"))
    return out


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


def _stats_for(store: Store, plan: QueryPlan, spread_rate: float) -> tuple[dict[str, Any], list[NptEvent]]:
    """Only compute totals when the question actually names something countable.

    Returns the figures and the ledger rows they were computed from.
    """
    filters: dict[str, Any] = {}
    if plan.field_name:
        filters["field_name"] = plan.field_name
    if plan.wells:
        filters["well"] = plan.wells
    if plan.codes:
        filters["code"] = plan.codes
    if plan.hole_section:
        filters["hole_section"] = plan.hole_section
    if plan.formation:
        filters["formation"] = plan.formation
    if not filters:
        return {}, []
    events = store.npt(**filters)
    if not events:
        return {}, []
    roll = rollup(events, spread_rate)
    stats = {
        "event_count": roll["event_count"],
        "well_count": roll["well_count"],
        "total_hours": roll["total_hours"],
        "total_cost_usd": roll["total_cost_usd"],
        "avoidable_share": roll["avoidable_share"],
        "by_code": roll["by_code"][:5],
        "worst_wells": roll["worst_wells"][:5],
        "filters_applied": {k: v for k, v in filters.items()},
    }
    return stats, events


def _mitigations_for(store: Store, plan: QueryPlan, limit: int = 4) -> list[dict[str, str]]:
    """Lessons written in end of well reports that match the question's scope."""
    if not (plan.formation or plan.hole_section or plan.codes):
        return []
    needles = [n.lower() for n in [plan.formation or "", plan.hole_section or ""] if n]
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in store.documents(doc_type="eowr", field_name=plan.field_name or None):
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
    for cited in extract_cited_ids(text):
        if cited in allowed:
            continue
        if store.get_document(cited) is None:
            warnings.append(f"cited document does not exist: {cited}")
        else:
            warnings.append(f"cited document was not in the evidence pack: {cited}")
    return warnings


def ask(
    question: str,
    store: Store,
    searcher: Searcher,
    narrator: Narrator | None = None,
    top_k: int = DEFAULT_TOP_K,
    spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY,
) -> Answer:
    narrator = narrator or OfflineNarrator()
    hits, plan = searcher.search(question, top_k=top_k)
    retrieval = _evidence_pack(hits, question)
    if retrieval:
        stats, events = _stats_for(store, plan, spread_rate)
        mitigations = _mitigations_for(store, plan)
    else:
        # Nothing was retrieved, so the narrator says nothing matched. That
        # answer carries no figures, no mitigations and no citations.
        stats, events, mitigations = {}, [], []
    figure_sources = _figure_sources(store, events)
    pack = _merge(retrieval, figure_sources, _mitigation_sources(store, mitigations))

    summary = {
        "stats": stats,
        "mitigations": mitigations,
        "plan": plan.describe(),
        "figure_sources": [e["doc_id"] for e in figure_sources],
    }
    text = narrator.answer(question, pack, summary)
    warnings = verify_citations(text, pack, store)

    citations = [
        Citation(doc_id=e["doc_id"], doc_type=e["doc_type"], well=e["well"], date=e["date"], quote=e["quote"])
        for e in pack
    ]
    return Answer(
        question=question,
        text=text,
        citations=citations,
        hits=hits,
        structured={
            "query_plan": plan.describe(),
            "stats": stats,
            "mitigations": mitigations,
            "citation_warnings": warnings,
            "narrator": narrator.name,
        },
    )
