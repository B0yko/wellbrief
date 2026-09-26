"""Question answering over the well archive.

The shape of an answer is fixed: computed figures first, document evidence
second, written mitigations third. The narrator only phrases what these three
stages produced. Any document id the narrator cites that is not in the
evidence pack is reported as a citation warning rather than passed through
silently.
"""

from __future__ import annotations

from typing import Any

from .analytics import rollup
from .config import DEFAULT_SPREAD_RATE_USD_PER_DAY, DEFAULT_TOP_K
from .llm import Narrator, OfflineNarrator, extract_cited_ids
from .models import Answer, Citation
from .parse import parse_eowr
from .search import QueryPlan, Searcher
from .store import Store
from .text import snippet, tokenize


def _evidence_pack(hits, question: str) -> list[dict[str, Any]]:
    terms = tokenize(question)
    pack: list[dict[str, Any]] = []
    for hit in hits:
        doc = hit.document
        pack.append({
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "well": doc.well,
            "field": doc.field_name,
            "date": doc.date,
            "quote": hit.snippet or snippet(doc.text, terms),
            "score": hit.score,
        })
    return pack


def _stats_for(store: Store, plan: QueryPlan, spread_rate: float) -> dict[str, Any]:
    """Only compute totals when the question actually names something countable."""
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
        return {}
    events = store.npt(**filters)
    if not events:
        return {}
    roll = rollup(events, spread_rate)
    return {
        "event_count": roll["event_count"],
        "well_count": roll["well_count"],
        "total_hours": roll["total_hours"],
        "total_cost_usd": roll["total_cost_usd"],
        "avoidable_share": roll["avoidable_share"],
        "by_code": roll["by_code"][:5],
        "worst_wells": roll["worst_wells"][:5],
        "filters_applied": {k: v for k, v in filters.items()},
    }


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
    pack = _evidence_pack(hits, question)
    stats = _stats_for(store, plan, spread_rate)
    mitigations = _mitigations_for(store, plan)

    summary = {"stats": stats, "mitigations": mitigations, "plan": plan.describe()}
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
