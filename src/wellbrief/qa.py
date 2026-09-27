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
- Mitigations come from the same miner `riskbrief.build_risk` uses
  (`miner.mine_mitigations`): a rule-based classifier keeps only sentences
  written as a practice, not a description of the failure. `_mitigations_for`
  also picks the answer's heading -- "Recorded mitigations on wells that
  avoided it" only when the miner drew them from a population it could
  compute as having avoided the named problem, "Lessons and recommendations
  from end-of-well reports" otherwise.
- Every document the narrator may cite is put into the evidence pack with a
  verbatim quote: the retrieval hits, the cited figure sources, and the
  documents the mitigations are quoted from -- a clean well's end of well
  report, or the corrective actions of a same-code incident report. Any
  document id the narrator cites that is not in the pack
  is reported as a citation warning.
- A quote is a span of the raw document text (see `quotes`). A daily report
  that the question selects through the NPT ledger (by code, hole section,
  formation, an "at" depth or the general NPT cue) is quoted by the
  description of its best selected entry; any other document by its
  best-matching content line (operations or remarks, lessons or
  recommendations, sequence of events, root cause or corrective actions),
  never by a header or label line when a content line exists, and never by
  the footer. A cited figure source is quoted by the description of an entry
  the figures count on it, and a retrieval hit that is also a cited figure
  source shows that same quote, so each report is quoted once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .config import (
    AVOIDABLE_CODES,
    DEFAULT_SPREAD_RATE_USD_PER_DAY,
    DEFAULT_TOP_K,
    FIGURE_SOURCE_LIMIT,
    NPT_CODES,
    hours_to_usd,
)
from .miner import MinerScope, exposed_clean_wells, mine_mitigations
from .models import Answer, Citation, Document, SearchHit
from .narrate import NO_MATCH, Narrator, OfflineNarrator, extract_cited_ids
from .quotes import evidence_quote, quote_page
from .search import QueryPlan, Searcher, plan_query, quote_terms
from .store import Store

__all__ = ["FIGURE_SOURCE_LIMIT", "NO_MATCH", "abstention_text", "ask", "ledger_figures", "verify_citations"]


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


def _evidence_pack(hits: list[SearchHit], terms: list[str]) -> list[dict[str, Any]]:
    """Pack entries for the retrieval hits, each quoted by its snippet (the evidence quote)."""
    return [
        _pack_entry(hit.document, hit.snippet or evidence_quote(hit.document, terms), "retrieval", hit.score)
        for hit in hits
    ]


def _quote_once(hits: list[SearchHit], figure_pack: list[dict[str, Any]]) -> list[SearchHit]:
    """The hits, where a hit that is also a cited figure source takes that source's quote,
    so every report in the answer is quoted once."""
    quotes = {e["doc_id"]: e["quote"] for e in figure_pack}
    return [replace(hit, snippet=quotes[hit.doc_id]) if hit.doc_id in quotes else hit for hit in hits]


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


def _figure_source_pack(store: Store, plan: QueryPlan, sources: list[dict[str, Any]],
                        terms: list[str]) -> list[dict[str, Any]]:
    """Pack entries for the cited figure sources, each quoted by the description of one of
    the entries the figures count on it.

    For a plan that chooses its reports through the ledger these are the
    entries `search.quoted_entries` selects, so the quote is the one a
    retrieval hit on the same report shows; for a plan that names only
    fields or wells they are all the report's entries.
    """
    out: list[dict[str, Any]] = []
    filters = plan.ledger_filters()
    for source in sources:
        doc = store.get_document(source["doc_id"])
        if doc is None:
            continue
        entries = store.npt(**filters, doc_id=doc.doc_id)
        out.append(_pack_entry(doc, evidence_quote(doc, terms, entries, plan.depth_m), "figure_source"))
    return out


# ---------------------------------------------------------------------------
# Mitigations and the evidence pack
# ---------------------------------------------------------------------------


def _mitigation_sources(store: Store, mitigations: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The documents the mitigations are quoted from, one entry per sentence: a
    clean well's end of well report, or the corrective actions of a same-code
    incident report."""
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


#: Heading `ask` shows above its mitigations when the miner drew them from a
#: population it knows avoided the named problem.
MITIGATIONS_HEADING_AVOIDED = "Recorded mitigations on wells that avoided it"
#: Heading otherwise: the question named no code, or no clean population is
#: known, so these are relevant lessons and recommendations, not a claim that
#: any of them come from a well that avoided anything.
MITIGATIONS_HEADING_GENERAL = "Lessons and recommendations from end-of-well reports"


def _mitigations_for(store: Store, plan: QueryPlan, limit: int = 3,
                     keywords: Mapping[str, list[str]] | None = None,
                     ) -> tuple[list[dict[str, str]], str]:
    """Mitigations relevant to the question's scope, through the same miner
    `riskbrief.build_risk` uses, and the heading the answer
    should show above them. `limit` defaults to 3, the same cap
    `mine_mitigations` itself applies, so `ask` and `brief` show the same
    "at most 3" everywhere; a caller only needs to pass it to ask for fewer.

    When the plan names exactly one code and a hole section or formation,
    the wells that avoided it there can be computed directly from the ledger
    (`miner.exposed_clean_wells`), the same rule a discovered risk's
    `clean_wells` already applies; if that population exists and yields a
    mitigation, the answer may say so. Otherwise (no code named, or no known
    clean well) the miner runs unrestricted -- any relevant end of well
    report counts, not only ones known to have avoided the problem -- and the
    weaker heading applies.

    A question that names no code, or more than one, leaves `code` empty:
    the miner's relevance test is keyed on one code's configured keywords
    (`miner._relevant`), so an empty code matches nothing and `ask` shows no
    mitigations at all rather than guessing which of several codes they
    belong to. This is a real scope limit of a single shared miner, not a
    bug: `riskbrief.build_risk` never has this problem because a discovered
    `Pattern` always carries exactly one code.
    """
    if not plan.scoped:
        return [], MITIGATIONS_HEADING_GENERAL
    fields: list[str] | None = list(plan.fields) or None
    single_field = plan.fields[0] if len(plan.fields) == 1 else None
    code = plan.codes[0] if len(plan.codes) == 1 else ""
    hole_section = plan.hole_sections[0] if len(plan.hole_sections) == 1 else ""
    formation = plan.formations[0] if len(plan.formations) == 1 else ""

    if code and (hole_section or formation):
        clean = exposed_clean_wells(store, single_field, code, hole_section, formation)
        if clean:
            scope = MinerScope(code=code, field_name=fields, hole_section=hole_section,
                               formation=formation, clean_wells=frozenset(clean))
            found = mine_mitigations(store, scope, limit, keywords=keywords)
            if found:
                return [m.to_dict() for m in found], MITIGATIONS_HEADING_AVOIDED

    broad = MinerScope(code=code, field_name=fields, hole_section=hole_section, formation=formation)
    mitigations = mine_mitigations(store, broad, limit, keywords=keywords)
    return [m.to_dict() for m in mitigations], MITIGATIONS_HEADING_GENERAL


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
    keywords: Mapping[str, list[str]] | None = None,
) -> Answer:
    """`keywords` is the mitigation miner's own configuration (see `miner.mine_mitigations`); a
    resolved `settings.Settings`' own `taxonomy.keywords`, default the canonical template's own."""
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
    mitigations, mitigations_heading = _mitigations_for(store, plan, keywords=keywords)
    terms = quote_terms(question, plan)
    figure_pack = _figure_source_pack(store, plan, cited_sources, terms)
    hits = _quote_once(hits, figure_pack)
    pack = _merge(_evidence_pack(hits, terms), figure_pack, _mitigation_sources(store, mitigations))

    summary = {
        "figures": figures,
        "figure_sources": [{"doc_id": s["doc_id"], "hours": s["hours"]} for s in cited_sources],
        "report_count": len(sources),
        "mitigations": mitigations,
        "mitigations_heading": mitigations_heading,
        "plan": plan.describe(),
        "scope": plan.filters_text(types=False),
    }
    text = narrator.answer(question, pack, summary)
    narrator_rejected = getattr(narrator, "last_rejection", None)
    pack_docs = {doc_id: store.get_document(doc_id) for doc_id in dict.fromkeys(e["doc_id"] for e in pack)}
    citations = [
        Citation(doc_id=e["doc_id"], doc_type=e["doc_type"], well=e["well"], date=e["date"], quote=e["quote"],
                page=quote_page(doc, e["quote"]) if (doc := pack_docs[e["doc_id"]]) else None)
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
        narrator_rejected=narrator_rejected,
    )
