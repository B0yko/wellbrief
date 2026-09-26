"""Offset-well risk register for a well that has not been drilled yet.

The other commands answer questions after the fact. This one runs before
spud, reads the field's own history, and gives the drilling engineer a ranked
list of what is likely to go wrong, how much non-productive time it cost on
the offset wells, and what the wells that avoided it did differently.

The mitigations are not generated. They are quoted verbatim from the end of
well reports of the wells that did not have the problem, or from the
corrective actions of incident reports for the same failure, so each one can
be traced back to the document it came from.
"""

from __future__ import annotations

import re
from typing import Any

from .analytics import Pattern, find_patterns
from .config import DEFAULT_SPREAD_RATE_USD_PER_DAY, NPT_CODES, hours_to_usd
from .llm import Narrator, OfflineNarrator
from .models import Citation, Risk, RiskBrief
from .parse import parse_eowr, parse_incident
from .quotes import evidence_quote, verbatim_quote
from .search import mentions_code
from .store import Store
from .text import tokenize


def _pattern_terms(pattern: Pattern) -> list[str]:
    """The terms a quote for a pattern is scored on: its code, hole section and formation."""
    return tokenize(f"{pattern.code} {pattern.hole_section} {pattern.formation}")


def _relevance(sentence: str, pattern: Pattern) -> int:
    """Cheap term overlap between a written lesson and a detected pattern."""
    low = sentence.lower()
    score = 0
    if pattern.formation and pattern.formation.lower() in low:
        score += 3
    section_digits = re.sub(r"[^0-9/]", "", pattern.hole_section)
    if section_digits and section_digits in re.sub(r"[^0-9/]", "", low):
        score += 2
    if mentions_code(sentence, pattern.code):
        score += 2
    if re.search(r"\b(recommend|should|hold at|reduce|spot|condition|confirm|inspect)\b", low):
        score += 1
    return score


def mine_mitigations(store: Store, pattern: Pattern, limit: int = 4) -> list[tuple[str, str]]:
    """Pull the written fix out of the wells that avoided the problem.

    Preference order: end of well reports from clean wells, then corrective
    actions on incident reports for the same failure. Each sentence is
    returned as the raw span of its report, so it can be cited verbatim.
    """
    candidates: list[tuple[int, str, str]] = []

    clean = set(pattern.clean_wells)
    for doc in store.documents(doc_type="eowr", field_name=pattern.field_name):
        if doc.well not in clean:
            continue
        parsed = parse_eowr(doc.text)
        for sentence in list(parsed.get("lessons", [])) + list(parsed.get("recommendations", [])):
            score = _relevance(sentence, pattern)
            quote = verbatim_quote(doc.text, sentence) if score >= 3 else None
            if quote:
                candidates.append((score, quote, doc.doc_id))

    if len(candidates) < limit:
        for doc in store.documents(doc_type="incident", field_name=pattern.field_name):
            parsed = parse_incident(doc.text)
            if parsed.get("code") != pattern.code:
                continue
            for action in parsed.get("corrective_actions", []):
                score = _relevance(action, pattern)
                quote = verbatim_quote(doc.text, action) if score >= 1 else None
                if quote:
                    candidates.append((score, quote, doc.doc_id))

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for _score, sentence, doc_id in sorted(candidates, key=lambda c: -c[0]):
        key = re.sub(r"[^a-z0-9 ]", "", sentence.lower())[:90]
        if key in seen:
            continue
        seen.add(key)
        out.append((sentence, doc_id))
        if len(out) >= limit:
            break
    return out


def _citations_for(store: Store, pattern: Pattern, limit: int = 4) -> list[Citation]:
    events = sorted(
        store.npt(
            field_name=pattern.field_name,
            code=pattern.code,
            hole_section=pattern.hole_section,
            formation=pattern.formation,
        ),
        key=lambda e: -e.hours,
    )
    cites: list[Citation] = []
    terms = _pattern_terms(pattern)
    for e in events[:limit]:
        doc = store.get_document(e.doc_id)
        if not doc:
            continue
        cites.append(Citation(
            doc_id=doc.doc_id, doc_type=doc.doc_type, well=doc.well, date=doc.date,
            quote=evidence_quote(doc, terms, [e]),
        ))
    return cites


def build_risk(store: Store, pattern: Pattern, spread_rate: float) -> Risk:
    probability = pattern.probability
    expected_hours = probability * pattern.mean_hours_per_affected_well
    mitigations = mine_mitigations(store, pattern)
    citations = _citations_for(store, pattern)

    for text, doc_id in mitigations:
        doc = store.get_document(doc_id)
        if doc:
            citations.append(Citation(
                doc_id=doc.doc_id, doc_type=doc.doc_type, well=doc.well,
                date=doc.date, quote=text,
            ))

    label = NPT_CODES.get(pattern.code, {}).get("label", pattern.code)
    return Risk(
        risk_id=f"{pattern.code}-{re.sub(r'[^0-9]', '', pattern.hole_section) or 'X'}",
        title=f"{label} in the {pattern.hole_section} section through {pattern.formation}",
        code=pattern.code,
        hole_section=pattern.hole_section,
        formation=pattern.formation,
        depth_window_m=pattern.depth_window_m,
        wells_total=pattern.wells_total,
        wells_affected=pattern.wells_affected,
        probability=round(probability, 3),
        mean_npt_hours=round(pattern.mean_hours_per_affected_well, 1),
        p90_npt_hours=round(pattern.p90_event_hours, 1),
        expected_npt_hours=round(expected_hours, 1),
        expected_cost_usd=round(hours_to_usd(expected_hours, spread_rate), 0),
        driver=pattern.driver,
        mitigations=[m for m, _ in mitigations],
        citations=citations,
        counter_examples=pattern.clean_wells,
    )


def build_brief(
    store: Store,
    well_name: str,
    field_name: str,
    planned_td_m: float,
    spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY,
    narrator: Narrator | None = None,
    max_risks: int = 8,
) -> RiskBrief:
    patterns = find_patterns(store, field_name)
    # Only carry risks whose depth window the planned well will actually reach.
    patterns = [p for p in patterns if p.depth_window_m[0] <= planned_td_m + 50]

    risks = [build_risk(store, p, spread_rate) for p in patterns]
    risks.sort(key=lambda r: -r.expected_cost_usd)
    risks = risks[:max_risks]

    brief = RiskBrief(
        well_name=well_name,
        field_name=field_name,
        planned_td_m=planned_td_m,
        generated_from_wells=[w.name for w in store.wells(field_name=field_name)],
        spread_rate_usd_per_day=spread_rate,
        risks=risks,
    )
    brief.narrative = (narrator or OfflineNarrator()).risk_brief(brief)
    return brief


def verify_brief(brief: RiskBrief, store: Store) -> dict[str, Any]:
    """Confirm every citation points at a document that exists and quotes it exactly.

    A reference that does not resolve, or a quote that is not in its source,
    makes every other line of the brief suspect, so the check runs on every
    build, its result is part of the CLI output, and a failed check makes the
    `risk` command exit with status 2.
    """
    problems: list[str] = []
    checked = 0
    for risk in brief.risks:
        for cite in risk.citations:
            checked += 1
            doc = store.get_document(cite.doc_id)
            if doc is None:
                problems.append(f"{risk.risk_id}: unknown document {cite.doc_id}")
                continue
            haystack = " ".join(doc.text.split())
            if " ".join(cite.quote.split()) not in haystack:
                problems.append(f"{risk.risk_id}: quote not found verbatim in {cite.doc_id}")
    return {"citations_checked": checked, "problems": problems, "ok": not problems}
