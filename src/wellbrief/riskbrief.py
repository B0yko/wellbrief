"""Offset-well risk register for a well that has not been drilled yet.

The other commands answer questions after the fact. This one runs before
spud, reads the field's own history, and gives the drilling engineer a ranked
list of what is likely to go wrong, how much non-productive time it cost on
the offset wells, and what the wells that avoided it did differently.

The mitigations are not generated. They are quoted verbatim from the end of
well reports of the wells that did not have the problem, or from the
corrective actions of incident reports for the same failure, so each one can
be traced back to the document it came from. The mining itself (`miner.py`)
is shared with `qa.ask`: this module only tells it which wells count as
having avoided a discovered pattern.
"""

from __future__ import annotations

import re
from typing import Any

from .analytics import Pattern, find_patterns, finite_or_none, unavoidable_background
from .config import (
    DEFAULT_SPREAD_RATE_USD_PER_DAY,
    EQUIPMENT_MIN_RATE,
    EQUIPMENT_MIN_RATIO,
    EQUIPMENT_MIN_WELLS,
    NPT_CODES,
    RISK_MAX_RISKS,
    RISK_MIN_LIFT,
    RISK_MIN_SUPPORT,
    RISK_MIN_WELLS,
    hours_to_usd,
)
from .llm import Narrator, OfflineNarrator
from .miner import MinerScope, mine_mitigations
from .models import Citation, Risk, RiskBrief
from .quotes import evidence_quote
from .store import Store, index_manifest_hash
from .text import tokenize

DISCLAIMER = ("Decision support built from the offset archive. Review by a qualified drilling "
             "engineer is required.")


def _pattern_terms(pattern: Pattern) -> list[str]:
    """The terms a quote for a pattern is scored on: its code, hole section and formation."""
    return tokenize(f"{pattern.code} {pattern.hole_section} {pattern.formation}")


def _scope_for(pattern: Pattern) -> MinerScope:
    """The pattern as the shared miner (`miner.mine_mitigations`) sees it.

    `clean_wells` is always the population `analytics.find_patterns` already
    computed for this pattern (offset wells exposed to the same section and
    formation, or on the field's other rig or tool, that never had the
    event): a discovered pattern always knows who avoided it, unlike a
    query plan (`qa._mitigations_for`), which may not.
    """
    return MinerScope(code=pattern.code, field_name=pattern.field_name, hole_section=pattern.hole_section,
                      formation=pattern.formation, clean_wells=frozenset(pattern.clean_wells))


def _citations_for(store: Store, pattern: Pattern, limit: int = 4) -> list[Citation]:
    events = sorted(
        store.npt(
            field_name=pattern.field_name,
            code=pattern.code,
            hole_section=pattern.hole_section or None,
            formation=pattern.formation or None,
            rig=pattern.rig or None,
            mwd=pattern.mwd or None,
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


def _names(planned: str, category: str) -> bool:
    """Whether `--rig`/`--mwd` names a risk's category.

    A tool is reported with its vendor ("Parvane Downhole PJ-3"); a plan
    typically gives just the model ("PJ-3"). Either side may be the shorter
    one, so this matches whichever is contained in the other, as a whole
    token (a plan of "PJ-3" never matches a category of "PJ-35").
    """
    a, b = planned.strip(), category.strip()
    if not a or not b:
        return False
    if a.lower() == b.lower():
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return re.search(rf"(?<![\w-]){re.escape(short)}(?![\w-])", long, re.I) is not None


def _applies(pattern: Pattern, plan_rig: str | None, plan_mwd: str | None) -> str:
    """Whether the planned well's `--rig`/`--mwd` names this equipment risk's category.

    Always `applies` for an interval risk: it does not depend on rig or tool.
    For an equipment risk: `does_not_apply` when the plan names a different
    rig or tool, `not_specified` when the plan gives none, `applies`
    otherwise (spec: "When the flag is not given, equipment risks count as
    applying and are marked 'plan did not specify'").
    """
    if pattern.scope != "equipment":
        return "applies"
    planned = plan_rig if pattern.rig else plan_mwd
    if not planned:
        return "not_specified"
    return "applies" if _names(planned, pattern.rig or pattern.mwd) else "does_not_apply"


def _risk_id(pattern: Pattern) -> str:
    if pattern.scope == "equipment":
        category = re.sub(r"[^A-Za-z0-9]+", "", pattern.rig or pattern.mwd) or "X"
        return f"{pattern.code}-{category}"
    digits = re.sub(r"[^0-9]", "", pattern.hole_section) or "X"
    return f"{pattern.code}-{digits}"


def _title(pattern: Pattern) -> str:
    label = NPT_CODES.get(pattern.code, {}).get("label", pattern.code)
    if pattern.scope == "equipment":
        category = f"rig {pattern.rig}" if pattern.rig else f"MWD {pattern.mwd}"
        return f"{label} on {category}"
    return f"{label} in the {pattern.hole_section} section through {pattern.formation}"


def build_risk(store: Store, pattern: Pattern, spread_rate: float,
              plan_rig: str | None = None, plan_mwd: str | None = None) -> Risk:
    probability = pattern.probability
    expected_hours = probability * pattern.mean_hours_per_affected_well
    mitigations = mine_mitigations(store, _scope_for(pattern))
    citations = _citations_for(store, pattern)

    for m in mitigations:
        doc = store.get_document(m.doc_id)
        if doc:
            citations.append(Citation(
                doc_id=doc.doc_id, doc_type=doc.doc_type, well=doc.well,
                date=doc.date, quote=m.text,
            ))

    return Risk(
        risk_id=_risk_id(pattern),
        title=_title(pattern),
        code=pattern.code,
        scope=pattern.scope,
        hole_section=pattern.hole_section,
        formation=pattern.formation,
        rig=pattern.rig,
        mwd=pattern.mwd,
        depth_window_m=pattern.depth_window_m,
        wells_total=pattern.wells_total,
        wells_affected=pattern.wells_affected,
        probability=round(probability, 3),
        mean_npt_hours=round(pattern.mean_hours_per_affected_well, 1),
        p90_npt_hours=round(pattern.p90_hours_per_affected_well, 1),
        expected_npt_hours=round(expected_hours, 1),
        expected_cost_usd=round(hours_to_usd(expected_hours, spread_rate), 0),
        driver=pattern.driver,
        lift=finite_or_none(pattern.lift),
        ratio=finite_or_none(pattern.ratio),
        applies=_applies(pattern, plan_rig, plan_mwd),
        mitigations=[m.text for m in mitigations],
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
    max_risks: int = RISK_MAX_RISKS,
    plan_rig: str | None = None,
    plan_mwd: str | None = None,
    provenance: dict[str, Any] | None = None,
    risk_filters: bool = True,
) -> RiskBrief:
    """`risk_filters=False` (`eval --no-risk-filters`) drops the lift gate
    (`min_lift=0`) and lets the unavoidable codes back in, the baseline the
    filters are measured against; every other threshold, and the TD and
    max-risks cuts below, stay the same."""
    if risk_filters:
        patterns = find_patterns(store, field_name)
    else:
        patterns = find_patterns(store, field_name, min_lift=0.0, avoidable_only=False)
    # Only carry risks whose depth window the planned well will actually reach.
    patterns = [p for p in patterns if p.depth_window_m[0] <= planned_td_m + 50]

    risks = [build_risk(store, p, spread_rate, plan_rig, plan_mwd) for p in patterns]
    risks.sort(key=lambda r: -r.expected_cost_usd)
    risks = risks[:max_risks]

    brief = RiskBrief(
        well_name=well_name,
        field_name=field_name,
        planned_td_m=planned_td_m,
        generated_from_wells=[w.name for w in store.wells(field_name=field_name)],
        spread_rate_usd_per_day=spread_rate,
        risks=risks,
        unavoidable_hours=unavoidable_background(store, field_name),
        planned_rig=plan_rig,
        planned_mwd=plan_mwd,
        provenance=provenance or {},
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


def build_provenance(store: Store, field_name: str, spread_rate: float, narrator_name: str,
                     verification: dict[str, Any]) -> dict[str, Any]:
    """Everything a reader needs to know what a brief was built from and
    whether to trust it: version, when, from which corpus and index, the
    thresholds behind the statistics, the spread rate, the narrator, and the
    citation verification result.

    `corpus_hash` and `index_manifest_hash` are computed straight from the
    store and the on-disk index files (see `Store.corpus_hash` and
    `store.index_manifest_hash`); there is no separate per-field workspace
    manifest yet, so this is what "the corpus" and "the index" mean today.
    """
    from datetime import UTC, datetime

    from . import __version__

    return {
        "wellbrief_version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_hash": store.corpus_hash(field_name),
        "index_manifest_hash": index_manifest_hash(store.db_path),
        "thresholds": {
            "min_lift": RISK_MIN_LIFT,
            "min_wells": RISK_MIN_WELLS,
            "min_support": RISK_MIN_SUPPORT,
            "max_risks": RISK_MAX_RISKS,
            "equipment_min_ratio": EQUIPMENT_MIN_RATIO,
            "equipment_min_rate": EQUIPMENT_MIN_RATE,
            "equipment_min_wells": EQUIPMENT_MIN_WELLS,
        },
        "spread_rate_usd_per_day": spread_rate,
        "narrator": narrator_name,
        "verification": verification,
    }


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(brief: RiskBrief, verification: dict[str, Any]) -> str:
    """A Markdown rendering of the brief: a summary, a risk table, provenance
    and the disclaimer. Every value comes from the same `RiskBrief` the text
    and JSON renderings show, so the three never disagree."""
    verified = verification.get("ok", True)
    lines = [
        f"# Offset-well risk brief: {brief.well_name} ({brief.field_name})",
        "",
        "**NOT VERIFIED**" if not verified else "",
        "",
        f"Planned total depth: {brief.planned_td_m:,.0f} m"
        + (f", rig {brief.planned_rig}" if brief.planned_rig else "")
        + (f", MWD {brief.planned_mwd}" if brief.planned_mwd else ""),
        f"Built from {len(brief.generated_from_wells)} offset wells in {brief.field_name}.",
        "",
        f"Total expected NPT: **{brief.total_expected_npt_hours:.1f} h** "
        f"/ **${brief.total_exposure_usd:,.0f}** at ${brief.spread_rate_usd_per_day:,.0f}/day "
        "(risks that do not apply to this plan excluded).",
        "",
        "| # | Risk | Applies | Wells | Prob. | Mean h | P90 h | Expected h | Expected $ | Driver |",
        "|---|------|---------|-------|-------|--------|-------|------------|------------|--------|",
    ]
    for i, r in enumerate(brief.risks, start=1):
        lines.append(
            f"| {i} | {_md_escape(r.title)} | {r.applies.replace('_', ' ')} "
            f"| {r.wells_affected}/{r.wells_total} | {r.probability * 100:.0f} % | {r.mean_npt_hours:.1f} "
            f"| {r.p90_npt_hours:.1f} | {r.expected_npt_hours:.1f} | {r.expected_cost_usd:,.0f} "
            f"| {_md_escape(r.driver) or '-'} |"
        )
    lines.append("")
    if brief.unavoidable_hours:
        total = sum(brief.unavoidable_hours.values())
        codes = ", ".join(brief.unavoidable_hours)
        lines.append(f"Unavoidable background NPT: {total:.1f} h ({codes})")
        lines.append("")
    for i, r in enumerate(brief.risks, start=1):
        if not (r.mitigations or r.citations):
            continue
        lines.append(f"## {i}. {_md_escape(r.title)}")
        for m in r.mitigations[:3]:
            lines.append(f"- Mitigation: {_md_escape(m)}")
        if r.citations:
            lines.append(f"- Evidence: {', '.join(c.doc_id for c in r.citations[:4])}")
        lines.append("")

    lines.append("## Provenance")
    prov = brief.provenance
    if prov:
        lines.append(f"- wellbrief {prov.get('wellbrief_version')}, generated {prov.get('generated_at')}")
        lines.append(f"- corpus hash `{prov.get('corpus_hash', '')[:16]}`, "
                     f"index hash `{prov.get('index_manifest_hash', '')[:16] or 'n/a'}`")
        lines.append(f"- narrator: {prov.get('narrator')}")
        thresholds = prov.get("thresholds", {})
        lines.append("- thresholds: " + ", ".join(f"{k}={v}" for k, v in thresholds.items()))
        lines.append(f"- verification: {'all citations verbatim' if verified else 'NOT VERIFIED'} "
                     f"({verification.get('citations_checked', 0)} checked)")
    lines.append("")
    lines.append(DISCLAIMER)
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip("\n") + "\n"
