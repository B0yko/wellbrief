"""Narrative layer.

Every number in an answer is computed before this module runs. What happens
here is only the wording. That split matters more than it sounds: any
narrator phrases the same computed figures, and a wrong answer is a retrieval
or arithmetic bug that can be found, rather than a model that made something
up.

The offline narrator is the default and, in this version, the only one. It
writes from templates over the evidence pack and needs no network and no
model weights. The `Narrator` base class is the interface another narrator
implements.
"""

from __future__ import annotations

import re
from typing import Any

# The first line of an answer that abstains (see `qa.abstention_text`).
NO_MATCH = "No records in this workspace match that question."


def _fmt_hours(h: float) -> str:
    return f"{h:.1f} h"


def _fmt_usd(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f} M"
    return f"${v:,.0f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


class Narrator:
    name = "base"

    def answer(self, question: str, evidence: list[dict[str, Any]], summary: dict[str, Any]) -> str:
        raise NotImplementedError

    def risk_brief(self, brief: Any) -> str:
        raise NotImplementedError


class OfflineNarrator(Narrator):
    """Deterministic prose. No model, no network, no download."""

    name = "offline"

    def answer(self, question: str, evidence: list[dict[str, Any]], summary: dict[str, Any]) -> str:
        retrieved = [ev for ev in evidence if ev.get("role", "retrieval") == "retrieval"]
        figures = summary.get("figures") or {}
        if not retrieved and not figures:
            return NO_MATCH
        lines: list[str] = []
        if figures.get("event_count"):
            n = figures["event_count"]
            lines.append(
                f"{n} NPT {'entry' if n == 1 else 'entries'} on {_plural(figures['well_count'], 'well')}: "
                f"{_fmt_hours(figures['total_hours'])} of non-productive time, "
                f"{_fmt_usd(figures['total_cost_usd'])} at {_fmt_usd(figures['spread_rate'])} per day."
            )
            if len(figures.get("by_field") or []) > 1:
                ff = ", ".join(f"{f['field']} {_fmt_hours(f['hours'])}" for f in figures["by_field"])
                lines.append(f"By field: {ff}.")
            if figures.get("by_code"):
                top = ", ".join(f"{c['code']} {_fmt_hours(c['hours'])}" for c in figures["by_code"][:3])
                lines.append(f"Largest codes: {top}.")
            if figures.get("by_well"):
                ww = ", ".join(f"{w['well']} {_fmt_hours(w['hours'])}" for w in figures["by_well"][:3])
                lines.append(f"Largest wells: {ww}.")
            sources = summary.get("figure_sources") or []
            if sources:
                cited = ", ".join(f"[{s['doc_id']}] {_fmt_hours(s['hours'])}" for s in sources)
                total = summary.get("report_count") or len(sources)
                lines.append(f"Largest contributing reports ({len(sources)} of {total}): {cited}.")
        elif figures:
            scope = summary.get("scope") or "no filter"
            where = "in this workspace" if scope == "no filter" else f"for {scope}"
            lines.append(f"No NPT entry is recorded {where}: {_fmt_hours(0.0)} of non-productive time.")
        if retrieved:
            if lines:
                lines.append("")
            lines.append("From the documents:")
            for ev in retrieved[:6]:
                lines.append(f"- {ev['quote']} [{ev['doc_id']}]")
        if summary.get("mitigations"):
            lines.append("")
            lines.append("Recorded mitigations on wells that avoided it:")
            for m in summary["mitigations"][:4]:
                lines.append(f"- {m['text']} [{m['doc_id']}]")
        return "\n".join(lines)

    def risk_brief(self, brief: Any) -> str:
        if not brief.risks:
            return (
                f"No recurring problem in the offset history of {brief.field_name} clears the "
                f"reporting threshold for {brief.well_name}. That is a statement about this archive, "
                f"not a statement that the well is low risk."
            )
        lines = [
            f"{brief.well_name} is planned to {int(brief.planned_td_m):,} m in {brief.field_name}. "
            f"The register below is built from {len(brief.generated_from_wells)} offset wells in the "
            f"same field.",
            "",
            f"Carried exposure across all listed risks: {_fmt_hours(brief.total_expected_npt_hours)} "
            f"of expected non-productive time, {_fmt_usd(brief.total_exposure_usd)} at "
            f"{_fmt_usd(brief.spread_rate_usd_per_day)} per day spread rate.",
            "",
        ]
        for i, r in enumerate(brief.risks, start=1):
            applies_note = {
                "does_not_apply": " (does not apply to this plan, excluded from totals)",
                "not_specified": " (plan did not specify; counted as applying)",
            }.get(r.applies, "")
            lines.append(f"{i}. {r.title}{applies_note}")
            lines.append(
                f"   Hit {r.wells_affected} of {r.wells_total} offset wells "
                f"({r.probability * 100:.0f} %). Mean {_fmt_hours(r.mean_npt_hours)} when it happens, "
                f"P90 {_fmt_hours(r.p90_npt_hours)}. Expected carry "
                f"{_fmt_hours(r.expected_npt_hours)} / {_fmt_usd(r.expected_cost_usd)}."
            )
            if r.scope == "equipment":
                lines.append(f"   Where: {r.driver or (r.rig and f'rig {r.rig}') or f'MWD {r.mwd}'}, "
                             f"{int(r.depth_window_m[0]):,}-{int(r.depth_window_m[1]):,} m.")
            else:
                lines.append(f"   Where: {r.hole_section} section, {r.formation}, "
                             f"{int(r.depth_window_m[0]):,}-{int(r.depth_window_m[1]):,} m.")
            if r.driver:
                lines.append(f"   Driver in the data: {r.driver}")
            for m in r.mitigations[:3]:
                lines.append(f"   Mitigation: {m}")
            if r.counter_examples:
                lines.append(f"   Wells that avoided it: {', '.join(r.counter_examples[:6])}")
            lines.append(f"   Evidence: {', '.join(c.doc_id for c in r.citations[:4])}")
            lines.append("")
        if brief.unavoidable_hours:
            total = sum(brief.unavoidable_hours.values())
            codes = ", ".join(brief.unavoidable_hours)
            lines.append(f"Unavoidable background NPT: {_fmt_hours(total)} ({codes}).")
            lines.append("")
        return "\n".join(lines).rstrip()


def get_narrator(backend: str = "offline") -> Narrator:
    """Return the narrator for a backend name.

    `offline` is the only narrator in this version, so every name resolves to
    it. A model-backed narrator would implement the `Narrator` interface above.
    """
    return OfflineNarrator()


CITE_RE = re.compile(r"\[([A-Z]{3,4}-[A-Z]{3}-\d{3}(?:-\d{3})?|[A-Z]{3,4}-[A-Z]{3}-\d{3})\]")


def extract_cited_ids(text: str) -> set[str]:
    return set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", text))
