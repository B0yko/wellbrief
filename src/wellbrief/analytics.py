"""Non-productive time arithmetic and pattern detection.

Nothing here involves a language model. Totals, rates and drivers are
computed from the parsed NPT ledger with plain statistics, so they are
reproducible, auditable and identical on every run, and a total can be
checked line by line against a spreadsheet of the same events.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

from .config import (
    AVOIDABLE_CODES,
    DEFAULT_SPREAD_RATE_USD_PER_DAY,
    NPT_CODES,
    RISK_MIN_SUPPORT,
    RISK_MIN_WELLS,
    hours_to_usd,
)
from .models import NptEvent
from .store import Store


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation percentile. Empty input returns 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def rollup(events: Iterable[NptEvent], spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY) -> dict[str, Any]:
    events = list(events)
    total = sum(e.hours for e in events)
    avoidable = sum(e.hours for e in events if e.code in AVOIDABLE_CODES)

    def group(key) -> list[dict[str, Any]]:
        acc: dict[str, list[float]] = defaultdict(list)
        for e in events:
            acc[key(e)].append(e.hours)
        rows = [
            {
                "key": k,
                "hours": round(sum(v), 1),
                "events": len(v),
                "mean_hours": round(statistics.fmean(v), 1),
                "cost_usd": round(hours_to_usd(sum(v), spread_rate), 0),
            }
            for k, v in acc.items()
        ]
        return sorted(rows, key=lambda r: -r["hours"])

    by_code = group(lambda e: e.code)
    for row in by_code:
        row["code"] = row["key"]
        row["label"] = NPT_CODES.get(row["key"], {}).get("label", row["key"])
        row["avoidable"] = row["key"] in AVOIDABLE_CODES

    worst = group(lambda e: e.well)
    for row in worst:
        row["well"] = row["key"]

    return {
        "event_count": len(events),
        "well_count": len({e.well for e in events}),
        "total_hours": round(total, 1),
        "total_cost_usd": round(hours_to_usd(total, spread_rate), 0),
        "avoidable_hours": round(avoidable, 1),
        "avoidable_cost_usd": round(hours_to_usd(avoidable, spread_rate), 0),
        "avoidable_share": round(avoidable / total, 3) if total else 0.0,
        "by_code": by_code,
        "by_section": group(lambda e: e.hole_section or "unknown"),
        "by_formation": group(lambda e: e.formation or "unknown"),
        "by_rig": group(lambda e: e.rig or "unknown"),
        "by_field": group(lambda e: e.field_name),
        "worst_wells": worst,
        "spread_rate_usd_per_day": spread_rate,
    }


@dataclass
class Pattern:
    """A problem that repeats across wells in the same place."""

    code: str
    hole_section: str
    formation: str
    field_name: str
    wells_total: int
    wells_affected: int
    events: int
    hours_total: float
    mean_hours_per_affected_well: float
    p90_event_hours: float
    depth_window_m: tuple[float, float]
    affected_wells: list[str] = dc_field(default_factory=list)
    clean_wells: list[str] = dc_field(default_factory=list)
    doc_ids: list[str] = dc_field(default_factory=list)
    driver: str = ""
    driver_detail: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def probability(self) -> float:
        return self.wells_affected / self.wells_total if self.wells_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "hole_section": self.hole_section,
            "formation": self.formation,
            "field_name": self.field_name,
            "wells_total": self.wells_total,
            "wells_affected": self.wells_affected,
            "probability": round(self.probability, 3),
            "events": self.events,
            "hours_total": round(self.hours_total, 1),
            "mean_hours_per_affected_well": round(self.mean_hours_per_affected_well, 1),
            "p90_event_hours": round(self.p90_event_hours, 1),
            "depth_window_m": [round(self.depth_window_m[0]), round(self.depth_window_m[1])],
            "affected_wells": self.affected_wells,
            "clean_wells": self.clean_wells,
            "driver": self.driver,
            "driver_detail": self.driver_detail,
        }


def find_patterns(
    store: Store,
    field_name: str,
    min_support: float = RISK_MIN_SUPPORT,
    min_wells: int = RISK_MIN_WELLS,
) -> list[Pattern]:
    """Group NPT by where it physically happened, then keep what repeats."""
    wells = [w.name for w in store.wells(field_name=field_name)]
    if not wells:
        return []
    events = store.npt(field_name=field_name)

    buckets: dict[tuple[str, str, str], list[NptEvent]] = defaultdict(list)
    for e in events:
        buckets[(e.code, e.hole_section or "unknown", e.formation or "unknown")].append(e)

    patterns: list[Pattern] = []
    for (code, section, formation), bucket in buckets.items():
        affected = sorted({e.well for e in bucket})
        if len(affected) < min_wells or len(affected) / len(wells) < min_support:
            continue
        per_well: dict[str, float] = defaultdict(float)
        for e in bucket:
            per_well[e.well] += e.hours
        depths = [e.depth_m for e in bucket if e.depth_m]
        patterns.append(Pattern(
            code=code,
            hole_section=section,
            formation=formation,
            field_name=field_name,
            wells_total=len(wells),
            wells_affected=len(affected),
            events=len(bucket),
            hours_total=sum(e.hours for e in bucket),
            mean_hours_per_affected_well=statistics.fmean(per_well.values()),
            p90_event_hours=percentile([e.hours for e in bucket], 0.9),
            depth_window_m=(min(depths) if depths else 0.0, max(depths) if depths else 0.0),
            affected_wells=affected,
            clean_wells=sorted(set(wells) - set(affected)),
            doc_ids=sorted({e.doc_id for e in bucket}),
        ))

    for p in patterns:
        p.driver, p.driver_detail = explain_driver(store, p)

    patterns.sort(key=lambda p: -p.hours_total)
    return patterns


# Which attribute is allowed to explain which failure.
#
# Without this constraint the search happily reports that wells delayed by
# weather ran 0.03 sg heavier mud. The correlation is real in the arithmetic
# and meaningless in the well. Physics narrows the hypothesis space: a missed
# driver does less harm than a spurious one.
MUD_WEIGHT_CODES = {"STUCK_PIPE", "LOST_CIRCULATION", "WELLBORE_INSTABILITY", "HOLE_CLEANING", "FISHING"}
RIG_CODES = {"RIG_REPAIR", "BOP_TEST_FAILURE", "CEMENT_ISSUE", "HSE_STOP"}
TOOL_CODES = {"DOWNHOLE_TOOL_FAILURE", "FISHING"}

MW_MIN_SEPARATION_SIGMA = 1.2
MW_MIN_ABSOLUTE_SG = 0.03


def explain_driver(store: Store, pattern: Pattern) -> tuple[str, dict[str, Any]]:
    """Say what separates the wells that hit a pattern from the wells that did not.

    Candidates are tested in order of how actionable they are: a controllable
    drilling parameter first, then the rig, then the downhole tool string.
    Each candidate is only considered for failures it could physically cause,
    and nothing is reported unless the two groups actually separate.
    """
    if not pattern.affected_wells or not pattern.clean_wells:
        return "", {}

    affected = set(pattern.affected_wells)

    if pattern.code in MUD_WEIGHT_CODES:
        mw: dict[str, list[float]] = defaultdict(list)
        for d in store.documents(field_name=pattern.field_name, doc_type="ddr"):
            meta = d.meta or {}
            if meta.get("formation") != pattern.formation:
                continue
            if meta.get("hole_section") != pattern.hole_section:
                continue
            if meta.get("mud_weight_sg"):
                mw[d.well].append(float(meta["mud_weight_sg"]))
        a_vals = [statistics.fmean(v) for w, v in mw.items() if w in affected and v]
        c_vals = [statistics.fmean(v) for w, v in mw.items() if w not in affected and v]
        if len(a_vals) >= 2 and len(c_vals) >= 2:
            a_mean, c_mean = statistics.fmean(a_vals), statistics.fmean(c_vals)
            pooled = statistics.pstdev(a_vals + c_vals) or 1e-9
            sigma = abs(a_mean - c_mean) / pooled
            if sigma > MW_MIN_SEPARATION_SIGMA and abs(a_mean - c_mean) >= MW_MIN_ABSOLUTE_SG:
                direction = "lower" if a_mean < c_mean else "higher"
                return (
                    f"Mud weight in {pattern.formation} averaged {a_mean:.2f} sg on the wells that hit this "
                    f"and {c_mean:.2f} sg on the wells that did not. Affected wells ran {direction}.",
                    {"attribute": "mud_weight_sg", "affected_mean": round(a_mean, 3),
                     "clean_mean": round(c_mean, 3), "separation_sigma": round(sigma, 2)},
                )

    if pattern.code in RIG_CODES:
        found = _categorical_rate(store, pattern, lambda d: (d.meta or {}).get("rig", ""), "rig")
        if found:
            return found

    if pattern.code in TOOL_CODES:
        found = _categorical_rate(store, pattern, lambda d: (d.meta or {}).get("mwd", ""), "tool_string")
        if found:
            return found

    return "", {}


def _categorical_rate(store: Store, pattern: Pattern, getter, label: str) -> tuple[str, dict[str, Any]] | None:
    """Report a category whose hit rate is at least twice the rest."""
    well_cat: dict[str, str] = {}
    for d in store.documents(field_name=pattern.field_name, doc_type="ddr"):
        cat = getter(d)
        if cat:
            well_cat.setdefault(d.well, cat)
    if len(set(well_cat.values())) < 2:
        return None

    affected = set(pattern.affected_wells)
    totals: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for well, cat in well_cat.items():
        totals[cat] += 1
        if well in affected:
            hits[cat] += 1

    rates = {c: hits[c] / totals[c] for c in totals if totals[c] >= 3}
    if len(rates) < 2:
        return None
    worst = max(rates, key=lambda c: rates[c])
    rest = [r for c, r in rates.items() if c != worst]
    rest_mean = statistics.fmean(rest) if rest else 0.0
    if rates[worst] >= 0.4 and (rest_mean == 0.0 or rates[worst] / max(rest_mean, 1e-9) >= 2.0):
        return (
            f"{worst} hit this on {hits[worst]} of {totals[worst]} wells ({rates[worst] * 100:.0f} %) "
            f"against {rest_mean * 100:.0f} % on the rest of the fleet.",
            {"attribute": label, "category": worst, "rate": round(rates[worst], 3),
             "baseline_rate": round(rest_mean, 3), "wells": totals[worst]},
        )
    return None


def digest(store: Store, since: str, spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY) -> dict[str, Any]:
    """Summary of NPT since a date: totals, the top codes and the largest events."""
    events = store.npt(since=since)
    roll = rollup(events, spread_rate)
    biggest = sorted(events, key=lambda e: -e.hours)[:5]
    return {
        "since": since,
        "summary": {
            "events": roll["event_count"],
            "wells": roll["well_count"],
            "npt_hours": roll["total_hours"],
            "npt_cost_usd": roll["total_cost_usd"],
            "avoidable_share": roll["avoidable_share"],
        },
        "by_code": roll["by_code"][:5],
        "biggest_events": [
            {
                "doc_id": e.doc_id, "well": e.well, "date": e.date, "code": e.code,
                "hours": e.hours, "formation": e.formation, "hole_section": e.hole_section,
                "description": e.description[:220],
            }
            for e in biggest
        ],
    }
