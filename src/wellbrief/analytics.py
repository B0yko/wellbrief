"""Non-productive time arithmetic and pattern detection.

Nothing here involves a language model. Totals, rates and drivers are
computed from the parsed NPT ledger with plain statistics, so they are
reproducible, auditable and identical on every run, and a total can be
checked line by line against a spreadsheet of the same events.

Two pattern scopes are detected. *Interval* patterns key on (code, hole
section, formation): a place in the well that keeps costing the same kind of
time. *Equipment* patterns key on (code, rig) or (code, MWD tool): a piece of
kit that costs more time than the rest of the same field's fleet, wherever it
is run. Each qualifies on its own statistical test (`_interval_candidates`,
`_equipment_candidates`); `find_patterns` runs both, then resolves event
ownership so a report's NPT entry counts toward at most one listed pattern.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field as dc_field
from typing import Any

from .config import (
    AVOIDABLE_CODES,
    DEFAULT_SPREAD_RATE_USD_PER_DAY,
    EQUIPMENT_MIN_RATE,
    EQUIPMENT_MIN_RATIO,
    EQUIPMENT_MIN_WELLS,
    NPT_CODES,
    RISK_MIN_LIFT,
    RISK_MIN_SUPPORT,
    RISK_MIN_WELLS,
    hours_to_usd,
)
from .models import NptEvent
from .store import Store

# An overlap above this share of an interval pattern's own events with a
# qualifying equipment pattern of the same code merges the interval pattern
# into the equipment one.
EVENT_OWNERSHIP_MERGE_SHARE = 0.5


def finite_or_none(value: float | None) -> float | None:
    """`value` rounded to 2 places, or None for a missing or infinite one
    (JSON has no literal for infinity)."""
    if value is None or math.isinf(value):
        return None
    return round(value, 2)


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


def rollup(events: Iterable[NptEvent],
           spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY) -> dict[str, Any]:
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


def npt_payload(store: Store, spread_rate: float = DEFAULT_SPREAD_RATE_USD_PER_DAY, *,
                field_name: str | None = None, well: str | None = None, code: str | None = None,
                since: str | None = None) -> dict[str, Any]:
    """The rollup over the matching ledger rows, as `wellbrief npt --json` and `GET /api/npt`
    both return it -- built once here from the same store filters, so the two never disagree.
    `since` is folded into the payload as its own key when given, alongside every key `rollup`
    itself already returns."""
    filters: dict[str, Any] = {}
    if field_name:
        filters["field_name"] = field_name
    if code:
        filters["code"] = code
    if well:
        filters["well"] = well
    if since:
        filters["since"] = since
    roll = rollup(store.npt(**filters), spread_rate)
    return {"since": since, **roll} if since else roll


@dataclass
class Pattern:
    """A problem that repeats across wells in the same place, or on the same
    piece of equipment.

    `wells_total` is the exposure denominator (offset wells that could have
    had the problem: a report in the section and formation for `interval`,
    a well on the rig or running the tool for `equipment`), never the whole
    field. `clean_wells` is the population the mitigation miner draws on: for
    `interval` it is the same exposure set minus the affected wells; for
    `equipment` it is the wells of the field *outside* the category (the
    other rig, or another tool) that never logged this code.
    """

    code: str
    scope: str                    # interval | equipment
    field_name: str
    hole_section: str = ""
    formation: str = ""
    rig: str = ""
    mwd: str = ""
    wells_total: int = 0
    wells_affected: int = 0
    events: int = 0
    hours_total: float = 0.0
    mean_hours_per_affected_well: float = 0.0
    p90_hours_per_affected_well: float = 0.0
    depth_window_m: tuple[float, float] = (0.0, 0.0)
    lift: float | None = None         # interval only
    ratio: float | None = None        # equipment only
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
            "scope": self.scope,
            "field_name": self.field_name,
            "hole_section": self.hole_section,
            "formation": self.formation,
            "rig": self.rig,
            "mwd": self.mwd,
            "wells_total": self.wells_total,
            "wells_affected": self.wells_affected,
            "probability": round(self.probability, 3),
            "events": self.events,
            "hours_total": round(self.hours_total, 1),
            "mean_hours_per_affected_well": round(self.mean_hours_per_affected_well, 1),
            "p90_hours_per_affected_well": round(self.p90_hours_per_affected_well, 1),
            "depth_window_m": [round(self.depth_window_m[0]), round(self.depth_window_m[1])],
            # Not JSON: a comparison group with zero baseline hours makes the
            # ratio/lift mathematically infinite rather than merely large, so
            # it is reported as unbounded (null) instead of a non-standard
            # `Infinity` token that not every JSON reader accepts.
            "lift": finite_or_none(self.lift),
            "ratio": finite_or_none(self.ratio),
            "affected_wells": self.affected_wells,
            "clean_wells": self.clean_wells,
            "driver": self.driver,
            "driver_detail": self.driver_detail,
        }


def _per_well_stats(events: list[NptEvent]) -> tuple[float, float, dict[str, float]]:
    """Per-affected-well hour sums, then mean and P90 taken over those sums
    (never over individual event hours: an affected well's own total is what
    the drilling engineer actually carries)."""
    per_well: dict[str, float] = defaultdict(float)
    for e in events:
        per_well[e.well] += e.hours
    sums = list(per_well.values())
    mean = statistics.fmean(sums) if sums else 0.0
    p90 = percentile(sums, 0.9)
    return mean, p90, per_well


def _depth_window(events: list[NptEvent]) -> tuple[float, float]:
    depths = [e.depth_m for e in events if e.depth_m]
    return (min(depths), max(depths)) if depths else (0.0, 0.0)


def _rate(hours: float, days: int) -> float:
    return hours / days if days else 0.0


def _ratio(rate_here: float, rate_rest: float) -> float:
    if rate_rest > 0:
        return rate_here / rate_rest
    return math.inf if rate_here > 0 else 0.0


class _Candidate:
    """A pattern still being qualified: raw event list plus the exposure
    context it was found in. Kept separate from `Pattern` so event-ownership
    merges can update the event list before any statistic is computed once,
    at the end, in `_finalize`."""

    def __init__(self, code: str, scope: str, field_name: str, events: list[NptEvent],
                 exposed_wells: set[str], clean_wells: set[str] | None = None, **key: str):
        self.code = code
        self.scope = scope
        self.field_name = field_name
        self.events = events
        self.exposed_wells = exposed_wells
        # The population the mitigation miner draws on. For `interval` this is
        # `exposed_wells` minus the affected wells (computed in `_finalize`,
        # since it never changes after a merge); for `equipment` it is fixed
        # here, because it names a different population (the rest of the
        # field's fleet) that a merge must not touch.
        self.clean_wells = clean_wells
        self.key = key            # hole_section/formation, or rig, or mwd
        self.lift: float | None = None
        self.ratio: float | None = None

    def event_ids(self) -> set[int]:
        return {id(e) for e in self.events}

    def merge(self, other: _Candidate) -> None:
        by_id = {id(e): e for e in self.events}
        for e in other.events:
            by_id[id(e)] = e
        self.events = list(by_id.values())


@dataclass
class _IntervalContext:
    """The exposure and baseline figures an interval candidate is tested
    against, computed once per `find_patterns` call so the event-ownership
    resolver can re-test a candidate on a reduced event list without
    recomputing them from the whole field again."""

    days_here: dict[tuple[str, str], int]
    exposed_here: dict[tuple[str, str], set[str]]
    total_days: int
    code_hours_field: dict[str, float]


def _interval_context(events: list[NptEvent], ddrs: list[dict[str, Any]]) -> _IntervalContext:
    days_here: Counter[tuple[str, str]] = Counter()
    exposed_here: dict[tuple[str, str], set[str]] = defaultdict(set)
    for d in ddrs:
        if d["hole_section"] and d["formation"]:
            key = (d["hole_section"], d["formation"])
            days_here[key] += 1
            exposed_here[key].add(d["well"])
    code_hours_field: dict[str, float] = defaultdict(float)
    for e in events:
        code_hours_field[e.code] += e.hours
    return _IntervalContext(dict(days_here), dict(exposed_here), len(ddrs), dict(code_hours_field))


def _qualify_interval(code: str, section: str, formation: str, bucket: list[NptEvent],
                      exposed: set[str], days: int, ctx: _IntervalContext, field_name: str,
                      min_wells: int, min_support: float, min_lift: float) -> _Candidate | None:
    """Build a `interval` candidate from `bucket` if it clears every
    threshold, or return None. Used both for the first pass over the whole
    field and, in `_resolve_ownership`, to re-test a candidate that has had
    events an equipment pattern already owns removed from it."""
    if not exposed or not bucket:
        return None
    affected = {e.well for e in bucket}
    if len(affected) < min_wells or len(affected) / len(exposed) < min_support:
        return None
    if not days or not ctx.total_days:
        return None
    field_rate = _rate(ctx.code_hours_field.get(code, 0.0), ctx.total_days)
    lift = _ratio(_rate(sum(e.hours for e in bucket), days), field_rate)
    if lift < min_lift:
        return None
    cand = _Candidate(code, "interval", field_name, list(bucket), exposed, None,
                      hole_section=section, formation=formation)
    cand.lift = lift
    return cand


def _interval_candidates(events: list[NptEvent], ctx: _IntervalContext, field_name: str,
                         min_wells: int, min_support: float, min_lift: float,
                         avoidable_only: bool = True) -> dict[tuple[str, ...], _Candidate]:
    buckets: dict[tuple[str, str, str], list[NptEvent]] = defaultdict(list)
    for e in events:
        if e.hole_section and e.formation and (not avoidable_only or e.code in AVOIDABLE_CODES):
            buckets[(e.code, e.hole_section, e.formation)].append(e)

    out: dict[tuple[str, ...], _Candidate] = {}
    for (code, section, formation), bucket in buckets.items():
        exposed = ctx.exposed_here.get((section, formation), set())
        days = ctx.days_here.get((section, formation), 0)
        cand = _qualify_interval(code, section, formation, bucket, exposed, days, ctx, field_name,
                                 min_wells, min_support, min_lift)
        if cand:
            out[(code, section, formation)] = cand
    return out


def _equipment_candidates(events: list[NptEvent], ddrs: list[dict[str, Any]], field_name: str,
                          dimension: str, min_ratio: float, min_rate: float, min_wells: int,
                          plausible_codes: set[str]) -> dict[tuple[str, ...], _Candidate]:
    """`dimension` is `rig` or `mwd`: which document field names the category.

    `plausible_codes` keeps the same physical constraint the driver logic
    already applies (a rig can plausibly explain a rig-floor failure, an MWD
    tool a downhole-tool failure) at the discovery stage, not only when
    explaining a pattern after the fact. Without it, a tool that is only ever
    run in one hole section (the corpus's own PJ-3, confined to 12 1/4") makes
    every section-specific code look like a false equipment pattern: the
    "rest of the fleet" bucket then has close to zero exposure to that code
    for a reason that has nothing to do with the tool.
    """
    days_by_cat: Counter[str] = Counter()
    wells_by_cat: dict[str, set[str]] = defaultdict(set)
    all_wells: set[str] = set()
    for d in ddrs:
        all_wells.add(d["well"])
        cat = d[dimension]
        if cat:
            days_by_cat[cat] += 1
            wells_by_cat[cat].add(d["well"])
    total_days = len(ddrs)

    code_hours_field: dict[str, float] = defaultdict(float)
    wells_with_code: dict[str, set[str]] = defaultdict(set)
    for e in events:
        code_hours_field[e.code] += e.hours
        wells_with_code[e.code].add(e.well)

    buckets: dict[tuple[str, str], list[NptEvent]] = defaultdict(list)
    for e in events:
        cat = e.mwd if dimension == "mwd" else e.rig
        if cat and e.code in plausible_codes:
            buckets[(e.code, cat)].append(e)

    out: dict[tuple[str, ...], _Candidate] = {}
    for (code, cat), bucket in buckets.items():
        days_here = days_by_cat.get(cat, 0)
        if not days_here:
            continue
        rest_days = total_days - days_here
        if rest_days <= 0:
            # No comparison population exists (a single rig, or a tool every
            # well runs): there is nothing to be "more than" here, so this
            # cannot be reported as an equipment-specific effect. Without this
            # guard, `_ratio` cannot tell "no comparison group" apart from "a
            # real comparison group that never saw this code" (rest_days > 0,
            # rest_hours == 0), which is where an informative infinite ratio
            # is supposed to come from.
            continue
        hours_here = sum(e.hours for e in bucket)
        rest_hours = code_hours_field[code] - hours_here
        ratio = _ratio(_rate(hours_here, days_here), _rate(rest_hours, rest_days))
        exposed = wells_by_cat.get(cat, set())
        affected = {e.well for e in bucket}
        share = len(affected) / len(exposed) if exposed else 0.0
        if ratio < min_ratio or share < min_rate or len(affected) < min_wells:
            continue
        # Clean wells for the miner: the rest of the field's fleet (not this
        # rig, or not running this tool), minus any well that ever logged
        # this code at all, on any rig or tool.
        clean = (all_wells - exposed) - wells_with_code[code]
        kwargs = {"mwd": cat} if dimension == "mwd" else {"rig": cat}
        cand = _Candidate(code, "equipment", field_name, list(bucket), exposed, clean, **kwargs)
        cand.ratio = ratio
        out[(code, dimension, cat)] = cand
    return out


def _resolve_ownership(interval: dict[tuple[str, ...], _Candidate],
                       equipment: dict[tuple[str, ...], _Candidate],
                       ctx: _IntervalContext, min_wells: int, min_support: float,
                       min_lift: float) -> list[_Candidate]:
    """An event counts toward at most one listed pattern.

    Two qualifying equipment patterns of the same code (a rig and an MWD
    tool both running hot on it, or two rigs that independently clear the
    equipment thresholds for it) are merged into the larger one first,
    whenever they share more than half of the smaller one's events.

    Every interval candidate of a code that a surviving equipment pattern
    also covers then has that equipment pattern's events removed from its own
    bucket before it is finalised: the equipment pattern is preferred, so it
    keeps every event it qualified on, whatever share of the interval
    candidate's own events that turns out to be. Above a half share, the
    interval candidate is folded into the equipment pattern entirely, events it did not share with it
    included, rather than left as a rump of its own non-equipment events.
    At or below a half share, only the shared events move; the interval
    candidate is re-tested on what is left (same thresholds, same field-wide
    baseline) and only listed if that remainder still qualifies on its own.
    Either way, no event that ends up counted in the equipment pattern is
    also counted in the interval one.
    """
    equip_items = sorted(equipment.items(), key=lambda kv: -sum(e.hours for e in kv[1].events))
    dropped: set[tuple[str, ...]] = set()
    for i, (key, cand) in enumerate(equip_items):
        if key in dropped:
            continue
        for other_key, other in equip_items[i + 1:]:
            if other_key in dropped or other.code != cand.code:
                continue
            other_ids = other.event_ids()
            if not other_ids:
                continue
            shared = len(cand.event_ids() & other_ids)
            if shared / len(other_ids) > EVENT_OWNERSHIP_MERGE_SHARE:
                cand.merge(other)
                dropped.add(other_key)
    equipment_final = {k: v for k, v in equip_items if k not in dropped}

    interval_final: dict[tuple[str, ...], _Candidate] = {}
    for key, cand in interval.items():
        own_ids = cand.event_ids()
        same_code_equip = [e for e in equipment_final.values() if e.code == cand.code]
        equip_ids: set[int] = set()
        for equip_cand in same_code_equip:
            equip_ids |= equip_cand.event_ids()
        shared_ids = own_ids & equip_ids if own_ids else set()
        if not shared_ids:
            interval_final[key] = cand
            continue
        if len(shared_ids) / len(own_ids) > EVENT_OWNERSHIP_MERGE_SHARE:
            # The equipment pattern that shares the most of this candidate's
            # events absorbs it, shared events or not. Any event this
            # candidate also shares with a *different* same-code equipment
            # pattern is left out of what gets absorbed (that other pattern
            # already owns it), so two equipment finals of the same code
            # can never end up sharing an event through this merge.
            target = max(same_code_equip, key=lambda e: len(e.event_ids() & own_ids))
            other_ids = equip_ids - target.event_ids()
            absorbed = _Candidate(cand.code, cand.scope, cand.field_name,
                                  [e for e in cand.events if id(e) not in other_ids],
                                  cand.exposed_wells)
            target.merge(absorbed)
            continue
        section = cand.key["hole_section"]
        formation = cand.key["formation"]
        remainder = [e for e in cand.events if id(e) not in shared_ids]
        requalified = _qualify_interval(
            cand.code, section, formation, remainder, cand.exposed_wells,
            ctx.days_here.get((section, formation), 0), ctx, cand.field_name,
            min_wells, min_support, min_lift,
        )
        if requalified is not None:
            interval_final[key] = requalified
        # Otherwise the remainder no longer clears its own thresholds: it is
        # dropped rather than listed twice or merged into a pattern whose own
        # exposure population it does not share.

    return [*interval_final.values(), *equipment_final.values()]


def _finalize(cand: _Candidate) -> Pattern:
    """Build the reported `Pattern` from a (possibly merged) candidate.

    `hours_total` and `events` cover every distinct event the candidate now
    owns, merges included. The per-well figures (`wells_affected`, `mean`/`p90`,
    `clean_wells`) stay scoped to the candidate's own exposed population, so
    a merge that absorbs an interval pattern reaching slightly outside an
    equipment category's own fleet cannot push `wells_affected` past
    `wells_total` (and `probability` past 1.0): a merge only happens above a
    50 % event overlap, so the wells outside the fleet are a small remainder,
    counted in the hours but not in the per-well population.
    """
    affected_all = {e.well for e in cand.events}
    in_scope = [e for e in cand.events if e.well in cand.exposed_wells] if cand.clean_wells is not None \
        else cand.events
    affected = sorted(affected_all & cand.exposed_wells) if cand.clean_wells is not None \
        else sorted(affected_all)
    mean, p90, _ = _per_well_stats(in_scope)
    clean = cand.clean_wells if cand.clean_wells is not None else cand.exposed_wells - set(affected)
    return Pattern(
        code=cand.code, scope=cand.scope, field_name=cand.field_name,
        hole_section=cand.key.get("hole_section", ""), formation=cand.key.get("formation", ""),
        rig=cand.key.get("rig", ""), mwd=cand.key.get("mwd", ""),
        wells_total=len(cand.exposed_wells), wells_affected=len(affected),
        events=len(cand.events), hours_total=sum(e.hours for e in cand.events),
        mean_hours_per_affected_well=mean, p90_hours_per_affected_well=p90,
        depth_window_m=_depth_window(cand.events), lift=cand.lift, ratio=cand.ratio,
        affected_wells=affected, clean_wells=sorted(clean),
        doc_ids=sorted({e.doc_id for e in cand.events}),
    )


def find_patterns(
    store: Store,
    field_name: str,
    min_support: float = RISK_MIN_SUPPORT,
    min_wells: int = RISK_MIN_WELLS,
    min_lift: float = RISK_MIN_LIFT,
    equipment_min_ratio: float = EQUIPMENT_MIN_RATIO,
    equipment_min_rate: float = EQUIPMENT_MIN_RATE,
    equipment_min_wells: int = EQUIPMENT_MIN_WELLS,
    avoidable_only: bool = True,
) -> list[Pattern]:
    """Every interval and equipment pattern of a field that clears its
    thresholds, event ownership resolved, drivers filled in, largest total
    hours first.

    `avoidable_only` excludes codes marked
    `avoidable = false` from ever becoming a pattern; `eval --no-risk-filters`
    passes `False`, together with `min_lift=0` and no equipment gate, to
    measure the filters against an unfiltered baseline. It never relaxes
    which codes are physically plausible for which equipment dimension (a rig
    does not explain a downhole tool failure): that is a structural
    constraint, not a tunable threshold.
    """
    ddrs = store.ddr_features(field_name)
    if not ddrs:
        return []
    events = store.npt(field_name=field_name)
    if not events:
        return []

    ctx = _interval_context(events, ddrs)
    interval = _interval_candidates(events, ctx, field_name, min_wells, min_support, min_lift,
                                    avoidable_only)
    rig_codes = (RIG_CODES & AVOIDABLE_CODES) if avoidable_only else RIG_CODES
    tool_codes = (TOOL_CODES & AVOIDABLE_CODES) if avoidable_only else TOOL_CODES
    equipment = {}
    equipment.update(_equipment_candidates(events, ddrs, field_name, "rig",
                                           equipment_min_ratio, equipment_min_rate, equipment_min_wells,
                                           rig_codes))
    equipment.update(_equipment_candidates(events, ddrs, field_name, "mwd",
                                           equipment_min_ratio, equipment_min_rate, equipment_min_wells,
                                           tool_codes))
    candidates = _resolve_ownership(interval, equipment, ctx, min_wells, min_support, min_lift)

    patterns = [_finalize(c) for c in candidates]
    for p in patterns:
        if p.scope == "equipment":
            p.driver, p.driver_detail = _equipment_driver(p)
        else:
            p.driver, p.driver_detail = explain_driver(store, p)

    patterns.sort(key=lambda p: -p.hours_total)
    return patterns


def _equipment_driver(pattern: Pattern) -> tuple[str, dict[str, Any]]:
    """An equipment pattern's driver is its own category: no discovery needed,
    it is the thing the pattern is keyed on."""
    if pattern.rig:
        label, attribute, category = "rig", "rig", pattern.rig
    else:
        label, attribute, category = "MWD", "tool_string", pattern.mwd
    ratio = round(pattern.ratio, 2) if pattern.ratio not in (None, math.inf) else None
    return f"{label} {category}", {"attribute": attribute, "category": category, "ratio": ratio}


def interval_patterns(store: Store, field_name: str, **kwargs: Any) -> list[Pattern]:
    """Convenience wrapper: only the interval-scope patterns of a field."""
    return [p for p in find_patterns(store, field_name, **kwargs) if p.scope == "interval"]


def unavoidable_background(store: Store, field_name: str) -> dict[str, float]:
    """Total hours per unavoidable code (weather, third-party standby, HSE
    stop): never risks, always reported as one background line."""
    events = store.npt(field_name=field_name)
    out: dict[str, float] = defaultdict(float)
    for e in events:
        if e.code not in AVOIDABLE_CODES:
            out[e.code] += e.hours
    return {code: round(hours, 1) for code, hours in sorted(out.items()) if hours}


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

    Interval patterns only: an equipment pattern's driver is its own rig or
    tool category, set directly in `find_patterns`.
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


def _categorical_rate(store: Store, pattern: Pattern, getter,
                      label: str) -> tuple[str, dict[str, Any]] | None:
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
