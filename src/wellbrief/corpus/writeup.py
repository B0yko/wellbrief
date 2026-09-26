"""What the end-of-well and incident reports say, written from the daily reports.

Every lesson is one sentence with one true label (practice, failure or
neutral), and every fact it quotes (a depth, a mud weight, a count, a
temperature, hours) is read off the well's own daily reports. A practice
lesson is written only by a well that followed the practice and avoided the
problem. A lesson may cite another well's recommendation only when that well
avoided the problem and its end-of-well report is dated before this well
spudded. A field's recommendation for a pattern first appears in the report
of the first clean well that followed the practice, and in every report of
the field dated after it.
"""

from __future__ import annotations

import random

from .events import CORRECTIVE_ACTIONS, DEFAULT_ROOT_CAUSE, ROOT_CAUSES
from .fields import (
    CARBONATE,
    G3_BHT_LIMIT_C,
    G3_SECTION,
    G4_RIG,
    MWD_VENDOR,
    SALT,
)
from .records import Incident, NptEvent, Sentence, WellRecord, fmt_m

PATTERNS = ("G1", "G2", "G3", "G4")
PATTERN_CODE = {
    "G1": "STUCK_PIPE", "G2": "LOST_CIRCULATION", "G3": "DOWNHOLE_TOOL_FAILURE", "G4": "RIG_REPAIR",
}

RECOMMENDATIONS = {
    "G1": "Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m.",
    "G2": ("Condition the hole and spot an LCM pill before entering Vessra Carbonate; reduce flow rate "
           "below 450 gpm for the first 60 m of carbonate."),
    "G3": ("Run MWD tools rated to at least 150 C in the 12 1/4\" section and confirm the rating against "
           "the offset BHT profile at the planning stage."),
    "G4": "Inspect and change mud pump fluid ends between wells rather than on the critical path.",
}

FIELD_RECOMMENDATION = {
    "Orrindale": "Run a caliper on the intermediate section before running casing.",
    "Vessra South": ("Confirm the intermediate casing seat against the offset formation tops "
                     "before running casing."),
}

INCIDENT_MIN_TENTHS = 160
MAX_INCIDENTS_PER_WELL = 3
SALT_SECTION = '17 1/2"'


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def _reports(n: int) -> str:
    return "one daily report" if n == 1 else f"{n} daily reports"


def _hours(t: int) -> str:
    return f"{t / 10:.1f}"


# ---------------------------------------------------------------------------
# Pattern status and originators
# ---------------------------------------------------------------------------

def pattern_status(w: WellRecord) -> dict[str, str]:
    """affected (had an event of the pattern), clean (exposed, no event) or out_of_scope."""
    spec = w.spec
    carbonate = next((f for f in spec.formations if f.name == CARBONATE), None)
    exposed = {
        "G1": any(d.section.size == SALT_SECTION and d.formation_end.name == SALT for d in w.days),
        "G2": carbonate is not None and any(
            d.section is spec.sections[-1] and d.depth_end >= carbonate.top_m for d in w.days),
        "G3": any(d.section.size == G3_SECTION and d.bht_c > G3_BHT_LIMIT_C for d in w.days),
        "G4": True,
    }
    out = {}
    for p in PATTERNS:
        if p not in spec.patterns:
            out[p] = "out_of_scope"
        elif w.pattern_events(p):
            out[p] = "affected"
        elif exposed[p]:
            out[p] = "clean"
        else:
            out[p] = "out_of_scope"
    return out


def follows_practice(w: WellRecord, pattern: str) -> bool:
    if pattern not in w.spec.patterns:
        return False
    return {
        "G1": w.salt_mw_ok,
        "G2": w.lcm_pretreat,
        "G3": w.mwd == "PJ-5",
        "G4": w.rig != G4_RIG,
    }[pattern]


def originators(wells: list[WellRecord]) -> dict[str, WellRecord]:
    """The first clean well (by end-of-well report date) that followed each practice."""
    out: dict[str, WellRecord] = {}
    for p in PATTERNS:
        followers = [w for w in wells if follows_practice(w, p)]
        for w in followers:
            if w.patterns[p] != "clean":
                raise RuntimeError(f"{w.name} followed the {p} practice but is not clean")
        if followers:
            out[p] = min(followers, key=lambda w: (w.eowr_date, w.index))
    return out


def _carries(w: WellRecord, origin: WellRecord | None) -> bool:
    """A recommendation appears from its originator's report onwards."""
    return origin is not None and (origin is w or w.eowr_date > origin.eowr_date)


# ---------------------------------------------------------------------------
# Lessons and recommendations
# ---------------------------------------------------------------------------

def _first(events: list[NptEvent], variant_prefix: str) -> NptEvent | None:
    return next((e for e in events if e.variant.startswith(variant_prefix)), None)


def _reports_with(w: WellRecord, variant: str) -> int:
    return sum(1 for d in w.days if any(e.variant == variant for e in d.entries))


def _salt_lessons(w: WellRecord, cite: str) -> Sentence:
    spec = w.spec
    g1 = w.pattern_events("G1")
    packoff = _first(g1, "g1_packoff")
    creep = _reports_with(w, "g1_creep")
    mw = f"{w.salt_mw_sg:.2f}"
    casing = spec.sections[1].casing
    base = f"Keldra Salt was drilled at {mw} sg in the 17 1/2\" section"
    if packoff is not None:
        stuck = f"the string packed off at {fmt_m(packoff.depth_m)} m on the trip out for the {casing}"
        if _first(g1, "g1_fishing") is not None:
            stuck += ", and a fishing run was needed to recover the BHA"
        if creep:
            text = f"{base}; tight hole from salt creep was recorded on {_reports(creep)}, and {stuck}."
        else:
            text = f"{base}; {stuck}."
        return Sentence("lesson", text, "failure", ("G1",), "STUCK_PIPE")
    if creep:
        return Sentence("lesson", f"{base}; tight hole from salt creep was recorded on {_reports(creep)}.",
                        "failure", ("G1",), "WELLBORE_INSTABILITY")
    if not w.salt_mw_ok:
        return Sentence("lesson", f"{base}.", "neutral", ("G1",), None)
    trip_reports = [d for d in w.days if d.section is spec.sections[1]
                    and any(a.tag == "casing_trip_out" for a in d.activities)]
    text = f"Keldra Salt was drilled at {mw} sg with a saturated brine sweep every 250 m; hole held gauge"
    if trip_reports and not any(d.entries for d in trip_reports):
        text += f" and the trip out for the {casing} was uneventful."
    else:
        text += "."
    return Sentence("lesson", text + cite, "practice", ("G1",), "STUCK_PIPE")


def _carbonate_lesson(w: WellRecord, cite: str) -> Sentence:
    g2 = w.pattern_events("G2")
    total = _first(g2, "g2_total")
    if total is not None:
        text = (f"Total losses were taken on entering {CARBONATE} at {fmt_m(total.depth_m)} m; the 8 1/2\" "
                f"section was drilled into the carbonate top at full flow rate with no pre-treatment")
        partial = _reports_with(w, "g2_partial")
        text += f", and partial losses followed on {_reports(partial)}." if partial else "."
        return Sentence("lesson", text, "failure", ("G2",), "LOST_CIRCULATION")
    if not w.lcm_pretreat:
        raise RuntimeError(f"{w.name} drilled into the carbonate untreated without the planted losses")
    text = ("Flow rate was cut to 430 gpm and a 30 bbl fibrous LCM pill was spotted just above the "
            f"{CARBONATE} top before drilling in. Only seepage losses were recorded.")
    return Sentence("lesson", text + cite, "practice", ("G2",), "LOST_CIRCULATION")


def _mwd_lesson(w: WellRecord, cite: str) -> Sentence:
    g3 = w.pattern_events("G3")
    if g3:
        max_bht = max(e.params["bht"] for e in g3)
        return Sentence(
            "lesson",
            f"The {MWD_VENDOR} PJ-3 MWD failed {_times(len(g3))} in the 12 1/4\" section, at BHT up to "
            f"{max_bht} C; the tool was run above its demonstrated temperature rating.",
            "failure", ("G3",), "DOWNHOLE_TOOL_FAILURE")
    if w.mwd == "PJ-5":
        text = f"Ran {MWD_VENDOR} PJ-5 rated to 150 C in the 12 1/4\" section; no MWD failures above 118 C."
        return Sentence("lesson", text + cite, "practice", ("G3",), "DOWNHOLE_TOOL_FAILURE")
    bht = max(d.bht_c for d in w.days if d.section.size == G3_SECTION)
    return Sentence(
        "lesson",
        f"{MWD_VENDOR} PJ-3 directional tools were run in the 12 1/4\" section, where BHT reached {bht} C.",
        "neutral", ("G3",), None)


def _pump_lesson(w: WellRecord, cite: str) -> Sentence:
    g4 = w.pattern_events("G4")
    if g4:
        hours = sum(e.tenths for e in g4)
        return Sentence(
            "lesson",
            f"The mud pump #2 fluid end on {G4_RIG} failed {_times(len(g4))} "
            f"and was repaired on the critical path, {_hours(hours)} h in total.",
            "failure", ("G4",), "RIG_REPAIR")
    if w.rig != G4_RIG:
        text = f"Mud pump fluid ends on {w.rig} were inspected and changed between wells; no pump NPT."
        return Sentence("lesson", text + cite, "practice", ("G4",), "RIG_REPAIR")
    return Sentence("lesson", f"Mud pumps on {G4_RIG} were run on the rig's standard maintenance schedule.",
                    "neutral", ("G4",), None)


def _cement_lesson(w: WellRecord) -> Sentence:
    jobs = [e.params["casing"] for e in w.events if e.variant == "cement"]
    if not jobs:
        return Sentence("lesson", "Cement jobs went to programme with no notable deviation.",
                        "neutral", (), None)
    names = list(dict.fromkeys(jobs))
    jobs_text = " and ".join(f"the {c}" for c in names)
    plural = "jobs" if len(names) > 1 else "job"
    return Sentence(
        "lesson",
        f"The cement head seal leaked during the cement {plural} on {jobs_text} and was replaced before "
        f"displacement.",
        "failure", (), "CEMENT_ISSUE")


def write_lessons(w: WellRecord, origins: dict[str, WellRecord]) -> None:
    spec = w.spec

    def cite(pattern: str) -> str:
        origin = origins.get(pattern)
        if origin is None or origin is w or not origin.eowr_date < w.spud:
            return ""
        w.citations[pattern] = origin.name
        return f" This follows the recommendation in the {origin.name} end of well report."

    lessons: list[Sentence] = []
    if "G1" in spec.patterns:
        lessons.append(_salt_lessons(w, cite("G1") if follows_practice(w, "G1") else ""))
    if "G2" in spec.patterns:
        lessons.append(_carbonate_lesson(w, cite("G2") if follows_practice(w, "G2") else ""))
    if "G3" in spec.patterns:
        lessons.append(_mwd_lesson(w, cite("G3") if follows_practice(w, "G3") else ""))
    if "G4" in spec.patterns:
        lessons.append(_pump_lesson(w, cite("G4") if follows_practice(w, "G4") else ""))
    lessons.append(_cement_lesson(w))
    w.lessons = lessons

    recs = [Sentence("recommendation", FIELD_RECOMMENDATION[spec.name], "practice", (), None)]
    for p in PATTERNS:
        if _carries(w, origins.get(p)):
            recs.append(Sentence("recommendation", RECOMMENDATIONS[p], "practice", (p,), PATTERN_CODE[p]))
    w.recommendations = recs


# ---------------------------------------------------------------------------
# Incident reports
# ---------------------------------------------------------------------------

def write_incidents(w: WellRecord, rng: random.Random) -> None:
    """One report for each of the first three events of 16 h or more.

    The first corrective action of the code answers the root cause and is
    always there; one or two of the others are added.
    """
    incidents: list[Incident] = []
    for event in w.events:
        if event.tenths < INCIDENT_MIN_TENTHS or len(incidents) >= MAX_INCIDENTS_PER_WELL:
            continue
        table = CORRECTIVE_ACTIONS[event.code]
        others = list(range(1, len(table)))
        extra = rng.sample(others, min(len(others), rng.choice((1, 2))))
        actions = [Sentence("corrective_action", table[i][0], table[i][1], table[i][2], event.code)
                   for i in [0, *sorted(extra)]]
        number = len(incidents) + 1
        incidents.append(Incident(
            doc_id=f"INC-{w.name}-{number:02d}",
            number=number,
            event=event,
            severity="High" if event.tenths >= 240 else "Medium",
            root_cause=ROOT_CAUSES.get(event.code, DEFAULT_ROOT_CAUSE),
            actions=actions,
        ))
    w.incidents = incidents
