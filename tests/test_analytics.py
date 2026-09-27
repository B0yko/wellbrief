"""Risk statistics and pattern detection: per-affected-well mean/P90, interval lift, the
equipment ratio rule (including the case a wells-affected-share rule misses), event
ownership (no event counted toward two listed patterns) and the unavoidable-code exclusion.

Uses the hand-built ledger in `analytics_ledger.py`: every expected figure below is restated
as its own arithmetic next to the assertion, so a test failure shows which number disagreed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import analytics_ledger as ledger
from wellbrief import analytics
from wellbrief.models import Document, NptEvent, Well
from wellbrief.store import Store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Store:
    return ledger.store(tmp_path_factory.mktemp("analytics"))


def by_code(patterns: list[analytics.Pattern], code: str) -> list[analytics.Pattern]:
    return [p for p in patterns if p.code == code]


# ---------------------------------------------------------------------------
# Per-affected-well mean and P90
# ---------------------------------------------------------------------------

def test_wellbore_instability_mean_and_p90_are_per_affected_well(store: Store) -> None:
    """RB-101 and RB-102 each carry one 6 h event: mean and P90 are both 6.0, not the
    field's raw event list (which would give the same 6.0 here only by coincidence of
    both wells having one event each -- the ownership test below is where a per-event
    P90 and a per-well P90 would actually disagree)."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0)
    found = by_code(patterns, "WELLBORE_INSTABILITY")
    assert len(found) == 1
    p = found[0]
    assert p.scope == "interval"
    assert p.hole_section == ledger.HOT_SECTION
    assert p.formation == ledger.HOT_FORMATION
    assert p.wells_total == 6      # every well has a report in the hot section
    assert p.wells_affected == 2   # RB-101, RB-102
    assert p.hours_total == pytest.approx(12.0)
    assert p.mean_hours_per_affected_well == pytest.approx(6.0)
    assert p.p90_hours_per_affected_well == pytest.approx(6.0)
    assert p.probability == pytest.approx(2 / 6)


# ---------------------------------------------------------------------------
# Interval lift
# ---------------------------------------------------------------------------

def test_wellbore_instability_lift_matches_hand_computed_rate_ratio(store: Store) -> None:
    """rate_here = 12 h / 24 drilling days in the hot section; field_rate = 12 h / 54
    field drilling days (there is no other WELLBORE_INSTABILITY on this ledger).
    lift = (12/24) / (12/54) = 54/24 = 2.25."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0)
    p = by_code(patterns, "WELLBORE_INSTABILITY")[0]
    assert p.lift == pytest.approx(54 / 24, abs=0.01)
    assert p.lift == pytest.approx(2.25, abs=0.01)


def test_lift_threshold_excludes_a_pattern_below_it(store: Store) -> None:
    """Raising `min_lift` past the pattern's own 2.25 drops it; the same call with the
    default 2.0 keeps it -- proof the gate is live, not a no-op."""
    kept = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2, min_lift=2.0)
    assert by_code(kept, "WELLBORE_INSTABILITY")
    dropped = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2, min_lift=2.3)
    assert not by_code(dropped, "WELLBORE_INSTABILITY")


# ---------------------------------------------------------------------------
# Equipment rule: hours-per-day ratio, not wells-affected share
# ---------------------------------------------------------------------------

def test_equipment_rig_repair_rate_ratio_is_about_12x(store: Store) -> None:
    """Rig-North: 24 h over 27 drilling days (3 wells x 9 days). Rig-South: 2 h over the
    same 27 days. ratio = (24/27) / (2/27) = 24/2 = 12.0."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0)
    rig_patterns = [p for p in patterns if p.scope == "equipment" and p.rig]
    assert len(rig_patterns) == 1
    p = rig_patterns[0]
    assert p.code == "RIG_REPAIR"
    assert p.rig == ledger.RIG_NORTH
    assert p.ratio == pytest.approx(12.0, abs=0.05)


def test_equipment_rule_catches_what_a_wells_affected_share_rule_would_miss(store: Store) -> None:
    """Rig-North: 3 of 3 wells affected (share 1.00). Rig-South: 2 of 3 (share 0.67).
    share_ratio = 1.00 / 0.67 = 1.5, below a 2x threshold -- an example of the
    prototype's rule missing this pattern. The hours-per-day ratio (12.0, see above) is
    what the product actually gates on."""
    share_north, share_south = 3 / 3, 2 / 3
    share_ratio = share_north / share_south
    assert share_ratio == pytest.approx(1.5)
    assert share_ratio < analytics.EQUIPMENT_MIN_RATIO   # below the threshold on wells-affected share alone

    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0, equipment_min_ratio=2.0,
                                       equipment_min_rate=0.4, equipment_min_wells=3)
    rig_patterns = [p for p in patterns if p.scope == "equipment" and p.rig]
    assert len(rig_patterns) == 1, "a share-ratio rule would find none; the ratio rule finds Rig-North"
    assert rig_patterns[0].rig == ledger.RIG_NORTH


def test_equipment_wells_threshold_excludes_a_small_fleet(store: Store) -> None:
    """Raising `equipment_min_wells` past Rig-North's 3 affected wells drops the pattern."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0, equipment_min_wells=4)
    assert not [p for p in patterns if p.scope == "equipment" and p.rig]


# ---------------------------------------------------------------------------
# Event ownership: an event counts toward at most one listed pattern
# ---------------------------------------------------------------------------

def test_rig_repair_interval_and_equipment_patterns_are_merged_not_duplicated(store: Store) -> None:
    """The interval bucket (RIG_REPAIR, 12 1/4", Bluff Shale) has all 5 events and would
    qualify on its own (5 of 6 wells, lift well above 2.0); it shares 3 of those 5 events
    (60 %) with the Rig-North equipment pattern, so it is merged into it rather than
    listed twice. Only one RIG_REPAIR pattern is listed, and its hours equal the true
    field total, not double the 26 h."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0)
    rig_repair = by_code(patterns, "RIG_REPAIR")
    assert len(rig_repair) == 1
    p = rig_repair[0]
    assert p.scope == "equipment"
    assert p.rig == ledger.RIG_NORTH
    assert p.hours_total == pytest.approx(26.0)   # 24 (North) + 2 (South), not 24 + 26
    assert p.events == 5


def test_rig_repair_merge_keeps_per_well_stats_scoped_to_the_equipment_fleet(store: Store) -> None:
    """The merge pulls in RB-104 and RB-105 (Rig-South) by event, but they are not on
    Rig-North's own fleet: `wells_affected` and the per-well mean/P90 stay scoped to the
    3 Rig-North wells (8 h each) so `probability` cannot exceed 1.0, while `hours_total`
    still counts every event."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0)
    p = by_code(patterns, "RIG_REPAIR")[0]
    assert p.wells_total == 3
    assert p.wells_affected == 3
    assert p.probability == pytest.approx(1.0)
    assert p.mean_hours_per_affected_well == pytest.approx(8.0)
    assert p.p90_hours_per_affected_well == pytest.approx(8.0)
    assert set(p.affected_wells) == {"RB-101", "RB-102", "RB-103"}


def test_no_true_event_double_counted_across_the_listed_patterns(store: Store) -> None:
    """Summed over every listed pattern of a code, the hours never exceed the true total
    for that code on this ledger (the merge test above checks the RIG_REPAIR case
    specifically; this checks it holds for every code the ledger plants)."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                       min_lift=2.0)
    true_totals: dict[str, float] = {}
    for e in store.npt(field_name=ledger.FIELD):
        true_totals[e.code] = true_totals.get(e.code, 0.0) + e.hours
    listed_totals: dict[str, float] = {}
    for p in patterns:
        listed_totals[p.code] = listed_totals.get(p.code, 0.0) + p.hours_total
    for code, total in listed_totals.items():
        assert total == pytest.approx(true_totals[code]), code
        assert total <= true_totals[code] + 1e-6, code


# ---------------------------------------------------------------------------
# Unavoidable codes
# ---------------------------------------------------------------------------

def test_unavoidable_code_never_becomes_a_pattern(store: Store) -> None:
    """WAIT_ON_WEATHER (avoidable = false) clears every threshold trivially (min_support 0,
    min_wells 1, min_lift 0) and still never appears as a pattern."""
    patterns = analytics.find_patterns(store, ledger.FIELD, min_support=0.0, min_wells=1,
                                       min_lift=0.0, equipment_min_ratio=0.0,
                                       equipment_min_rate=0.0, equipment_min_wells=1)
    assert not by_code(patterns, "WAIT_ON_WEATHER")


def test_unavoidable_background_line_totals_the_excluded_hours(store: Store) -> None:
    background = analytics.unavoidable_background(store, ledger.FIELD)
    assert background == {"WAIT_ON_WEATHER": 5.0}   # 3 h on RB-101 + 2 h on RB-104


def test_avoidable_only_false_lets_unavoidable_codes_back_in(store: Store) -> None:
    """`eval --no-risk-filters` passes `avoidable_only=False`; the pattern discovery
    itself, not just the background line, then considers WAIT_ON_WEATHER (2 of 6 wells,
    every threshold relaxed to trivial so only the avoidable-code gate is under test)."""
    filtered = analytics.find_patterns(store, ledger.FIELD, min_support=0.0, min_wells=1,
                                       min_lift=0.0)
    assert not by_code(filtered, "WAIT_ON_WEATHER")
    unfiltered = analytics.find_patterns(store, ledger.FIELD, min_support=0.0, min_wells=1,
                                         min_lift=0.0, avoidable_only=False)
    assert by_code(unfiltered, "WAIT_ON_WEATHER")


# ---------------------------------------------------------------------------
# JSON-safety of an unbounded ratio (Rig-South never had RIG_REPAIR before Rig-North's
# was compared against a zero baseline is not this ledger's case, so this exercises the
# actual mechanism directly)
# ---------------------------------------------------------------------------

def test_infinite_ratio_serialises_as_none_not_a_literal_infinity() -> None:
    assert analytics.finite_or_none(math.inf) is None
    assert analytics.finite_or_none(None) is None
    assert analytics.finite_or_none(12.345) == pytest.approx(12.35, abs=0.01)


# ---------------------------------------------------------------------------
# `interval_patterns`: the public wrapper that filters `find_patterns` down
# to interval scope (used by the evaluation adapter's discovery cases).
# ---------------------------------------------------------------------------

def test_interval_patterns_wrapper_excludes_equipment_scope(store: Store) -> None:
    """This ledger has both an equipment pattern (RIG_REPAIR on Rig-North) and an
    interval one (WELLBORE_INSTABILITY): the wrapper keeps only the latter kind."""
    wrapped = analytics.interval_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2,
                                         min_lift=2.0)
    assert wrapped
    assert all(p.scope == "interval" for p in wrapped)
    full = analytics.find_patterns(store, ledger.FIELD, min_support=0.3, min_wells=2, min_lift=2.0)
    assert any(p.scope == "equipment" for p in full)   # the field does have one; the wrapper drops it
    assert {p.code for p in wrapped} == {p.code for p in full if p.scope == "interval"}


# ---------------------------------------------------------------------------
# Event ownership on a genuine partial overlap (not 0 %, not 100 %+): an
# interval candidate that shares only some of its own events with a
# qualifying equipment pattern of the same code. `analytics_ledger.py`'s
# RIG_REPAIR case only ever exercises a >50 % overlap (full merge) or no
# overlap at all; this is the corner in between.
#
# Field "Millbrook": two rigs, Rig-Delta (D1..D4) and Rig-Echo (E1..E4).
# Every well has exactly one DDR in the hot section (12 1/4" / Test Shale)
# and nine in a quiet one (17 1/2" / Calm Marl); BOP_TEST_FAILURE events:
#   D1  4 h, hot section    (Rig-Delta AND the interval bucket: the overlap)
#   D2 20 h, quiet section  (Rig-Delta only)
#   D3 20 h, quiet section  (Rig-Delta only)
#   D4  -    clean
#   E1  2 h, quiet section  (field noise only: neither pattern claims it)
#   E2  5 h, hot section    (the interval bucket only)
#   E3  5 h, hot section    (the interval bucket only)
#   E4  5 h, hot section    (the interval bucket only)
# The interval bucket (hot section) starts as {D1, E2, E3, E4}: 4 events, one
# (D1, 25 %) shared with the Rig-Delta equipment pattern {D1, D2, D3}.
# ---------------------------------------------------------------------------

MILLBROOK = "Millbrook"
RIG_DELTA, RIG_ECHO = "Rig-Delta", "Rig-Echo"
MB_HOT_SECTION, MB_HOT_FORMATION = '12 1/4"', "Test Shale"
MB_QUIET_SECTION, MB_QUIET_FORMATION = '17 1/2"', "Calm Marl"
MB_WELLS = {"D1": RIG_DELTA, "D2": RIG_DELTA, "D3": RIG_DELTA, "D4": RIG_DELTA,
           "E1": RIG_ECHO, "E2": RIG_ECHO, "E3": RIG_ECHO, "E4": RIG_ECHO}
# (well, hours, section is hot?)
MB_EVENTS = [
    ("D1", 4.0, True), ("D2", 20.0, False), ("D3", 20.0, False),
    ("E1", 2.0, False), ("E2", 5.0, True), ("E3", 5.0, True), ("E4", 5.0, True),
]


def _millbrook_store(tmp_path: Path) -> Store:
    docs = []
    for well in MB_WELLS:
        docs.append(Document(
            doc_id=f"DDR-{well}-H", doc_type="ddr", well=well, field_name=MILLBROOK,
            date="2024-01-01", title="DAILY DRILLING REPORT", text="DAILY DRILLING REPORT\n",
            meta={"hole_section": MB_HOT_SECTION, "formation": MB_HOT_FORMATION, "rig": MB_WELLS[well]},
        ))
        for i in range(9):
            docs.append(Document(
                doc_id=f"DDR-{well}-Q{i:02d}", doc_type="ddr", well=well, field_name=MILLBROOK,
                date="2024-01-01", title="DAILY DRILLING REPORT", text="DAILY DRILLING REPORT\n",
                meta={"hole_section": MB_QUIET_SECTION, "formation": MB_QUIET_FORMATION,
                     "rig": MB_WELLS[well]},
            ))
    events = []
    for well, hours, hot in MB_EVENTS:
        section = MB_HOT_SECTION if hot else MB_QUIET_SECTION
        formation = MB_HOT_FORMATION if hot else MB_QUIET_FORMATION
        events.append(NptEvent(
            doc_id=f"DDR-{well}-{'H' if hot else 'Q00'}", well=well, field_name=MILLBROOK,
            date="2024-01-01", code="BOP_TEST_FAILURE", hours=hours, hole_section=section,
            formation=formation, depth_m=2500.0, mud_weight_sg=1.30, rig=MB_WELLS[well],
            description=f"BOP_TEST_FAILURE at {well}.",
        ))
    wells = [Well(name=w, field_name=MILLBROOK, rig=rig, spud_date="2024-01-01", td_m=3500.0,
                 sections=[MB_QUIET_SECTION, MB_HOT_SECTION],
                 formations=[MB_QUIET_FORMATION, MB_HOT_FORMATION])
            for w, rig in MB_WELLS.items()]
    s = Store(tmp_path / "wellbrief.db")
    s.put_documents(docs)
    s.put_wells(wells)
    s.put_npt(events)
    return s


def test_partial_event_overlap_keeps_both_patterns_without_double_counting(tmp_path: Path) -> None:
    """25 % overlap is at or below the merge threshold (`EVENT_OWNERSHIP_MERGE_SHARE`,
    0.5), so both patterns are listed rather than fully merged -- but D1's event, once
    claimed by the equipment pattern, is not also counted in the interval one."""
    store = _millbrook_store(tmp_path)
    patterns = analytics.find_patterns(store, MILLBROOK)
    found = [p for p in patterns if p.code == "BOP_TEST_FAILURE"]
    assert len(found) == 2, [(p.scope, p.rig, p.hole_section, p.events) for p in found]

    equipment = next(p for p in found if p.scope == "equipment")
    interval = next(p for p in found if p.scope == "interval")

    assert equipment.rig == RIG_DELTA
    assert equipment.events == 3                        # D1, D2, D3
    assert equipment.hours_total == pytest.approx(44.0)  # 4 + 20 + 20
    assert set(equipment.affected_wells) == {"D1", "D2", "D3"}

    assert interval.hole_section == MB_HOT_SECTION
    assert interval.formation == MB_HOT_FORMATION
    # D1 was claimed by the equipment pattern above: only E2, E3, E4 remain.
    assert interval.events == 3
    assert interval.hours_total == pytest.approx(15.0)   # 5 + 5 + 5, not 19 (D1 excluded)
    assert set(interval.affected_wells) == {"E2", "E3", "E4"}

    # The true field total for this code is 61 h (4+20+20+2+5+5+5); D1's 4 h counts
    # once (in the equipment pattern), and E1's 2 h is not claimed by either listed
    # pattern -- the invariant is "at most one", not "every event is listed somewhere".
    true_total = sum(h for _, h, _ in MB_EVENTS)
    assert true_total == pytest.approx(61.0)
    assert equipment.hours_total + interval.hours_total == pytest.approx(59.0)
    assert equipment.hours_total + interval.hours_total <= true_total + 1e-6

    # No document id claimed by one pattern is also claimed by the other.
    assert not (set(equipment.doc_ids) & set(interval.doc_ids))
    assert "DDR-D1-H" in equipment.doc_ids
    assert "DDR-D1-H" not in interval.doc_ids


# ---------------------------------------------------------------------------
# Equipment ratio must not degenerate to infinity for lack of a comparison
# group: a single-rig (or single-tool) field has no "rest of the fleet" to
# compare against, so no code confined to it may be reported as an
# equipment-driven pattern purely on that account.
# ---------------------------------------------------------------------------

def test_single_rig_field_never_reports_a_false_equipment_pattern(tmp_path: Path) -> None:
    field = "Solitaire"
    rig = "Only-Rig"
    section, formation = '9 5/8"', "Plain Shale"
    wells = [f"S{i}" for i in range(1, 7)]
    docs = []
    for well in wells:
        for i in range(10):
            docs.append(Document(
                doc_id=f"DDR-{well}-{i:02d}", doc_type="ddr", well=well, field_name=field,
                date="2024-01-01", title="DAILY DRILLING REPORT", text="DAILY DRILLING REPORT\n",
                meta={"hole_section": section, "formation": formation, "rig": rig},
            ))
    events = [
        NptEvent(doc_id=f"DDR-{well}-00", well=well, field_name=field, date="2024-01-01",
                 code="BOP_TEST_FAILURE", hours=5.0, hole_section=section, formation=formation,
                 depth_m=2000.0, mud_weight_sg=1.30, rig=rig, description="BOP test failure.")
        for well in wells[:4]   # S1..S4 affected, S5/S6 clean
    ]
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents(docs)
    store.put_wells([Well(name=w, field_name=field, rig=rig, spud_date="2024-01-01", td_m=3000.0,
                          sections=[section], formations=[formation]) for w in wells])
    store.put_npt(events)

    patterns = analytics.find_patterns(store, field)
    assert not [p for p in patterns if p.scope == "equipment"], (
        "a single-rig field has no comparison group; no equipment pattern should qualify from it"
    )
    # The only section field-wide, so the interval lift is exactly 1.0 (never >= min_lift
    # either) -- this ledger plants no pattern at all, only the bug this test guards.
    assert patterns == []


# ---------------------------------------------------------------------------
# example_topics: the material the UI's example questions are built from
# (GET /api/status -> app.js's renderExamples), read straight from a field's
# own ledger with no name of any kind written into this module.
# ---------------------------------------------------------------------------

def test_example_topics_reads_the_fields_own_busiest_code_section_and_formation(store: Store) -> None:
    # Riverbend's own numbers (see the module docstring above): RIG_REPAIR is the busiest
    # avoidable code by hours (26 h against WELLBORE_INSTABILITY's 12 h), entirely inside
    # 12 1/4" / Bluff Shale, which is also the field's busiest formation overall (its 38 h
    # of RIG_REPAIR + WELLBORE_INSTABILITY beats the quiet section's 5 h of weather).
    topics = analytics.example_topics(store, ledger.FIELD)
    assert topics == {
        "top_avoidable_code_label": "Rig equipment repair",
        "top_section_for_code": ledger.HOT_SECTION,
        "top_formation": ledger.HOT_FORMATION,
    }


def test_example_topics_is_empty_for_a_field_with_no_npt_events(store: Store) -> None:
    # A name absent from the ledger entirely: every value falls back to "" rather than
    # raising or picking an arbitrary code.
    assert analytics.example_topics(store, "No Such Field") == {
        "top_avoidable_code_label": "",
        "top_section_for_code": "",
        "top_formation": "",
    }


def test_example_topics_when_only_unavoidable_codes_are_recorded(tmp_path: Path) -> None:
    # No avoidable code exists in scope, so the code/section pair falls back to "" even
    # though the field does have NPT events and an overall busiest formation.
    field, well = "Placid", "PLC-101"
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([Document(
        doc_id=f"DDR-{well}-001", doc_type="ddr", well=well, field_name=field, date="2024-01-01",
        title="DAILY DRILLING REPORT", text="DAILY DRILLING REPORT\n",
        meta={"hole_section": '17 1/2"', "formation": "Fenmoor Marl"},
    )])
    store.put_npt([NptEvent(
        doc_id=f"DDR-{well}-001", well=well, field_name=field, date="2024-01-01",
        code="WAIT_ON_WEATHER", hours=4.0, hole_section='17 1/2"', formation="Fenmoor Marl",
        depth_m=2000.0, mud_weight_sg=1.30, rig="Rig-1", description="Waiting on weather.",
    )])
    assert analytics.example_topics(store, field) == {
        "top_avoidable_code_label": "",
        "top_section_for_code": "",
        "top_formation": "Fenmoor Marl",
    }


def test_example_topics_falls_back_when_the_busiest_code_has_no_recorded_section(
    tmp_path: Path,
) -> None:
    # An avoidable code exists, but its own events carry no hole section or formation: the
    # code's own label is still reported, while the section and (since it is the only code
    # recorded) the formation both fall back to "".
    field, well = "Rimwood", "RMW-101"
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([Document(
        doc_id=f"DDR-{well}-001", doc_type="ddr", well=well, field_name=field, date="2024-01-01",
        title="DAILY DRILLING REPORT", text="DAILY DRILLING REPORT\n",
    )])
    store.put_npt([NptEvent(
        doc_id=f"DDR-{well}-001", well=well, field_name=field, date="2024-01-01",
        code="RIG_REPAIR", hours=5.0, hole_section="", formation="",
        depth_m=2000.0, mud_weight_sg=1.30, rig="Rig-1", description="Fluid end replaced.",
    )])
    assert analytics.example_topics(store, field) == {
        "top_avoidable_code_label": "Rig equipment repair",
        "top_section_for_code": "",
        "top_formation": "",
    }
