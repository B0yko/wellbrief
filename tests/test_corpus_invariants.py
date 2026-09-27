"""Invariants of the synthetic corpus, checked on the written files.

Each test reads the generated text files back (with the product's parser
where it applies, and with small independent readers for the rest) and
checks one consistency rule a drilling engineer would expect of real
paperwork. Facts are always taken from the daily reports' text; the
ground-truth sidecar is only compared against them. Every corpus test runs
on the default seed and on seeds 7 and 42.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from corpus_files import DEPTH_QUOTE, DURATION_QUOTE, Ddr, ParsedCorpus, depth_value, ledger
from wellbrief.config import SYNTHETIC_FOOTER
from wellbrief.corpus import (
    SEED,
    OutputDirError,
    build_corpus,
    generator as generator_module,
    manifest_hash,
    write_corpus,
)
from wellbrief.corpus.events import CORRECTIVE_ACTIONS
from wellbrief.corpus.writeup import RECOMMENDATIONS

TRIP_OUT_CODES = {"STUCK_PIPE", "FISHING"}
G1_CODES = {"STUCK_PIPE", "FISHING", "WELLBORE_INSTABILITY"}
SALT = "Keldra Salt"
CARBONATE = "Vessra Carbonate"
SALT_SECTION = '17 1/2"'
MWD_SECTION = '12 1/4"'
WELL_ID = re.compile(r"^[A-Za-z]{2,5}-\d{2,4}$")
CITATION = re.compile(r"This follows the recommendation in the ([A-Z]{2,5}-\d{2,4}) end of well report\.")
TIMES = {"once": 1, "twice": 2, "one": 1}

PRACTICE_MARKER = {
    "G1": "hole held gauge",
    "G2": "Only seepage losses were recorded.",
    "G3": "Ran Parvane Downhole PJ-5 rated to 150 C",
    "G4": "were inspected and changed between wells; no pump NPT.",
}
PATTERN_FIELDS = {"G1": {"Orrindale"}, "G2": {"Vessra South"}, "G3": {"Orrindale", "Vessra South"},
                  "G4": {"Vessra South"}}

# Words every corrective action of a code must contain at least one of.
ACTION_KEYWORDS = {
    "STUCK_PIPE": ("salt", "mud weight", "pump out", "caliper"),
    "FISHING": ("fishing", "jars", "stuck"),
    "WELLBORE_INSTABILITY": ("salt", "overpull", "ream"),
    "LOST_CIRCULATION": ("LCM", "flow rate", "losses"),
    "DOWNHOLE_TOOL_FAILURE": ("MWD", "temperature", "BHT", "rated"),
    "RIG_REPAIR": ("mud pump", "fluid end"),
    "HOLE_CLEANING": ("cuttings", "sweep"),
    "CEMENT_ISSUE": ("cement",),
    "BOP_TEST_FAILURE": ("annular", "preventer"),
    "WAIT_ON_MATERIALS": ("barite",),
    "WAIT_ON_WEATHER": ("weather",),
    "THIRD_PARTY_STANDBY": ("crew", "standby"),
    "HSE_STOP": ("lifting", "stop-work"),
}
# The action that answers each code's root cause; every incident of the code carries it first.
ROOT_CAUSE_ACTION = {
    "STUCK_PIPE": "Raise mud weight to at least 1.42 sg",
    "FISHING": "Place the jars",
    "LOST_CIRCULATION": "Spot a fibrous LCM pill",
    "DOWNHOLE_TOOL_FAILURE": "Run MWD tools rated to at least 150 C",
}

TRIP_LINE = re.compile(
    r"^(Pulled out of hole(?: with the liner running tool)?|Ran in hole|Pumped out of hole|Reamed Keldra Salt"
    r"|Ran the [^.]*?) from (surface|[\d,]+ m) to (surface|[\d,]+ m)\b"
)
DRILL_LINE = re.compile(r"^Drilled (\S+(?: \d/\d)?\") hole from ([\d,]+) m to ([\d,]+) m\.$")
UNITS = {"m", "MD", "sg", "C", "h", "bbl", "bbl/h", "gpm", "t", "rpm", "cP", "%"}


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _count(word: str) -> int:
    return TIMES.get(word) or int(word.split()[0])


def _is_start(entry: dict[str, Any]) -> bool:
    return not entry["description"].startswith("Continued")


def _facts(pc: ParsedCorpus, well: str) -> dict[str, Any]:
    """What a well's daily reports say happened, read from the text."""
    ddrs = pc.ddrs_of(well)
    entries = [(d, e) for d in ddrs for e in d.entries]
    ops = " ".join(o.text for d in ddrs for o in d.ops)

    def starts(code: str, marker: str = "") -> list[dict[str, Any]]:
        return [e for _, e in entries if e["code"] == code and _is_start(e) and marker in e["description"]]

    return {
        "packoff": [e for _, e in entries if e["description"].startswith("String packed off")],
        "stuck": [e for _, e in entries if e["code"] == "STUCK_PIPE"],
        "fishing": [e for _, e in entries if e["code"] == "FISHING"],
        "total_losses": [e for _, e in entries if e["description"].startswith("Total losses on entering")],
        "total_loss_reports": [d for d, e in entries
                               if e["description"].startswith("Total losses on entering")],
        "losses": [e for _, e in entries if e["code"] == "LOST_CIRCULATION"],
        "partial_reports": {d.doc_id for d, e in entries
                            if e["code"] == "LOST_CIRCULATION" and "artial losses" in e["description"]},
        "g1": [e for d, e in entries
               if e["code"] in G1_CODES and d.section == SALT_SECTION and e["formation"] == SALT],
        "creep_reports": {d.doc_id for d, e in entries if e["code"] == "WELLBORE_INSTABILITY"},
        "mwd_failures": starts("DOWNHOLE_TOOL_FAILURE"),
        "mwd_bht": [int(m) for _, e in entries if e["code"] == "DOWNHOLE_TOOL_FAILURE"
                    for m in re.findall(r"BHT (\d+) C", e["description"])],
        "pump_repairs": starts("RIG_REPAIR", "fluid end"),
        "pump_hours": round(sum(e["hours"] for _, e in entries
                                if e["code"] == "RIG_REPAIR" and "fluid end" in e["description"]), 1),
        "cement": starts("CEMENT_ISSUE"),
        "mwd": {d.mwd for d in ddrs},
        "mwd_12": {d.mwd for d in ddrs if d.section == MWD_SECTION},
        "bht_12": max((int(d.labels["Static BHT estimate"].split()[0])
                       for d in ddrs if d.section == MWD_SECTION), default=0),
        "rig": {d.rig for d in ddrs},
        "salt_mw": {d.mud_weight for d in ddrs if d.section == SALT_SECTION and d.formation_at_td == SALT},
        "brine_sweeps": [depth_value(m)
                         for m in re.findall(r"Pumped a saturated brine sweep at ([\d,]+ m)", ops)],
        "lcm_pill": "Spotted a 30 bbl fibrous LCM pill" in ops,
        "pump_inspection": "changed the fluid ends before spud" in ops,
        "seepage": any("Seepage losses" in r for d in ddrs for r in d.remarks),
        "casing_trip_reports": [d for d in ddrs if d.section == SALT_SECTION
                                and any("to run the 13 3/8\" intermediate casing" in o.text for o in d.ops)],
    }


def _avoided(facts: dict[str, Any], pattern: str) -> bool:
    key = {"G1": "g1", "G2": "losses", "G3": "mwd_failures", "G4": "pump_repairs"}[pattern]
    return not facts[key]


def _lesson_pattern(lesson: str) -> str | None:
    return next((p for p, marker in PRACTICE_MARKER.items() if marker in lesson), None)


def _occurrences(pc: ParsedCorpus, well: str) -> list[dict[str, Any]]:
    """NPT events rebuilt from the text: a start entry plus its continuations in the next reports."""
    out: list[dict[str, Any]] = []
    open_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for d in pc.ddrs_of(well):
        for e in d.entries:
            key = (e["code"], e["depth_m"])
            if _is_start(e):
                occ = {"code": e["code"], "depth": e["depth_m"], "hours": e["hours"], "reports": [d.doc_id],
                       "formation": e["formation"], "section": d.section, "entries": [e]}
                out.append(occ)
                open_by_key[key] = occ
            else:
                occ = open_by_key[key]
                occ["hours"] = round(occ["hours"] + e["hours"], 1)
                occ["reports"].append(d.doc_id)
                occ["entries"].append(e)
    return out


def _flow(d: Ddr) -> int | None:
    m = re.search(r"/ (\d+) gpm", d.labels["WOB / RPM / Flow"])
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# One-off events
# ---------------------------------------------------------------------------

def test_one_off_events_happen_at_most_once_per_well(corpus: ParsedCorpus) -> None:
    packoffs: Counter[str] = Counter()
    total_losses: Counter[str] = Counter()
    fishing: Counter[str] = Counter()
    for d in corpus.ddrs:
        for e in d.entries:
            if "packed off" in e["description"]:
                packoffs[d.well] += 1
            if e["description"].startswith("Total losses on entering"):
                total_losses[d.well] += 1
            if e["code"] == "FISHING" and _is_start(e):
                fishing[d.well] += 1
    assert packoffs and total_losses, "the planted one-off events are missing"
    assert max(packoffs.values()) == 1
    assert max(total_losses.values()) == 1
    assert not fishing or max(fishing.values()) == 1
    # The sidecar agrees: one pack-off, fishing job and total-losses event per well at most.
    starts = Counter((e["well"], e["variant"]) for e in corpus.truth["npt_events"] if e["part"] == 1)
    assert all(n == 1 for (_, v), n in starts.items()
               if v in {"g1_packoff_freed", "g1_packoff_stuck", "g1_fishing", "g2_total"})


# ---------------------------------------------------------------------------
# NPT DETAIL entries: depth, formation, interval
# ---------------------------------------------------------------------------

def test_every_npt_entry_carries_its_own_depth_and_formation(corpus: ParsedCorpus) -> None:
    entries = 0
    for d in corpus.ddrs:
        labels = Counter(d.raw_npt_labels)
        assert (labels["Code"] == labels["Depth"] == labels["Formation"] == labels["Description"]
                == len(d.entries))
        for e in d.entries:
            entries += 1
            assert e["depth_m"] is not None and e["formation"], d.doc_id
    assert entries > 200


def test_depth_quoted_in_a_description_equals_the_entry_depth(corpus: ParsedCorpus) -> None:
    quoted = 0
    for d in corpus.ddrs:
        for e in d.entries:
            for q in DEPTH_QUOTE.findall(e["description"]):
                quoted += 1
                assert _num(q) == e["depth_m"], (d.doc_id, e["description"])
    assert quoted > 50


def test_drilling_events_lie_within_the_reported_interval(corpus: ParsedCorpus) -> None:
    for d in corpus.ddrs:
        for e in d.entries:
            if e["code"] in TRIP_OUT_CODES:
                continue
            assert d.depth_start <= e["depth_m"] <= d.depth_end, (d.doc_id, e)
    kinds = {(e["code"] in TRIP_OUT_CODES, e["kind"] == "trip_out") for e in corpus.truth["npt_events"]}
    assert kinds <= {(True, True), (False, False)}


def test_trip_out_events_lie_above_depth_at_end_in_the_formation_at_their_depth(corpus: ParsedCorpus) -> None:
    trips = 0
    for d in corpus.ddrs:
        top = corpus.section_top(d.field_name, d.section)
        for e in d.entries:
            if e["code"] not in TRIP_OUT_CODES:
                continue
            trips += 1
            assert top <= e["depth_m"] <= d.depth_end, (d.doc_id, e)
            assert e["formation"] == corpus.formation_at(d.field_name, e["depth_m"]), (d.doc_id, e)
    assert trips > 0


def test_every_g1_stuck_pipe_event_is_filed_under_keldra_salt_in_the_ledger(corpus: ParsedCorpus) -> None:
    rows: dict[str, Any] = {}
    for doc_id, group in _by_doc(ledger(corpus)).items():
        rows.update({f"{doc_id}#{n}": r for n, r in enumerate(group, start=1)})
    g1 = [e for e in corpus.truth["npt_events"] if e["pattern"] == "G1"]
    assert any(e["code"] == "STUCK_PIPE" for e in g1)
    for e in g1:
        row = rows[e["event_id"]]
        assert (row.code, row.hours) == (e["code"], e["hours"])
        assert row.formation == SALT, (e["event_id"], row.formation)
        assert row.hole_section == SALT_SECTION, e["event_id"]


def _by_doc(rows: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        out[r.doc_id].append(r)
    return out


def test_formation_at_td_is_the_formation_at_depth_at_end(corpus: ParsedCorpus) -> None:
    changed = 0
    for d in corpus.ddrs:
        assert d.formation_at_td == corpus.formation_at(d.field_name, d.depth_end), d.doc_id
        if corpus.formation_at(d.field_name, d.depth_start) != d.formation_at_td:
            changed += 1
    assert changed > 0, "no report crossed a formation top, so the rule was not exercised"


# ---------------------------------------------------------------------------
# Time accounting and the operations log
# ---------------------------------------------------------------------------

def test_productive_plus_npt_is_24_hours_and_npt_is_the_sum_of_entries(corpus: ParsedCorpus) -> None:
    for d in corpus.ddrs:
        assert round(d.productive + d.npt, 1) == 24.0, d.doc_id
        assert 0.0 <= d.npt <= 24.0, d.doc_id
        assert round(sum(e["hours"] for e in d.entries), 1) == d.npt, d.doc_id


def test_operations_log_covers_the_24_hours_and_its_npt_lines_match_npt_detail(corpus: ParsedCorpus) -> None:
    for d in corpus.ddrs:
        assert d.ops[0].start == 6 * 60 and d.ops[-1].end == 6 * 60, d.doc_id
        for o1, o2 in pairwise(d.ops):
            assert o1.end == o2.start, d.doc_id
        npt_lines = []
        for o in d.ops:
            assert o.minutes > 0, d.doc_id
            m = o.npt
            if m:
                assert o.minutes == round(float(m.group(2)) * 60), d.doc_id
                npt_lines.append((m.group(1), float(m.group(2)), m.group(3)))
        assert sum(o.minutes for o in d.ops) == 24 * 60, d.doc_id
        assert npt_lines == [(e["code"], e["hours"], e["description"]) for e in d.entries], d.doc_id


def test_durations_quoted_in_descriptions_do_not_exceed_the_entry_hours(corpus: ParsedCorpus) -> None:
    quoted = 0
    for d in corpus.ddrs:
        for e in d.entries:
            for q in DURATION_QUOTE.findall(e["description"]):
                quoted += 1
                assert float(q) <= e["hours"], (d.doc_id, e["description"])
    assert quoted > 0


def test_drilling_lines_are_continuous_and_add_up_to_the_progress(corpus: ParsedCorpus) -> None:
    by_well: dict[str, float] = {}
    for d in corpus.ddrs:
        assert d.depth_start == by_well.get(d.well, 0.0), d.doc_id
        by_well[d.well] = d.depth_end
        depth = d.depth_start
        for o in d.ops:
            m = DRILL_LINE.match(o.text)
            if not m:
                continue
            a, b = _num(m.group(2)), _num(m.group(3))
            assert a == depth and b > a, (d.doc_id, o.text)
            assert (b - a) / (o.minutes / 60) <= 25.0, (d.doc_id, o.text)
            depth = b
        assert depth == d.depth_end, d.doc_id


def test_trips_run_at_a_realistic_speed(corpus: ParsedCorpus) -> None:
    trips = 0
    for d in corpus.ddrs:
        for o in d.ops:
            m = TRIP_LINE.match(o.text)
            if not m:
                continue
            trips += 1
            distance = abs(depth_value(m.group(2)) - depth_value(m.group(3)))
            assert distance > 0, (d.doc_id, o.text)
            assert distance / (o.minutes / 60) <= 500.0, (d.doc_id, o.text, o.minutes)
    assert trips > 500


def test_no_lost_time_is_booked_as_productive(corpus: ParsedCorpus) -> None:
    drilling_only = re.compile(
        r"^(Drilled |Circulated bottoms up and took a survey|Pumped a saturated brine sweep)")
    for d in corpus.ddrs:
        for o in d.ops:
            if not o.npt and o.text.startswith("Circulated"):
                assert o.minutes <= 4 * 60, (d.doc_id, o.text)
        if d.drilled and not d.entries and d.depth_end - d.depth_start < 10:
            assert not all(drilling_only.match(o.text) for o in d.ops), d.doc_id


def test_every_line_is_at_most_100_characters(corpus: ParsedCorpus) -> None:
    for doc_id, text in corpus.texts.items():
        for line in text.splitlines():
            assert len(line) <= 100, (doc_id, line)


def test_no_line_starts_with_a_unit_split_from_its_number(corpus: ParsedCorpus) -> None:
    for doc_id, text in corpus.texts.items():
        for line in text.splitlines():
            words = line.split()
            if words:
                first = words[0].rstrip(".,;:)")
                assert first not in UNITS and not re.fullmatch(r'\d/\d"?', first), (doc_id, line)


def test_wrapped_lines_continue_at_the_value_or_item_column(corpus: ParsedCorpus) -> None:
    """A continuation line starts where the text of the line it continues started."""
    starts = [
        (re.compile(r"^  [A-Za-z][A-Za-z /-]{21}: "), 26),        # aligned label lines
        (re.compile(r"^  \d\d:\d\d-\d\d:\d\d  "), 15),          # operations log
        (re.compile(r"^  \d\. "), 5),                             # numbered lessons
        (re.compile(r"^  - "), 4),                                 # bullets
    ]
    wrapped = 0
    for doc_id, text in corpus.texts.items():
        indent = None
        for line in text.splitlines():
            if not line.strip() or not line.startswith(" "):
                indent = None   # a blank line or a heading ends the block
                continue
            match = next((col for rx, col in starts if rx.match(line)), None)
            if match is not None:
                indent = match
            elif indent is not None and line.startswith(" " * indent) and line[indent] != " ":
                wrapped += 1
            elif line.startswith("  ") and indent is not None:
                raise AssertionError(f"{doc_id}: continuation not aligned: {line!r}")
    assert wrapped > 100


# ---------------------------------------------------------------------------
# Well construction: mud, bits, casing sequences, tools
# ---------------------------------------------------------------------------

def test_mud_weight_holds_the_salt_until_the_intermediate_casing(corpus: ParsedCorpus) -> None:
    for well in corpus.eowrs:
        ddrs = corpus.ddrs_of(well)
        field_name = ddrs[0].field_name
        salt_top = corpus.formation_top(field_name, SALT)
        salt_mw = {d.mud_weight for d in ddrs if d.section == SALT_SECTION and d.formation_at_td == SALT}
        assert len(salt_mw) == 1, (well, salt_mw)
        after_top = [d for d in ddrs if d.section == SALT_SECTION and d.depth_end >= salt_top]
        assert {d.mud_weight for d in after_top} == salt_mw, well


def test_mud_weight_changes_only_in_steps(corpus: ParsedCorpus) -> None:
    for well in corpus.eowrs:
        ddrs = [d for d in corpus.ddrs_of(well) if d.drilled]
        for size in {d.section for d in ddrs}:
            weights = [d.mud_weight for d in ddrs if d.section == size]
            if size == SALT_SECTION:
                assert weights == sorted(weights) and len(set(weights)) <= 2, (well, weights)
            else:
                assert len(set(weights)) == 1, (well, size, weights)


def test_mud_systems_match_the_section_and_their_weight(corpus: ParsedCorpus) -> None:
    for d in corpus.ddrs:
        if d.drilled:
            assert d.labels["System"] == corpus.section(d.field_name, d.section)["mud_system"], d.doc_id
        if d.labels["System"] == "Salt-saturated polymer":
            assert d.mud_weight >= 1.20, d.doc_id
        if d.labels["System"].startswith("Spud mud"):
            assert d.mud_weight <= 1.10, d.doc_id
        if d.field_name == "Vessra South" and d.section == '8 1/2"' and d.drilled:
            assert d.mud_weight <= 1.15, d.doc_id


def test_bits_and_drilling_parameters_follow_the_operations(corpus: ParsedCorpus) -> None:
    for well in corpus.eowrs:
        ddrs = corpus.ddrs_of(well)
        for size in {d.section for d in ddrs}:
            bits = {d.labels["Bit"] for d in ddrs if d.section == size}
            assert len(bits) == 1, (well, size, bits)
            (bit,) = bits
            assert ("roller cone" in bit) == (size == '26"'), (well, bit)
    for d in corpus.ddrs:
        if d.drilled:
            assert re.fullmatch(r"\d+ t / \d+ rpm / \d+ gpm", d.labels["WOB / RPM / Flow"]), d.doc_id
            assert float(d.labels["ECD"].split()[0]) > d.mud_weight, d.doc_id
        else:
            assert d.labels["WOB / RPM / Flow"] == "- / - / -" and d.labels["ECD"] == "-", d.doc_id


def test_each_section_opens_with_a_shoe_drill_out_and_a_fit_and_casing_waits_on_cement(
    corpus: ParsedCorpus,
) -> None:
    for well in corpus.eowrs:
        ddrs = corpus.ddrs_of(well)
        field_name = ddrs[0].field_name
        ops = [o for d in ddrs for o in d.ops]
        texts = [o.text for o in ops]
        sections = corpus.field_spec(field_name)["sections"]
        for prev in sections[:-1]:
            shoe = prev["base_m"]
            assert any(t.startswith("Drilled out the float collar") and f"shoe at {shoe:,} m" in t
                       for t in texts)
            assert any(re.fullmatch(rf"Performed a FIT to \d\.\d\d sg EMW at {shoe + 3:,} m\.", t)
                       for t in texts)
        waits: list[int] = []
        for prev, o in pairwise([None, *ops]):
            if (o.text == "Displaced the cement and bumped the plug."
                    and (prev is None or prev.text != o.text)):
                waits.append(0)
            elif o.text == "Waited on cement.":
                waits[-1] += o.minutes
        assert len(waits) == len(sections) - 1 and min(waits) >= 8 * 60, (well, waits)


def test_lcm_pill_is_spotted_in_open_hole_just_above_the_carbonate(corpus: ParsedCorpus) -> None:
    pills = 0
    for d in corpus.ddrs:
        for o in d.ops:
            m = re.match(r"Spotted a 30 bbl fibrous LCM pill at ([\d,]+) m", o.text)
            if not m:
                continue
            pills += 1
            depth = _num(m.group(1))
            shoe = corpus.section(d.field_name, MWD_SECTION)["base_m"]
            assert shoe < depth < corpus.formation_top(d.field_name, CARBONATE), (d.doc_id, depth)
            assert d.section == '8 1/2"', d.doc_id
    assert pills > 0


def test_pj3_is_never_run_below_the_12_1_4_section(corpus: ParsedCorpus) -> None:
    for well, eowr in corpus.eowrs.items():
        ddrs = corpus.ddrs_of(well)
        assert {d.mwd for d in ddrs if d.section == '8 1/2"'} == {"Parvane Downhole PJ-5"}, well
        by_section = {d.section: d.mwd.split()[-1] for d in ddrs}
        (summary,) = [s for s in eowr.summary if s.startswith("MWD:")]
        if by_section[MWD_SECTION] == "PJ-3":
            assert summary == ("MWD: Parvane Downhole PJ-3 in the 26\" to 12 1/4\" sections and PJ-5 in the "
                               "8 1/2\" section."), well
            assert {by_section[s] for s in ('26"', SALT_SECTION, MWD_SECTION)} == {"PJ-3"}, well
        else:
            assert summary == "MWD: Parvane Downhole PJ-5 in every section.", well
            assert set(by_section.values()) == {"PJ-5"}, well


def test_mwd_failures_quote_the_bht_at_their_depth_on_pj3_in_12_1_4(corpus: ParsedCorpus) -> None:
    failures = 0
    for d in corpus.ddrs:
        spec = corpus.field_spec(d.field_name)
        for e in d.entries:
            if e["code"] != "DOWNHOLE_TOOL_FAILURE":
                continue
            failures += 1
            (bht,) = re.findall(r"BHT (\d+) C", e["description"])
            expected = round(spec["surface_temp_c"] + spec["geothermal_c_per_m"] * e["depth_m"])
            assert int(bht) == expected and int(bht) > 118, (d.doc_id, e["description"])
            assert d.mwd == "Parvane Downhole PJ-3" and d.section == MWD_SECTION, d.doc_id
    assert failures > 0


def test_only_a_repeat_pump_repair_is_called_one(corpus: ParsedCorpus) -> None:
    repairs = 0
    for well in corpus.eowrs:
        starts = _facts(corpus, well)["pump_repairs"]
        repairs += len(starts)
        repeats = ["Repeat failure" in e["description"] for e in starts]
        assert repeats == [n > 0 for n in range(len(starts))], well
    assert repairs > 0


# ---------------------------------------------------------------------------
# G1: the salt pack-off
# ---------------------------------------------------------------------------

def test_a_pack_off_followed_by_fishing_is_never_worked_free(corpus: ParsedCorpus) -> None:
    fished = 0
    for well in corpus.eowrs:
        f = _facts(corpus, well)
        freed = [e for e in f["stuck"]
                 if "jarred free" in e["description"] or "circulated free" in e["description"]]
        if f["fishing"]:
            fished += 1
            assert not freed, well
            fishing_hours = sum(e["hours"] for e in f["fishing"])
            assert fishing_hours >= 16.0, (well, fishing_hours)
        elif f["packoff"]:
            assert freed, well
    assert fished > 0 or not any(_facts(corpus, w)["packoff"] for w in corpus.eowrs)


def test_packed_off_wells_ream_the_salt_before_running_casing(corpus: ParsedCorpus) -> None:
    for well in corpus.eowrs:
        f = _facts(corpus, well)
        if not f["packoff"]:
            continue
        texts = [o.text for d in corpus.ddrs_of(well) for o in d.ops]
        stuck = next(i for i, t in enumerate(texts) if "String packed off" in t)
        casing = next(i for i, t in enumerate(texts) if t.startswith("Ran the 13 3/8\" intermediate casing"))
        assert any(t.startswith("Reamed Keldra Salt") for t in texts[stuck:casing]), well


# ---------------------------------------------------------------------------
# End-of-well reports
# ---------------------------------------------------------------------------

def test_eowr_failure_lessons_match_the_daily_reports(corpus: ParsedCorpus) -> None:
    for well, eowr in corpus.eowrs.items():
        f = _facts(corpus, well)
        lessons = " ".join(eowr.lessons)
        # G1: pack-off depth, fishing, creep reports, salt mud weight.
        m = re.search(r"the string packed off at ([\d,]+) m on the trip out "
                      r"for the 13 3/8\" intermediate casing", lessons)
        assert bool(m) == bool(f["packoff"]), well
        if m:
            assert _num(m.group(1)) == f["packoff"][0]["depth_m"], well
        assert ("fishing run" in lessons) == bool(f["fishing"]), well
        m = re.search(r"tight hole from salt creep was recorded on (one|\d+) daily reports?", lessons)
        assert bool(m) == bool(f["creep_reports"]), well
        if m:
            assert _count(m.group(1)) == len(f["creep_reports"]), well
        m = re.search(r"Keldra Salt was drilled at (\d\.\d\d) sg", lessons)
        if eowr.field_name == "Orrindale":
            assert m is not None and {float(m.group(1))} == f["salt_mw"], well
        # G2: total-losses depth, partial losses, flow rate.
        m = re.search(r"Total losses were taken on entering Vessra Carbonate at ([\d,]+) m", lessons)
        assert bool(m) == bool(f["total_losses"]), well
        if m:
            assert _num(m.group(1)) == f["total_losses"][0]["depth_m"], well
            flow = _flow(f["total_loss_reports"][0])
            assert flow is not None and flow >= 450, well
        m = re.search(r"partial losses followed on (one|\d+) daily reports?", lessons)
        assert bool(m) == bool(f["partial_reports"]), well
        if m:
            assert _count(m.group(1)) == len(f["partial_reports"]), well
        # G3: failure count and temperature, or the temperature reached without a failure.
        m = re.search(r"PJ-3 MWD failed (once|twice|\d+ times) in the 12 1/4\" section, "
                      r"at BHT up to (\d+) C", lessons)
        assert bool(m) == bool(f["mwd_failures"]), well
        if m:
            assert _count(m.group(1)) == len(f["mwd_failures"]), well
            assert int(m.group(2)) == max(f["mwd_bht"]), well
        m = re.search(r"PJ-3 directional tools were run in the 12 1/4\" section, "
                      r"where BHT reached (\d+) C", lessons)
        if m:
            assert not f["mwd_failures"] and f["mwd_12"] == {"Parvane Downhole PJ-3"}, well
            assert int(m.group(1)) == f["bht_12"], well
        # G4: repair count and hours.
        m = re.search(r"fluid end on Vessra-3 failed (once|twice|\d+ times) .* (\d+\.\d) h in total", lessons)
        assert bool(m) == bool(f["pump_repairs"]), well
        if m:
            assert _count(m.group(1)) == len(f["pump_repairs"]), well
            assert float(m.group(2)) == f["pump_hours"], well
        # Cementing: the casing strings named are the ones with a cement head leak.
        m = re.search(r"The cement head seal leaked during the cement jobs? on (.*) and was replaced",
                      lessons)
        assert bool(m) == bool(f["cement"]), well
        if m:
            named = set(re.findall(r"the (\d[^,]*?(?:casing|liner))", m.group(1)))
            leaked = {re.search(r"cementing the (.*?);", e["description"]).group(1) for e in f["cement"]}  # type: ignore[union-attr]
            assert named == leaked, well
        assert ("went to programme with no notable deviation" in lessons) == (not f["cement"]), well


def test_eowr_practice_lessons_come_only_from_wells_that_followed_the_practice(corpus: ParsedCorpus) -> None:
    seen: Counter[str] = Counter()
    for well, eowr in corpus.eowrs.items():
        f = _facts(corpus, well)
        for lesson in eowr.lessons:
            p = _lesson_pattern(lesson)
            if p is None:
                continue
            seen[p] += 1
            assert _avoided(f, p), (well, p)
            if p == "G1":
                assert min(f["salt_mw"]) >= 1.42 and f["brine_sweeps"], well
                sweeps = sorted(f["brine_sweeps"])
                assert all(b - a <= 250 for a, b in pairwise(sweeps)), well
                if "uneventful" in lesson:
                    trip_reports = f["casing_trip_reports"]
                    assert trip_reports and not any(d.entries for d in trip_reports), well
            elif p == "G2":
                assert f["lcm_pill"] and f["seepage"], well
            elif p == "G3":
                assert f["mwd_12"] == {"Parvane Downhole PJ-5"}, well
            else:
                assert "Vessra-3" not in f["rig"] and f["pump_inspection"], well
    assert set(seen) == {"G1", "G2", "G3", "G4"}


def test_seepage_is_claimed_only_where_the_daily_reports_record_it(corpus: ParsedCorpus) -> None:
    for well, eowr in corpus.eowrs.items():
        f = _facts(corpus, well)
        claimed = any("Only seepage losses were recorded." in lesson for lesson in eowr.lessons)
        assert claimed == f["seepage"] == (f["lcm_pill"] and not f["losses"]), well
        for d in corpus.ddrs_of(well):
            if any("Seepage losses" in r for r in d.remarks):
                assert _flow(d) == 430, d.doc_id


def test_cross_well_citations_point_to_an_earlier_clean_eowr_with_that_recommendation(
    corpus: ParsedCorpus,
) -> None:
    citations = 0
    for well, eowr in corpus.eowrs.items():
        for lesson in eowr.lessons:
            m = CITATION.search(lesson)
            if not m:
                continue
            citations += 1
            pattern = _lesson_pattern(lesson)
            assert pattern is not None, lesson
            cited = corpus.eowrs[m.group(1)]
            assert cited.well != well
            assert cited.field_name == eowr.field_name
            assert cited.date < eowr.spud, (well, cited.well)
            assert _avoided(_facts(corpus, cited.well), pattern), (well, cited.well)
            assert RECOMMENDATIONS[pattern] in cited.recommendations, (well, cited.well)
    assert citations > 0


def test_recommendations_appear_only_from_the_first_clean_well_that_wrote_them(corpus: ParsedCorpus) -> None:
    checked = 0
    for field_name in {e.field_name for e in corpus.eowrs.values()}:
        eowrs = sorted((e for e in corpus.eowrs.values() if e.field_name == field_name),
                       key=lambda e: (e.date, e.well))
        for pattern, rec in RECOMMENDATIONS.items():
            carriers = [e for e in eowrs if rec in e.recommendations]
            writers = [e for e in eowrs if any(PRACTICE_MARKER[pattern] in lesson for lesson in e.lessons)]
            if field_name not in PATTERN_FIELDS[pattern]:
                assert not carriers and not writers
                continue
            if not carriers:
                assert not writers
                continue
            checked += 1
            origin = carriers[0]
            assert origin is writers[0], (field_name, pattern, origin.well, writers[0].well)
            assert _avoided(_facts(corpus, origin.well), pattern)
            for e in carriers[1:]:
                assert e.date > origin.date, (field_name, pattern, e.well)
            # Every later report carries it: once written, the recommendation stays in the field.
            assert [e for e in eowrs if e.date > origin.date] == carriers[1:]
    assert checked >= 4


def test_each_pattern_has_a_practice_lesson_written_only_by_clean_wells(corpus: ParsedCorpus) -> None:
    status = {(w["well"], p): s for w in corpus.truth["wells"] for p, s in w["patterns"].items()}
    for pp in corpus.truth["planted_patterns"]:
        practice = [s for s in corpus.truth["sentences"]
                    if s["field"] == pp["field"] and pp["id"] in s["patterns"]
                    and s["kind"] == "lesson" and s["label"] == "practice"]
        assert practice, (pp["id"], pp["field"])
        for s in practice:
            assert status[(s["well"], pp["id"])] == "clean", (pp["id"], s["well"])
            assert s["text"] in corpus.eowrs[s["well"]].lessons
            assert _avoided(_facts(corpus, s["well"]), pp["id"]), (pp["id"], s["well"])


def test_eowr_header_and_npt_by_code_equal_the_daily_reports(corpus: ParsedCorpus) -> None:
    for well, eowr in corpus.eowrs.items():
        ddrs = corpus.ddrs_of(well)
        sums: dict[str, float] = defaultdict(float)
        for d in ddrs:
            for e in d.entries:
                sums[e["code"]] += e["hours"]
        assert {code: round(h, 1) for code, h in sums.items()} == eowr.npt_by_code, well
        total = re.search(r"Non-productive time\s+: (\d+\.\d) h", eowr.text)
        assert total is not None
        assert float(total.group(1)) == round(sum(sums.values()), 1), well
        assert eowr.days == len(ddrs) and eowr.spud == ddrs[0].date, well
        assert eowr.td == ddrs[-1].depth_end == max(d.depth_end for d in ddrs), well


# ---------------------------------------------------------------------------
# Incident reports
# ---------------------------------------------------------------------------

def test_incident_corrective_actions_are_specific_to_the_npt_code(corpus: ParsedCorpus) -> None:
    codes_by_action: dict[str, set[str]] = defaultdict(set)
    for inc in corpus.incidents:
        assert 2 <= len(inc.actions) <= 3, inc.doc_id
        table = [text for text, _, _ in CORRECTIVE_ACTIONS[inc.code]]
        assert inc.actions[0] == table[0], inc.doc_id
        for action in inc.actions:
            assert action in table, (inc.doc_id, action)
            codes_by_action[action].add(inc.code)
        if inc.code in ROOT_CAUSE_ACTION:
            assert inc.actions[0].startswith(ROOT_CAUSE_ACTION[inc.code]), inc.doc_id
    assert corpus.incidents
    assert all(len(codes) == 1 for codes in codes_by_action.values())
    for code, table in CORRECTIVE_ACTIONS.items():
        for text, _, _ in table:
            assert any(k.lower() in text.lower() for k in ACTION_KEYWORDS[code]), (code, text)
    all_actions = [text for table in CORRECTIVE_ACTIONS.values() for text, _, _ in table]
    assert len(all_actions) == len(set(all_actions))
    for s in corpus.truth["sentences"]:
        if s["kind"] == "corrective_action":
            inc = next(i for i in corpus.incidents if i.doc_id == s["doc_id"])
            assert s["code"] == inc.code


def test_incident_reports_describe_their_own_npt_event(corpus: ParsedCorpus) -> None:
    by_well: dict[str, list[Any]] = defaultdict(list)
    for inc in corpus.incidents:
        by_well[inc.well].append(inc)
        assert "Operations were suspended" not in inc.text, inc.doc_id
        occ = next(o for o in _occurrences(corpus, inc.well) if o["reports"][0] == inc.reports[0]
                   and o["code"] == inc.code and o["depth"] == inc.depth)
        assert occ["reports"] == inc.reports, inc.doc_id
        assert inc.hours == occ["hours"] >= 16.0, inc.doc_id
        assert (inc.depth, inc.formation, inc.section) == (occ["depth"], occ["formation"], occ["section"])
        assert inc.date == corpus.ddr(inc.reports[0]).date, inc.doc_id
    for well in corpus.eowrs:
        long_events = [(o["reports"][0], o["code"]) for o in _occurrences(corpus, well) if o["hours"] >= 16.0]
        assert [(i.reports[0], i.code) for i in by_well[well]] == long_events[:3], well
    assert "LOST_CIRCULATION" in {i.code for i in corpus.incidents}


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def test_every_document_ends_with_the_footer_and_it_is_never_parsed_as_content(corpus: ParsedCorpus) -> None:
    for doc_id, text in corpus.texts.items():
        lines = text.splitlines()
        assert lines[-1] == SYNTHETIC_FOOTER, doc_id
        assert lines.count(SYNTHETIC_FOOTER) == 1, doc_id
    marker = "Synthetic demonstration document"
    for d in corpus.ddrs:
        assert all(marker not in e["description"] for e in d.entries)
    for e in corpus.eowrs.values():
        assert all(marker not in s for s in e.lessons + e.recommendations)
    for inc in corpus.incidents:
        assert all(marker not in a for a in inc.actions)


# ---------------------------------------------------------------------------
# The ground-truth sidecar against the text
# ---------------------------------------------------------------------------

def _label(kind: str, text: str) -> str:
    """An independent marker rule for the true label of a candidate sentence."""
    if kind == "recommendation":
        return "practice"
    if kind == "corrective_action":
        neutral = ("outside the crew's control", "recorded against the service contract",
                   "was made correctly")
        return "neutral" if any(m in text for m in neutral) else "practice"
    failure = ("packed off", "tight hole", "failed", "Total losses were taken", "seal leaked")
    practice = ("hole held gauge", "Only seepage losses", "no MWD failures", "no pump NPT")
    if any(m in text for m in failure):
        return "failure"
    if any(m in text for m in practice):
        return "practice"
    return "neutral"


def test_truth_sentences_are_exactly_the_parsed_sentences_with_true_labels(corpus: ParsedCorpus) -> None:
    by_doc: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in corpus.truth["sentences"]:
        by_doc[(s["doc_id"], s["kind"])].append(s["text"])
        assert s["label"] == _label(s["kind"], s["text"]), s
    for e in corpus.eowrs.values():
        assert by_doc[(e.doc_id, "lesson")] == e.lessons
        assert by_doc[(e.doc_id, "recommendation")] == e.recommendations
    for inc in corpus.incidents:
        assert by_doc[(inc.doc_id, "corrective_action")] == inc.actions
    documents = {d["doc_id"] for d in corpus.truth["documents"]}
    assert documents == set(corpus.texts)
    assert {s["label"] for s in corpus.truth["sentences"]} == {"practice", "failure", "neutral"}


def test_parsed_ledger_equals_the_true_npt_events(corpus: ParsedCorpus) -> None:
    got = []
    for doc_id, rows in _by_doc(ledger(corpus)).items():
        mwd = corpus.ddr(doc_id).mwd.split()[-1]
        got += [(f"{doc_id}#{n}", r.code, r.hours, r.depth_m, r.formation, r.hole_section, r.date, r.rig, mwd,
                 r.well, r.field_name) for n, r in enumerate(rows, start=1)]
    want = [(e["event_id"], e["code"], e["hours"], float(e["depth_m"]), e["formation"], e["section"],
             e["date"], e["rig"], e["mwd"], e["well"], e["field"]) for e in corpus.truth["npt_events"]]
    assert sorted(got) == sorted(want)


def _text_pattern(d: Ddr, e: dict[str, Any]) -> str | None:
    if (d.field_name == "Orrindale" and e["code"] in G1_CODES and d.section == SALT_SECTION
            and e["formation"] == SALT):
        return "G1"
    if d.field_name == "Vessra South" and e["code"] == "LOST_CIRCULATION":
        return "G2"
    if e["code"] == "DOWNHOLE_TOOL_FAILURE":
        return "G3"
    if e["code"] == "RIG_REPAIR" and d.rig == "Vessra-3" and "fluid end" in e["description"]:
        return "G4"
    return None


def test_truth_patterns_match_what_the_daily_reports_show(corpus: ParsedCorpus) -> None:
    truth_events = {e["event_id"]: e for e in corpus.truth["npt_events"]}
    affected: dict[tuple[str, str], bool] = defaultdict(bool)
    for d in corpus.ddrs:
        for n, e in enumerate(d.entries, start=1):
            pattern = _text_pattern(d, e)
            assert truth_events[f"{d.doc_id}#{n}"]["pattern"] == pattern, (d.doc_id, e)
            if pattern:
                affected[(d.well, pattern)] = True
    for w in corpus.truth["wells"]:
        ddrs = corpus.ddrs_of(w["well"])
        carbonate = corpus.field_spec(w["field"])["formations"][-1]
        exposed = {
            "G1": w["field"] == "Orrindale" and any(d.section == SALT_SECTION and d.formation_at_td == SALT
                                                    for d in ddrs),
            "G2": w["field"] == "Vessra South" and any(d.section == '8 1/2"'
                                                       and d.depth_end >= carbonate["top_m"]
                                                       for d in ddrs),
            "G3": any(d.section == MWD_SECTION and int(d.labels["Static BHT estimate"].split()[0]) > 118
                      for d in ddrs),
            "G4": w["field"] == "Vessra South",
        }
        for p, status in w["patterns"].items():
            want = "affected" if affected[(w["well"], p)] else "clean" if exposed[p] else "out_of_scope"
            assert status == want, (w["well"], p)


def test_truth_documents_carry_the_sections_and_formations_of_the_text(corpus: ParsedCorpus) -> None:
    docs = {d["doc_id"]: d for d in corpus.truth["documents"]}
    for d in corpus.ddrs:
        formations = corpus.field_spec(d.field_name)["formations"]
        want = {corpus.formation_at(d.field_name, d.depth_start)}
        want |= {f["name"] for f in formations if d.depth_start < f["top_m"] <= d.depth_end}
        want |= {e["formation"] for e in d.entries}
        t = docs[d.doc_id]
        assert (t["type"], t["well"], t["field"], t["date"]) == ("ddr", d.well, d.field_name, d.date)
        assert t["sections"] == [d.section] and set(t["formations"]) == want, d.doc_id
    for well, eowr in corpus.eowrs.items():
        ddrs = corpus.ddrs_of(well)
        t = docs[eowr.doc_id]
        assert (t["type"], t["date"]) == ("eowr", eowr.date)
        assert t["sections"] == list(dict.fromkeys(d.section for d in ddrs)), well
        assert set(t["formations"]) == {f for d in ddrs for f in docs[d.doc_id]["formations"]}, well
    for inc in corpus.incidents:
        t = docs[inc.doc_id]
        assert (t["type"], t["date"], t["code"], t["hours"], t["depth_m"]) == (
            "incident", inc.date, inc.code, inc.hours, inc.depth)
        assert t["sections"] == [inc.section] and t["formations"] == [inc.formation], inc.doc_id
        assert [i.split("#")[0] for i in t["event_ids"]] == inc.reports, inc.doc_id


def test_truth_wells_match_the_daily_reports(corpus: ParsedCorpus) -> None:
    for w in corpus.truth["wells"]:
        ddrs = corpus.ddrs_of(w["well"])
        eowr = corpus.eowrs[w["well"]]
        f = _facts(corpus, w["well"])
        assert {w["rig"]} == f["rig"], w["well"]
        assert w["mwd_by_section"] == {d.section: d.mwd.split()[-1] for d in ddrs}, w["well"]
        assert {f"Parvane Downhole {w['mwd']}"} == f["mwd_12"], w["well"]
        dates = (ddrs[0].date, ddrs[-1].date, eowr.date)
        assert (w["spud"], w["release"], w["eowr_date"]) == dates, w["well"]
        assert (w["td_m"], w["days_on_well"]) == (eowr.td, len(ddrs)), w["well"]
        assert {w["salt_mw_sg"]} == f["salt_mw"], w["well"]
        if w["field"] == "Orrindale":
            assert w["salt_mw_ok"] == (w["salt_mw_sg"] >= 1.38) == bool(f["brine_sweeps"]), w["well"]
        else:
            assert w["lcm_pretreat"] == f["lcm_pill"], w["well"]


def test_planted_patterns_are_strong_enough_to_rediscover(corpus: ParsedCorpus) -> None:
    """Each planted pattern has at least 3 affected wells; an equipment pattern hits at least 40 % of the
    wells on its rig or tool (the product's equipment rule)."""
    wells = corpus.truth["wells"]
    for pp in corpus.truth["planted_patterns"]:
        field_wells = [w for w in wells if w["field"] == pp["field"]]
        affected = [w for w in field_wells if w["patterns"][pp["id"]] == "affected"]
        assert len(affected) >= 3, (pp["id"], pp["field"], len(affected))
        if pp["scope"] == "equipment":
            key = pp["keys"][0]
            category = [w for w in field_wells if w["mwd"] == key["mwd"]] if "mwd" in key else \
                [w for w in field_wells if w["rig"] == key["rig"]]
            hit = [w for w in category if w["patterns"][pp["id"]] == "affected"]
            assert len(category) >= 3 and len(hit) / len(category) >= 0.4, (pp["id"], pp["field"])
            assert all(w in category for w in affected), (pp["id"], pp["field"])


# ---------------------------------------------------------------------------
# Schedule, determinism, output directory, scale, random streams
# ---------------------------------------------------------------------------

def test_rigs_drill_one_well_at_a_time_and_move_on_within_a_week(corpus: ParsedCorpus) -> None:
    spans: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for well in corpus.eowrs:
        ddrs = corpus.ddrs_of(well)
        (rig,) = {d.rig for d in ddrs}
        spans[rig].append((ddrs[0].date, ddrs[-1].date, well))
    for rig, wells in spans.items():
        wells.sort()
        for (_, end, w1), (start, _, w2) in pairwise(wells):
            gap = (date.fromisoformat(start) - date.fromisoformat(end)).days
            assert 1 <= gap <= 7, (rig, w1, w2, gap)


def test_manifest_hash_is_deterministic(tmp_path: Path) -> None:
    hashes = []
    for name in ("a", "b"):
        written = write_corpus(build_corpus(seed=SEED), tmp_path / name)
        hashes.append(manifest_hash(tmp_path / name, written))
        assert manifest_hash(tmp_path / name) == hashes[-1]
    assert hashes[0] == hashes[1]
    assert (tmp_path / "a" / "_truth.json").read_bytes() == (tmp_path / "b" / "_truth.json").read_bytes()
    write_corpus(build_corpus(seed=7), tmp_path / "c")
    assert manifest_hash(tmp_path / "c") != hashes[0]


def test_manifest_hash_does_not_depend_on_the_process(tmp_path: Path) -> None:
    """Two separate processes with different string-hash seeds write the same corpus."""
    hashes = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-m", "wellbrief", "corpus", "generate",
             "--out", str(tmp_path / hash_seed), "--json"],
            check=True, capture_output=True, text=True, env=env, cwd=tmp_path,
        )
        hashes.append(json.loads(out.stdout)["manifest_hash"])
    assert hashes[0] == hashes[1] == manifest_hash(tmp_path / "1")


def test_regenerating_into_the_same_directory_replaces_the_corpus(tmp_path: Path) -> None:
    write_corpus(build_corpus(seed=7), tmp_path / "same")
    write_corpus(build_corpus(seed=SEED), tmp_path / "same")
    write_corpus(build_corpus(seed=SEED), tmp_path / "fresh")
    assert manifest_hash(tmp_path / "same") == manifest_hash(tmp_path / "fresh")


@pytest.mark.parametrize("names", [
    ["DDR-ABC-12-01.txt"],                       # well files of someone else's field
    ["DDR-ORD-101-001.txt"],                     # a generated name, but no sidecar: not an earlier corpus
    ["EOWR-XYZ-900.pdf", "notes.txt"],
    ["_truth.json", "DDR-ORD-101-001.txt", "DDR-ABC-12-01.txt"],   # an earlier corpus plus a foreign file
])
def test_a_directory_that_is_not_an_earlier_corpus_is_refused_and_left_alone(tmp_path: Path,
                                                                             names: list[str]) -> None:
    for name in names:
        (tmp_path / name).write_text("keep me\n", encoding="utf-8")
    with pytest.raises(OutputDirError):
        write_corpus(build_corpus(seed=SEED), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)
    assert all((tmp_path / name).read_text(encoding="utf-8") == "keep me\n" for name in names)


def test_scale_multiplies_the_wells_and_keeps_names_and_dates_valid() -> None:
    corpus = build_corpus(seed=SEED, scale=2)
    per_field = Counter(w.spec.name for w in corpus.wells)
    assert per_field == {"Orrindale": 56, "Vessra South": 28}
    assert all(WELL_ID.match(w.name) for w in corpus.wells)
    rigs = {w.rig for w in corpus.wells}
    assert all(re.fullmatch(r"[A-Za-z]+-\d", r) for r in rigs)
    assert not any(WELL_ID.match(r) for r in rigs)
    assert {w.rig for w in corpus.wells for e in w.entries if e.pattern == "G4"} == {"Vessra-3"}
    spans: dict[str, list[tuple[date, date]]] = defaultdict(list)
    for w in corpus.wells:
        spans[w.rig].append((w.spud, w.spud + timedelta(days=len(w.days) - 1)))
    for rig, intervals in spans.items():
        intervals.sort()
        assert all(a[1] < b[0] for a, b in pairwise(intervals)), rig


def test_the_largest_scale_stays_in_the_past() -> None:
    corpus = build_corpus(seed=SEED, scale=10)
    assert Counter(w.spec.name for w in corpus.wells) == {"Orrindale": 280, "Vessra South": 140}
    assert max(w.eowr_date for w in corpus.wells) < date(2026, 6, 30)


def test_scale_is_bounded() -> None:
    for scale in (0, 11):
        with pytest.raises(ValueError):
            build_corpus(scale=scale)


def test_random_streams_are_keyed_on_the_field_prefix_never_the_name(monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[object] = []
    original = random.Random

    class Recording(original):  # type: ignore[valid-type, misc]
        def __init__(self, x: object = None) -> None:
            keys.append(x)
            super().__init__(x)

    monkeypatch.setattr(generator_module.random, "Random", Recording)
    build_corpus(seed=SEED)
    assert keys
    pattern = re.compile(r"^\d+:(ORD|VSS):(\d+(:text|:incident)?|mwd)$")
    assert all(isinstance(k, str) and pattern.fullmatch(k) for k in keys), keys[:5]
