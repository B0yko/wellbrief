"""Parser behaviour on hand-written reports.

Covers per-entry Depth and Formation in NPT DETAIL (with the fallback to the
report header when an entry has neither), continuation lines, and the
synthetic-document footer, which ends every section and is never content.
"""

from __future__ import annotations

from wellbrief import parse
from wellbrief.config import SYNTHETIC_FOOTER
from wellbrief.models import Document
from wellbrief.text import snippet

DDR_WITH_ENTRY_LOCATION = f"""DAILY DRILLING REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Rig: Orrin-2    Report No: 014    Date: 2023-05-02
Report period: 06:00 - 06:00

DEPTH
  Depth at start        : 1,610 m MD
  Depth at end          : 1,642 m MD
  Progress              : 32 m
  Hole section          : 17 1/2"
  Formation at TD       : Dovrin Shale (reactive shale)

MUD
  Weight                : 1.33 sg

TIME BREAKDOWN
  Productive time       : 4.4 h
  Non-productive time   : 19.6 h

NPT DETAIL
  Code                  : STUCK_PIPE
  Hours                 : 15.2
  Depth                 : 1,402 m MD
  Formation             : Keldra Salt (halite)
  Description           : String packed off at 1,402 m while pulling out of hole for the 13 3/8"
                          intermediate casing. Salt creep at 1.33 sg across Keldra Salt. Worked pipe
                          and jarred free after 15.2 h.

  Code                  : WAIT_ON_WEATHER
  Hours                 : 4.4
  Depth                 : 1,642 m MD
  Formation             : Dovrin Shale
  Description           : Storm warning, operations suspended.

REMARKS
  Next 24 h: continue 17 1/2" hole in Dovrin Shale.

{SYNTHETIC_FOOTER}
"""

# An older template: NPT entries without their own Depth and Formation lines.
DDR_WITHOUT_ENTRY_LOCATION = """DAILY DRILLING REPORT
Operator: Quillfen Energy    Field: Vessra South    Well: VSS-230
Rig: Vessra-5    Report No: 031    Date: 2023-08-19
Report period: 06:00 - 06:00

DEPTH
  Depth at start        : 2,640 m MD
  Depth at end          : 2,688 m MD
  Hole section          : 8 1/2"
  Formation at TD       : Vessra Carbonate (fractured and vuggy limestone)

NPT DETAIL
  Code                  : LOST_CIRCULATION
  Hours                 : 9.5
  Description           : Partial losses below the carbonate top. Pumped LCM sweeps until returns
                          were regained.

REMARKS
  Next 24 h: continue 8 1/2" hole.
"""

# The footer follows the last entry directly, with no blank line and no REMARKS.
DDR_FOOTER_AFTER_ENTRY = f"""DAILY DRILLING REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-151
Rig: Orrin-1    Report No: 003    Date: 2023-06-01

DEPTH
  Depth at end          : 700 m MD
  Hole section          : 17 1/2"
  Formation at TD       : Keldra Salt (halite with anhydrite stringers)

NPT DETAIL
  Code                  : HSE_STOP
  Hours                 : 1.5
  Depth                 : 690 m MD
  Formation             : Keldra Salt
  Description           : Stop-work called on rig floor, lifting plan reviewed before restart.
{SYNTHETIC_FOOTER}
"""

EOWR = f"""END OF WELL REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Rig: Orrin-2    Spud: 2023-04-19    Report date: 2023-06-02
Days on well: 31    Total depth: 3,120 m MD

2. TIME ANALYSIS
  Non-productive time   : 19.6 h (2.6 %)

3. NPT BREAKDOWN BY CODE
  STUCK_PIPE                  15.2 h
  WAIT_ON_WEATHER              4.4 h

4. LESSONS LEARNED
  1. Keldra Salt was drilled at 1.33 sg in the 17 1/2" section; the string packed off at 1,402 m on
     the trip out for the 13 3/8" intermediate casing.
  2. Cement jobs went to programme with no notable deviation.

5. RECOMMENDATIONS FOR FUTURE WELLS
  - Run a caliper on the intermediate section before running casing.
  - Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m, and pump
    out of hole through the salt.
{SYNTHETIC_FOOTER}
"""

INCIDENT = f"""WELL OPERATIONS INCIDENT REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Rig: Orrin-2    Date: 2023-05-02    Severity: Medium
Classification: STUCK_PIPE    Lost time: 15.2 h
Daily report: DDR-ORD-150-014

1. LOCATION
  Depth: 1,402 m MD    Hole section: 17 1/2"    Formation: Keldra Salt

4. ROOT CAUSE
  Mud weight below the value required to hold the salt in gauge.

5. CORRECTIVE ACTIONS
  - Raise mud weight to at least 1.42 sg before drilling into Keldra Salt.
  - Pump out of hole through the salt interval rather than pulling on elevators, and record the
    overpull at every stand.

{SYNTHETIC_FOOTER}
"""


def _ddr(text: str, doc_id: str = "DDR-TEST-001") -> Document:
    return Document(doc_id=doc_id, doc_type="ddr", well="", field_name="", date="", title="", text=text)


def test_npt_entry_depth_and_formation_are_read_from_the_entry() -> None:
    entries = parse.parse_npt_blocks(DDR_WITH_ENTRY_LOCATION)
    assert [(e["code"], e["hours"], e["depth_m"], e["formation"]) for e in entries] == [
        ("STUCK_PIPE", 15.2, 1402.0, "Keldra Salt"),
        ("WAIT_ON_WEATHER", 4.4, 1642.0, "Dovrin Shale"),
    ]


def test_ledger_rows_use_the_entry_location_not_the_report_header() -> None:
    rows = parse.npt_events(_ddr(DDR_WITH_ENTRY_LOCATION))
    assert [(r.code, r.depth_m, r.formation, r.hole_section) for r in rows] == [
        ("STUCK_PIPE", 1402.0, "Keldra Salt", '17 1/2"'),
        ("WAIT_ON_WEATHER", 1642.0, "Dovrin Shale", '17 1/2"'),
    ]
    assert rows[0].well == "ORD-150" and rows[0].field_name == "Orrindale" and rows[0].mud_weight_sg == 1.33


def test_ledger_rows_fall_back_to_the_header_when_the_entry_has_no_location() -> None:
    entries = parse.parse_npt_blocks(DDR_WITHOUT_ENTRY_LOCATION)
    assert entries[0]["depth_m"] is None and entries[0]["formation"] is None
    (row,) = parse.npt_events(_ddr(DDR_WITHOUT_ENTRY_LOCATION))
    assert (row.depth_m, row.formation, row.hole_section) == (2688.0, "Vessra Carbonate", '8 1/2"')


def test_a_zero_entry_depth_is_kept_rather_than_replaced_by_the_header() -> None:
    text = DDR_WITH_ENTRY_LOCATION.replace("1,642 m MD\n  Formation             : Dovrin", "0 m MD\n"
                                           "  Formation             : Dovrin")
    rows = parse.npt_events(_ddr(text))
    assert rows[1].depth_m == 0.0


def test_description_continuation_lines_are_joined() -> None:
    entries = parse.parse_npt_blocks(DDR_WITH_ENTRY_LOCATION)
    assert entries[0]["description"] == (
        "String packed off at 1,402 m while pulling out of hole for the 13 3/8\" intermediate casing. "
        "Salt creep at 1.33 sg across Keldra Salt. Worked pipe and jarred free after 15.2 h."
    )
    entries = parse.parse_npt_blocks(DDR_WITHOUT_ENTRY_LOCATION)
    assert entries[0]["description"].endswith("Pumped LCM sweeps until returns were regained.")


def test_npt_detail_ends_at_the_next_heading() -> None:
    entries = parse.parse_npt_blocks(DDR_WITHOUT_ENTRY_LOCATION)
    assert len(entries) == 1
    assert "Next 24 h" not in entries[0]["description"]


def test_footer_right_after_an_entry_is_not_part_of_the_description() -> None:
    (entry,) = parse.parse_npt_blocks(DDR_FOOTER_AFTER_ENTRY)
    assert entry["description"] == "Stop-work called on rig floor, lifting plan reviewed before restart."


def test_eowr_lessons_and_recommendations_join_continuation_lines() -> None:
    p = parse.parse_eowr(EOWR)
    assert p["lessons"] == [
        ("Keldra Salt was drilled at 1.33 sg in the 17 1/2\" section; the string packed off at 1,402 m on "
         "the trip out for the 13 3/8\" intermediate casing."),
        "Cement jobs went to programme with no notable deviation.",
    ]
    assert p["recommendations"] == [
        "Run a caliper on the intermediate section before running casing.",
        ("Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m, and pump "
         "out of hole through the salt."),
    ]
    assert p["date"] == "2023-06-02"
    assert p["npt_by_code"] == {"STUCK_PIPE": 15.2, "WAIT_ON_WEATHER": 4.4}


def test_footer_is_never_a_recommendation_or_a_corrective_action() -> None:
    recs = parse.parse_eowr(EOWR)["recommendations"]
    actions = parse.parse_incident(INCIDENT)["corrective_actions"]
    assert all("Synthetic demonstration document" not in s for s in recs + actions)
    assert recs[-1].endswith("out of hole through the salt.")


def test_incident_corrective_actions_join_continuation_lines() -> None:
    p = parse.parse_incident(INCIDENT)
    assert p["corrective_actions"] == [
        "Raise mud weight to at least 1.42 sg before drilling into Keldra Salt.",
        ("Pump out of hole through the salt interval rather than pulling on elevators, and record the "
         "overpull at every stand."),
    ]
    assert (p["code"], p["hours"], p["depth_m"], p["hole_section"], p["formation"]) == (
        "STUCK_PIPE", 15.2, 1402.0, '17 1/2"', "Keldra Salt")


def test_strip_footer_cuts_at_the_footer_line_only() -> None:
    assert parse.strip_footer(f"a\nb\n{SYNTHETIC_FOOTER}\nc\n") == "a\nb"
    assert parse.strip_footer("a\nno footer here\n") == "a\nno footer here\n"


def test_snippet_never_quotes_the_footer() -> None:
    quote = snippet(INCIDENT, ["operator", "fields", "wells", "rigs", "vendors", "fictional", "synthetic"])
    assert quote and "Synthetic demonstration document" not in quote


DDR_HEADER_WITHOUT_DRILLING = f"""DAILY DRILLING REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Rig: Orrin-2    Report No: 015    Date: 2023-05-03
Report period: 06:00 - 06:00

DEPTH
  Depth at start        : 1,850 m MD
  Depth at end          : 1,850 m MD
  Hole section          : 17 1/2"
  Formation at TD       : Dovrin Shale (reactive shale)

MUD
  System                : Salt-saturated polymer
  Weight                : 1.33 sg
  ECD                   : -

BHA AND PARAMETERS
  MWD                   : Parvane Downhole PJ-3
  WOB / RPM / Flow      : - / - / -
  Static BHT estimate   : 96 C

OPERATIONS SUMMARY (24 h)
  06:00-06:00  Waited on cement.

NPT DETAIL
  None reported this period.

{SYNTHETIC_FOOTER}
"""


def test_a_report_without_drilling_has_no_ecd_or_flow_and_reads_the_static_bht() -> None:
    p = parse.parse_ddr(DDR_HEADER_WITHOUT_DRILLING)
    assert p["ecd_sg"] is None and "flow_gpm" not in p
    assert p["bht_c"] == 96.0 and p["mud_weight_sg"] == 1.33
    assert p["npt"] == []


def test_incident_with_several_daily_reports_parses_its_header() -> None:
    text = INCIDENT.replace("Daily report: DDR-ORD-150-014",
                            "Daily reports: DDR-ORD-150-014, DDR-ORD-150-015")
    p = parse.parse_incident(text)
    assert (p["code"], p["hours"], p["depth_m"]) == ("STUCK_PIPE", 15.2, 1402.0)
