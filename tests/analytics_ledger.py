"""A hand-built ledger for the risk-statistics tests (`test_analytics.py`, `test_riskbrief.py`).

One field, two rigs, six wells, a section every well drills through (where
one rig runs hotter on RIG_REPAIR) and a second, quiet section that only
dilutes the field-wide drilling-day count. Every figure below is small enough
to check by hand; the tests restate them as plain arithmetic next to the
assertions instead of trusting this module.

Wells and rigs (3 a side):
    RB-101, RB-102, RB-103  ->  Rig-North
    RB-104, RB-105, RB-106  ->  Rig-South

Every well has 4 daily reports in `12 1/4" / Bluff Shale` (the section RIG_REPAIR
concentrates in) and 5 quiet daily reports in `17 1/2" / Fenmoor Marl` (no NPT),
so every well's own exposure is identical: 9 drilling days, of which 4 are in
the RIG_REPAIR section. Field-wide: 54 drilling days, 24 of them in that section.

RIG_REPAIR (all in `12 1/4" / Bluff Shale`):
    RB-101: one entry, 8 h        RB-104: one entry, 1 h
    RB-102: one entry, 8 h        RB-105: one entry, 1 h
    RB-103: one entry, 8 h        RB-106: none (clean on both counts)
  Rig-North: 3 events, 24 h, 3 of 3 wells affected (share 1.00).
  Rig-South: 2 events,  2 h, 2 of 3 wells affected (share 0.67): a
  wells-affected-share rule would call this only 1.5x Rig-North's rate,
  below a 2x threshold; the hours-per-drilling-day rule gives 12x (see
  `test_analytics.py`).
  Interval bucket (RIG_REPAIR, 12 1/4", Bluff Shale): all 5 events, 26 h,
  5 of 6 wells exposed to the section affected: it independently clears the
  interval thresholds too, and shares 3 of its 5 events (60 %) with the
  Rig-North equipment pattern, so event ownership merges it away.

A second code, WELLBORE_INSTABILITY, gives the lift test a pattern with *no*
equipment counterpart to merge into, in the same `12 1/4" / Bluff Shale`
section:
    RB-101: one entry, 6 h        RB-104: none
    RB-102: one entry, 6 h        RB-105: none
    RB-103: none                  RB-106: none
  2 of 6 wells affected, 12 h, days_here 24, field days 54: lift well above
  2.0 (see `test_analytics.py`), no rig or tool correlation to merge into.

WAIT_ON_WEATHER (unavoidable, avoidable = False): 3 h on RB-101, 2 h on
RB-104, both in the quiet section, so it never has a section/formation
overlap to interfere with the codes above; it is the "unavoidable
background" line.

Depths: the RIG_REPAIR/WELLBORE_INSTABILITY events are all logged at 2,600 m,
comfortably above a planned TD of 2,000 m (used by the TD-filter test) and
below one of 3,000 m.
"""

from __future__ import annotations

from pathlib import Path

from wellbrief.models import Document, NptEvent, Well
from wellbrief.store import Store

FIELD = "Riverbend"
RIG_NORTH = "Rig-North"
RIG_SOUTH = "Rig-South"
HOT_SECTION = '12 1/4"'
HOT_FORMATION = "Bluff Shale"
QUIET_SECTION = '17 1/2"'
QUIET_FORMATION = "Fenmoor Marl"
DEPTH_M = 2600.0

WELLS = {
    "RB-101": RIG_NORTH, "RB-102": RIG_NORTH, "RB-103": RIG_NORTH,
    "RB-104": RIG_SOUTH, "RB-105": RIG_SOUTH, "RB-106": RIG_SOUTH,
}

# (well, code, hours)
RIG_REPAIR_EVENTS = [
    ("RB-101", "RIG_REPAIR", 8.0), ("RB-102", "RIG_REPAIR", 8.0), ("RB-103", "RIG_REPAIR", 8.0),
    ("RB-104", "RIG_REPAIR", 1.0), ("RB-105", "RIG_REPAIR", 1.0),
]
WELLBORE_INSTABILITY_EVENTS = [
    ("RB-101", "WELLBORE_INSTABILITY", 6.0), ("RB-102", "WELLBORE_INSTABILITY", 6.0),
]
UNAVOIDABLE_EVENTS = [
    ("RB-101", "WAIT_ON_WEATHER", 3.0), ("RB-104", "WAIT_ON_WEATHER", 2.0),
]

HOT_DDRS_PER_WELL = 4
QUIET_DDRS_PER_WELL = 5


def _ddr_doc(doc_id: str, well: str, section: str, formation: str) -> Document:
    return Document(
        doc_id=doc_id, doc_type="ddr", well=well, field_name=FIELD, date="2024-01-01",
        title="DAILY DRILLING REPORT", text="DAILY DRILLING REPORT\n",
        meta={"hole_section": section, "formation": formation, "rig": WELLS[well]},
    )


def documents() -> list[Document]:
    docs: list[Document] = []
    for well in WELLS:
        for i in range(HOT_DDRS_PER_WELL):
            docs.append(_ddr_doc(f"DDR-{well}-H{i:02d}", well, HOT_SECTION, HOT_FORMATION))
        for i in range(QUIET_DDRS_PER_WELL):
            docs.append(_ddr_doc(f"DDR-{well}-Q{i:02d}", well, QUIET_SECTION, QUIET_FORMATION))
    return docs


def events() -> list[NptEvent]:
    out = []
    for well, code, hours in [*RIG_REPAIR_EVENTS, *WELLBORE_INSTABILITY_EVENTS, *UNAVOIDABLE_EVENTS]:
        section = QUIET_SECTION if code == "WAIT_ON_WEATHER" else HOT_SECTION
        formation = QUIET_FORMATION if code == "WAIT_ON_WEATHER" else HOT_FORMATION
        doc_id = f"DDR-{well}-{'Q00' if code == 'WAIT_ON_WEATHER' else 'H00'}"
        out.append(NptEvent(
            doc_id=doc_id, well=well, field_name=FIELD, date="2024-01-01", code=code, hours=hours,
            hole_section=section, formation=formation, depth_m=DEPTH_M, mud_weight_sg=1.30,
            rig=WELLS[well], description=f"{code} at {well}.",
        ))
    return out


def wells() -> list[Well]:
    return [
        Well(name=w, field_name=FIELD, rig=rig, spud_date="2024-01-01", td_m=3500.0,
            sections=[QUIET_SECTION, HOT_SECTION], formations=[QUIET_FORMATION, HOT_FORMATION])
        for w, rig in WELLS.items()
    ]


def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "wellbrief.db")
    s.put_documents(documents())
    s.put_wells(wells())
    s.put_npt(events())
    return s
