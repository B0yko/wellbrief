"""A hand-built workspace for planner and answer tests: two fields, six wells, a known NPT ledger.

The ledger is small enough to total by hand; the tests state the expected
figures as numbers and check them against their own SQL as well. ORD-104 has
daily reports and an end-of-well report but no NPT entry.
"""

from __future__ import annotations

from pathlib import Path

from wellbrief.embed import get_embedder
from wellbrief.models import Document, NptEvent, Well
from wellbrief.search import Searcher
from wellbrief.store import Store, build_indexes

SECTIONS = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']
FORMATIONS = {
    "Orrindale": ["Hesk Overburden", "Keldra Salt", "Dovrin Shale", "Orrindale Sand"],
    "Vessra South": ["Tessivar Marl", "Keldra Salt", "Ulvent Claystone", "Vessra Carbonate"],
}
WELLS = {"ORD-101": "Orrindale", "ORD-102": "Orrindale", "ORD-103": "Orrindale", "ORD-104": "Orrindale",
         "VSS-201": "Vessra South", "VSS-202": "Vessra South"}

# The NPT-free reports of ORD-104: (doc_id, depth at start, depth at end, operations line). Their
# metadata carries the drilled interval, as ingest parses it from "Depth at start" / "Depth at end".
QUIET_DDRS = [
    ("DDR-ORD-104-003", 1200.0, 1450.0,
     'Drilled 17 1/2" hole through Keldra Salt at 1.45 sg; hole held gauge.'),
    ("DDR-ORD-104-011", 2950.0, 3080.0, 'Drilled 8 1/2" hole from 2,950 m to 3,080 m with full returns.'),
]

# (doc_id, code, hours, section, formation, depth_m, description)
EVENTS = [
    ("DDR-ORD-101-005", "STUCK_PIPE", 23.5, '17 1/2"', "Keldra Salt", 1402,
     "String packed off at 1,402 m while pulling out of hole. Worked pipe free."),
    ("DDR-ORD-101-006", "STUCK_PIPE", 6.0, '17 1/2"', "Keldra Salt", 1402,
     "Continued to work the stuck string and jarred free."),
    ("DDR-ORD-101-006", "WAIT_ON_WEATHER", 4.0, '17 1/2"', "Keldra Salt", 1402,
     "Storm warning, operations suspended."),
    ("DDR-ORD-101-010", "DOWNHOLE_TOOL_FAILURE", 12.0, '12 1/4"', "Dovrin Shale", 2400,
     "MWD pulses lost; tripped to change the tool."),
    ("DDR-ORD-102-004", "STUCK_PIPE", 10.2, '17 1/2"', "Keldra Salt", 1390,
     "Pack-off on the trip out in the salt; worked free after circulating."),
    ("DDR-ORD-102-005", "WELLBORE_INSTABILITY", 3.1, '17 1/2"', "Keldra Salt", 1380,
     "Tight hole across the salt; reamed the interval."),
    ("DDR-ORD-102-006", "STUCK_PIPE", 0.8, '17 1/2"', "Keldra Salt", 1385,
     "Brief overpull and sticking at the connection."),
    ("DDR-ORD-102-009", "CEMENT_ISSUE", 5.5, '12 1/4"', "Dovrin Shale", 1850,
     "Cement head leaked; repaired before displacing."),
    ("DDR-ORD-103-004", "STUCK_PIPE", 2.5, '17 1/2"', "Keldra Salt", 1395,
     "Differential sticking at the connection."),
    ("DDR-ORD-103-007", "STUCK_PIPE", 1.5, '12 1/4"', "Dovrin Shale", 2100,
     "Stuck briefly on the connection, jarred free."),
    ("DDR-ORD-103-008", "STUCK_PIPE", 4.0, '26"', "Hesk Overburden", 300,
     "Stuck in unconsolidated sand, worked free."),
    ("DDR-VSS-201-012", "LOST_CIRCULATION", 18.0, '8 1/2"', "Vessra Carbonate", 2660,
     "Total losses on entering the carbonate at 2,660 m."),
    ("DDR-VSS-201-013", "RIG_REPAIR", 7.0, '8 1/2"', "Vessra Carbonate", 2700,
     "Mud pump fluid end washed out; changed the module."),
    ("DDR-VSS-202-011", "LOST_CIRCULATION", 9.0, '8 1/2"', "Vessra Carbonate", 2655,
     "Partial losses below the carbonate top."),
    ("DDR-VSS-202-003", "WAIT_ON_WEATHER", 6.0, '26"', "Tessivar Marl", 300,
     "Waiting on weather, high winds."),
]


def _ddr(doc_id: str, well: str, field: str, entries: list[tuple[str, float, str]],
         operations: str = "Drilled ahead.") -> str:
    lines = [
        "DAILY DRILLING REPORT",
        f"Operator: Quillfen Energy    Field: {field}    Well: {well}",
        f"Rig: Orrin-1    Report No: {doc_id[-3:]}    Date: 2024-01-{int(doc_id[-2:]):02d}",
        "",
        "OPERATIONS SUMMARY (24 h)",
    ]
    lines += [f"  06:00-10:00  NPT {code} {hours} h: {text}" for code, hours, text in entries]
    lines += [f"  10:00-06:00  {operations}", "", "NPT DETAIL"]
    for code, hours, text in entries:
        lines += [f"  Code                  : {code}", f"  Hours                 : {hours}",
                  f"  Description           : {text}", ""]
    return "\n".join(lines)


def documents() -> list[Document]:
    by_doc: dict[str, list[tuple[str, float, str]]] = {}
    for doc_id, code, hours, _section, _formation, _depth, text in EVENTS:
        by_doc.setdefault(doc_id, []).append((code, hours, text))
    # A report without NPT on every well.
    for well in WELLS:
        by_doc.setdefault(f"DDR-{well}-001", [])
    docs = []
    for doc_id, entries in sorted(by_doc.items()):
        well = doc_id[4:11]
        field = WELLS[well]
        docs.append(Document(doc_id, "ddr", well, field, "2024-01-01", "DAILY DRILLING REPORT",
                             _ddr(doc_id, well, field, entries)))
    for doc_id, start, end, operations in QUIET_DDRS:
        well = doc_id[4:11]
        docs.append(Document(doc_id, "ddr", well, WELLS[well], "2024-01-01", "DAILY DRILLING REPORT",
                             _ddr(doc_id, well, WELLS[well], [], operations),
                             meta={"depth_start_m": start, "depth_end_m": end}))
    for well, field in WELLS.items():
        docs.append(Document(
            f"EOWR-{well}", "eowr", well, field, "2024-03-01", "END OF WELL REPORT",
            f"END OF WELL REPORT\nOperator: Quillfen Energy    Field: {field}    Well: {well}\n\n"
            "4. LESSONS LEARNED\n  1. Hold the mud weight at 1.40 sg or more through Keldra Salt.\n"))
    docs.append(Document(
        "INC-ORD-101-01", "incident", "ORD-101", "Orrindale", "2024-01-05", "INCIDENT REPORT",
        "INCIDENT REPORT\nClassification: STUCK_PIPE   Lost time: 29.5 h\nPack-off in Keldra Salt.",
        meta={"code": "STUCK_PIPE", "hole_section": '17 1/2"', "formation": "Keldra Salt",
              "depth_m": 1402.0}))
    docs.append(Document(
        "INC-VSS-201-01", "incident", "VSS-201", "Vessra South", "2024-02-12", "INCIDENT REPORT",
        "INCIDENT REPORT\nClassification: LOST_CIRCULATION   Lost time: 18.0 h\n"
        "Total losses in the carbonate.",
        meta={"code": "LOST_CIRCULATION", "hole_section": '8 1/2"', "formation": "Vessra Carbonate",
              "depth_m": 2660.0}))
    return docs


def events() -> list[NptEvent]:
    return [
        NptEvent(doc_id, doc_id[4:11], WELLS[doc_id[4:11]], "2024-01-01", code, hours, section, formation,
                 float(depth), 1.30, "Orrin-1", text)
        for doc_id, code, hours, section, formation, depth, text in EVENTS
    ]


def store(tmp_path: Path) -> Store:
    """The store alone: enough for the planner."""
    s = Store(tmp_path / "wellbrief.db")
    s.put_documents(documents())
    s.put_wells(Well(w, f, "Orrin-1", "2024-01-01", 3000.0, SECTIONS, FORMATIONS[f])
                for w, f in WELLS.items())
    s.put_npt(events())
    return s


def searcher(s: Store) -> Searcher:
    """Indexes built next to the store (inside the test's tmp dir) and a searcher over them."""
    bm25, vectors = build_indexes(s)
    return Searcher(s, bm25, vectors, get_embedder("offline"))
