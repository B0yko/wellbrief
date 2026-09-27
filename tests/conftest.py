from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from corpus_files import ParsedCorpus, generated
from wellbrief import netguard
from wellbrief.corpus import SEED

SEEDS = (SEED, 7, 42)


@pytest.fixture(autouse=True)
def _uninstalled_netguard_after_every_test() -> Iterator[None]:
    """`cli.main()` installs the process-level network guard on every call (see
    `cli.py`), and it stays installed until something removes it: nothing in the CLI
    itself uninstalls it, since the guard is meant to protect the whole process for as
    long as it runs. A test that calls `cli.main()` would otherwise leave the guard
    installed for every test that runs after it in the same session, so this fixture
    removes it once the test is done, regardless of whether the test installed it."""
    yield
    netguard.uninstall()


@pytest.fixture(scope="session")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("corpora")


@pytest.fixture(scope="session", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def corpus(request: pytest.FixtureRequest, corpus_root: Path) -> ParsedCorpus:
    """A corpus written to disk and read back from its files, for each seed."""
    seed = request.param
    return generated(str(corpus_root / f"seed-{seed}"), seed)


# ---------------------------------------------------------------------------
# Sample reports for the reader/writer round-trip tests (readers, pdfwriter, docxwriter)
# ---------------------------------------------------------------------------
# Three realistic synthetic reports following the canonical templates of the corpus
# generator: a daily drilling report with a header split by runs of two or more spaces,
# aligned label lines, hole sizes with an inch mark and an NPT DETAIL block; an
# end-of-well report with numbered lessons and "- " recommendations; and an incident
# report. All names are fictional.

FOOTER = "Synthetic demonstration document. Operator, fields, wells, rigs and vendors are fictional."

DDR_LINES: list[str] = [
    "DAILY DRILLING REPORT",
    "Operator: Quillfen Energy    Field: Orrindale    Well: ORD-105",
    "Rig: Orrin-1    Report No: 014    Date: 2026-03-18",
    "Report period: 06:00 - 06:00",
    "",
    "DEPTH",
    "  Depth at start        : 1,318 m MD",
    "  Depth at end          : 1,402 m MD",
    "  Progress              : 84 m",
    '  Hole section          : 17 1/2"',
    "  Formation at TD       : Keldra Salt (halite)",
    "",
    "MUD",
    "  System                : Saturated salt polymer",
    "  Weight                : 1.34 sg",
    "  PV / YP               : 24 cP / 18 lb/100ft2",
    "  ECD at shoe           : 1.42 sg",
    "",
    "BHA AND PARAMETERS",
    '  Bit                   : 17 1/2" PDC, IADC M323',
    "  MWD                   : Parvane Downhole PJ-3",
    "  WOB / RPM / Flow      : 14 t / 120 rpm / 850 gpm",
    "  BHT max circulating   : 71 C",
    "",
    "OPERATIONS SUMMARY (24 h)",
    '  06:00-09:30  Drilled 17 1/2" hole from 1,318 m to 1,346 m MD.',
    "  09:30-10:00  Circulated bottoms up; shakers clean.",
    '  10:00-13:00  Drilled 17 1/2" hole from 1,346 m to 1,371 m MD.',
    "  13:00-14:30  NPT WELLBORE_INSTABILITY 1.5 h: Tight hole at 1,360 m; reamed down twice.",
    '  14:30-18:00  Drilled 17 1/2" hole from 1,371 m to 1,402 m MD (section TD).',
    "  18:00-19:30  Pumped 50 bbl hi-vis sweep and circulated hole clean.",
    "  19:30-22:00  Pulled out of hole to 1,340 m; overpull 25 t at 1,352 m.",
    "  22:00-05:30  NPT STUCK_PIPE 7.5 h: String packed off at 1,336 m while pulling out of hole.",
    "  05:30-06:00  Pulled out of hole to 1,212 m.",
    "",
    "TIME BREAKDOWN",
    "  Productive time       : 15.0 h",
    "  Non-productive time   : 9.0 h",
    "",
    "NPT DETAIL",
    "  Code                  : WELLBORE_INSTABILITY",
    "  Hours                 : 1.5",
    "  Depth                 : 1,360 m MD",
    "  Formation             : Keldra Salt",
    "  Description           : Tight hole at 1,360 m; reamed down twice before drilling ahead.",
    "",
    "  Code                  : STUCK_PIPE",
    "  Hours                 : 7.5",
    "  Depth                 : 1,336 m MD",
    "  Formation             : Keldra Salt",
    "  Description           : String packed off at 1,336 m while pulling out of hole; worked free.",
    "",
    "HSE",
    "  Incidents             : None",
    "  Drills                : Kick drill (crew B), 2 min 40 s",
    "  Personnel on board    : 86",
    "",
    "WEATHER",
    "  Wind                  : 12 kn NW    Visibility: 8 km",
    "  Temperature           : 4 C",
    "",
    "REMARKS",
    '  Casing programme for this section: 13 3/8" casing to 1,402 m.',
    '  Next 24 h: run 13 3/8" casing and cement.',
    "  Salt saturation checked every 2 h (chlorides 188,000 mg/l).",
    "",
    FOOTER,
]

EOWR_LINES: list[str] = [
    "END OF WELL REPORT",
    "Operator: Quillfen Energy    Field: Orrindale    Well: ORD-112",
    "Rig: Orrin-2    Spud: 2026-05-02    Days on well: 38",
    "Total depth: 3,120 m MD",
    "",
    "1. WELL SUMMARY",
    "  ORD-112 was drilled as a development well on Orrindale to 3,120 m MD.",
    '  Four hole sections were drilled: 26", 17 1/2", 12 1/4", 8 1/2".',
    "  Mud system throughout: Saturated salt polymer.",
    "",
    "2. TIME ANALYSIS",
    "  Total time on well    : 912.0 h",
    "  Non-productive time   : 41.5 h (4.6 %)",
    "",
    "3. NPT BREAKDOWN BY CODE",
    "  RIG_REPAIR                  18.0 h",
    "  DOWNHOLE_TOOL_FAILURE       14.5 h",
    "  WEATHER                      9.0 h",
    "",
    "4. LESSONS LEARNED",
    (
        "  1. Keldra Salt was drilled at 1.45 sg with a saturated brine sweep every 250 m. Hole held gauge "
        'and the trip out for the 13 3/8" casing was uneventful.'
    ),
    (
        '  2. Parvane Downhole PJ-3 directional tools were run in the 12 1/4" section where BHT reaches '
        "124 C; one tool failed at 2,410 m."
    ),
    "  3. Cementing and casing running went to programme with no notable deviation.",
    "",
    "5. RECOMMENDATIONS FOR FUTURE WELLS",
    "  - Hold at least 1.45 sg across Keldra Salt and sweep with saturated brine every 250 m.",
    "  - Run a caliper on the intermediate section before running casing.",
    "  - Confirm MWD temperature rating against the offset BHT profile at the planning stage.",
    "",
    FOOTER,
]

INCIDENT_LINES: list[str] = [
    "WELL OPERATIONS INCIDENT REPORT",
    "Operator: Quillfen Energy    Field: Vessra South    Well: VSS-207",
    "Rig: Vessra-3    Date: 2026-04-11    Severity: High",
    "Classification: LOST_CIRCULATION    Lost time: 31.5 h",
    "",
    "1. LOCATION",
    '  Depth: 2,662 m MD    Hole section: 8 1/2"    Formation: Vessra Carbonate',
    "  Bottom hole temperature: 109 C",
    "",
    "2. SEQUENCE OF EVENTS",
    "  Total losses were taken on entering Vessra Carbonate at 2,662 m (static loss rate > 60 m3/h).",
    "  Rig manager & company representative notified at 02:15.",
    "  Operations were suspended for 31.5 h. No injuries and no environmental release.",
    "",
    "3. IMMEDIATE CAUSE",
    "  The carbonate top was drilled at 780 gpm without a pre-emptive LCM pill.",
    "",
    "4. ROOT CAUSE",
    "  The carbonate top was entered at full flow rate without hole conditioning.",
    "",
    "5. CORRECTIVE ACTIONS",
    "  - Spot a 30 bbl fibrous LCM pill 20 m above the Vessra Carbonate top before drilling in.",
    "  - Reduce flow rate to below 450 gpm for the first 60 m of carbonate.",
    '  - Keep 200 bbl of pre-mixed LCM on location while drilling the 8 1/2" section.',
    "",
    FOOTER,
]

SAMPLES: dict[str, list[str]] = {"ddr": DDR_LINES, "eowr": EOWR_LINES, "incident": INCIDENT_LINES}


@pytest.fixture(params=sorted(SAMPLES))
def sample_lines(request: pytest.FixtureRequest) -> list[str]:
    """Return each of the three sample reports in turn."""
    lines: list[str] = SAMPLES[request.param]
    return lines


@pytest.fixture(params=sorted(SAMPLES))
def named_sample(request: pytest.FixtureRequest) -> tuple[str, list[str]]:
    """Return ``(name, lines)`` for each of the three sample reports in turn."""
    name: str = request.param
    return name, SAMPLES[name]


@pytest.fixture
def ddr_lines() -> list[str]:
    """Return the daily drilling report sample (longer than one PDF page)."""
    return list(DDR_LINES)


@pytest.fixture
def eowr_lines() -> list[str]:
    """Return the end-of-well report sample (its longest lines need a widened PDF page)."""
    return list(EOWR_LINES)
