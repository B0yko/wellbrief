"""Built-in defaults: NPT taxonomy, cost model, retrieval knobs, parse labels and
headings, detection headings, and CSV ledger columns.

Everything in this module is a *default*: the value a workspace gets when its
own `wellbrief.toml`, an `ingest --config` file, an environment variable or a
CLI flag does not say otherwise. `settings.py` is the one place that reads
those layers and resolves the precedence between them (CLI flags >
environment > `ingest --config` / the ingested folder's `wellbrief.toml` >
`<workspace>/wellbrief.toml` > the defaults below); every other module reads
a `settings.Settings` (or one of its parts, passed down by its caller) rather
than importing a constant from here directly, except where a value genuinely
never varies by site (the depth-band width used to *rank* a question's depth
preference, not to configure a report template, for example).
"""

from __future__ import annotations

import os
from typing import Any

# --------------------------------------------------------------------------
# NPT taxonomy
# --------------------------------------------------------------------------
# The codes are a simplified taxonomy modelled on common industry practice,
# not an official code list. `avoidable` marks the classes an engineering
# change can realistically prevent; weather and third party standby cannot be
# engineered away, so they are kept out of the avoidable totals on purpose.

NPT_CODES: dict[str, dict[str, Any]] = {
    "STUCK_PIPE": {"label": "Stuck pipe / pack-off", "avoidable": True, "family": "hole"},
    "LOST_CIRCULATION": {"label": "Lost circulation", "avoidable": True, "family": "hole"},
    "WELLBORE_INSTABILITY": {"label": "Wellbore instability", "avoidable": True, "family": "hole"},
    "HOLE_CLEANING": {"label": "Hole cleaning / reaming", "avoidable": True, "family": "hole"},
    "FISHING": {"label": "Fishing", "avoidable": True, "family": "hole"},
    "DOWNHOLE_TOOL_FAILURE": {"label": "Downhole tool failure", "avoidable": True, "family": "equipment"},
    "RIG_REPAIR": {"label": "Rig equipment repair", "avoidable": True, "family": "equipment"},
    "BOP_TEST_FAILURE": {"label": "BOP test failure", "avoidable": True, "family": "equipment"},
    "CEMENT_ISSUE": {"label": "Cementing problem", "avoidable": True, "family": "well_construction"},
    "WAIT_ON_MATERIALS": {"label": "Waiting on materials", "avoidable": True, "family": "logistics"},
    "WAIT_ON_WEATHER": {"label": "Waiting on weather", "avoidable": False, "family": "external"},
    "THIRD_PARTY_STANDBY": {"label": "Third party standby", "avoidable": False, "family": "external"},
    "HSE_STOP": {"label": "HSE stop work", "avoidable": False, "family": "external"},
}

AVOIDABLE_CODES = {c for c, v in NPT_CODES.items() if v["avoidable"]}

# Phrases in a question that name an NPT code. The query planner matches them
# on word boundaries, longer phrases first; the words of a phrase may be joined
# by spaces, hyphens or underscores or written together ("pack-off", "packoff"),
# and a trailing plural "s"/"es" is allowed. The code name itself ("stuck_pipe")
# always matches its code. There is deliberately no bare "plug" (a "plugged"
# nozzle is not a cementing problem) and no bare "lost" ("time was lost to
# weather" is not lost circulation).
CODE_SYNONYMS: dict[str, list[str]] = {
    "STUCK_PIPE": ["stuck pipe", "stuck", "pack off", "packed off", "packing off", "differential sticking"],
    "LOST_CIRCULATION": ["lost circulation", "losses", "total losses", "partial losses", "lost returns",
                         "loss of returns", "loss of circulation"],
    "WELLBORE_INSTABILITY": ["wellbore instability", "instability", "tight hole", "washout", "caving",
                             "overpull"],
    "HOLE_CLEANING": ["hole cleaning", "cuttings bed", "back ream", "reaming"],
    "FISHING": ["fishing", "fish", "junk in hole"],
    "DOWNHOLE_TOOL_FAILURE": ["mwd", "mwd failure", "lwd", "tool failure", "downhole tool",
                              "directional tool"],
    "RIG_REPAIR": ["rig repair", "mud pump", "fluid end", "drawworks", "top drive", "equipment failure"],
    "BOP_TEST_FAILURE": ["bop", "blowout preventer", "preventer", "pressure test"],
    "CEMENT_ISSUE": ["cement", "cementing", "cement job", "bumped plug"],
    "WAIT_ON_MATERIALS": ["waiting on materials", "wait on materials", "wom", "barite", "logistics"],
    "WAIT_ON_WEATHER": ["weather", "waiting on weather", "wow", "storm", "wind"],
    "THIRD_PARTY_STANDBY": ["standby", "third party", "wireline standby"],
    "HSE_STOP": ["hse", "stop work", "safety stand down"],
}

HOLE_SECTIONS = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']

# Words that put a candidate mitigation sentence in the same subject as an
# NPT code (the mitigation miner's relevance test; see `miner._relevant`). Matched on
# word boundaries against the sentence text alone; an interval risk also
# requires the sentence, or an incident's own recorded section and
# formation, to name the place in the well (`miner._relevant`). The default
# for `[taxonomy.keywords]`; a site's own table replaces a code's whole list,
# it does not add to this one.
TAXONOMY_KEYWORDS: dict[str, list[str]] = {
    "STUCK_PIPE": ["pack-off", "mud weight", "sg", "salt", "trip"],
    "FISHING": ["fish", "jar", "bha"],
    "WELLBORE_INSTABILITY": ["salt", "sg", "mud weight", "overpull", "tight hole"],
    "HOLE_CLEANING": ["cuttings", "sweep", "hole cleaning", "reaming", "annular velocity"],
    "LOST_CIRCULATION": ["losses", "lcm", "flow rate", "pill"],
    "DOWNHOLE_TOOL_FAILURE": ["mwd", "temperature rating", "temperature"],
    "RIG_REPAIR": ["mud pump", "fluid end"],
    "BOP_TEST_FAILURE": ["bop", "preventer", "pressure test"],
    "CEMENT_ISSUE": ["cement"],
    "WAIT_ON_MATERIALS": ["barite", "materials", "logistics"],
    "WAIT_ON_WEATHER": ["weather", "forecast"],
    "THIRD_PARTY_STANDBY": ["standby", "third party", "third-party"],
    "HSE_STOP": ["hse", "stop work", "lifting", "safety"],
}

# Every document the synthetic corpus generator writes ends with this line.
# The parsers end every section at it, so it is never read as an NPT
# description, a lesson, a recommendation or a corrective action.
SYNTHETIC_FOOTER = (
    "Synthetic demonstration document. Operator, fields, wells, rigs and vendors are fictional."
)

# A site's own spelling of an NPT code ("SP", "LC", "DH-TOOL") mapped to the taxonomy
# (`STUCK_PIPE`, `LOST_CIRCULATION`, `DOWNHOLE_TOOL_FAILURE`); matched case-insensitively
# once the code is folded the same way an unmapped one is (`ingest._normalise_code`:
# non-alphanumeric runs become a single underscore, upper-cased). The default for
# `[taxonomy.aliases]` is empty: a code that is not in `NPT_CODES` and not aliased becomes
# `OTHER`, the built-in fallback every alias table still falls back to.
DEFAULT_TAXONOMY_ALIASES: dict[str, str] = {}

# --------------------------------------------------------------------------
# Parse labels and section headings
# --------------------------------------------------------------------------
# The label text (before the colon) each parser looks for. A site that writes
# its daily reports, end-of-well reports or incident reports with different
# labels or section headings maps them with `[parse.ddr.labels]`,
# `[parse.eowr.sections]` and `[parse.incident.sections]`; these are the
# canonical template's own labels, used when a site's `wellbrief.toml` does
# not override them (or there is none).

# `[parse.ddr.labels]`: the header fields shared by every daily report, plus
# the five fields of one NPT DETAIL entry (`npt_*`). A canonical value here
# never collides with another: the header's own "Formation at TD" and an NPT
# entry's "Formation" are two different keys (`formation_at_td`, `npt_formation`)
# precisely so a site can rename one without the other.
DEFAULT_DDR_LABELS: dict[str, str] = {
    "depth_at_start": "Depth at start",
    "depth_at_end": "Depth at end",
    "progress": "Progress",
    "hole_section": "Hole section",
    "formation_at_td": "Formation at TD",
    "weight": "Weight",
    "ecd": "ECD",
    "bht": "Static BHT estimate",
    "mwd": "MWD",
    "productive_time": "Productive time",
    "non_productive_time": "Non-productive time",
    "npt_code": "Code",
    "npt_hours": "Hours",
    "npt_depth": "Depth",
    "npt_formation": "Formation",
    "npt_description": "Description",
}

# `[parse.eowr.sections]`: the heading each block starts at, written either with
# its canonical leading number ("3. NPT BREAKDOWN BY CODE") or bare ("NPT
# BREAKDOWN BY CODE", "LESSONS"): `parse.parse_eowr` accepts an optional
# "<number>. " prefix on any configured heading, so a site need not renumber
# its own template to match this one.
DEFAULT_EOWR_SECTIONS: dict[str, str] = {
    "npt_by_code": "NPT BREAKDOWN BY CODE",
    "lessons": "LESSONS LEARNED",
    "recommendations": "RECOMMENDATIONS FOR FUTURE WELLS",
}

# `[parse.incident.sections]`: same convention as `DEFAULT_EOWR_SECTIONS`.
DEFAULT_INCIDENT_SECTIONS: dict[str, str] = {
    "root_cause": "ROOT CAUSE",
    "corrective_actions": "CORRECTIVE ACTIONS",
}

# --------------------------------------------------------------------------
# Document-type detection
# --------------------------------------------------------------------------
# `[detect.headings]`: document type -> the exact first-lines-of-file heading
# (`detect.py` scans the first few lines) that names it; a site whose reports
# open with a different line maps it here. The filename prefix fallback
# (`DDR-`, `EOWR-`, `INC-`) is not configurable: it is a fallback for a
# heading that is missing or unrecognised, not a template convention.
DEFAULT_DETECT_HEADINGS: dict[str, str] = {
    "ddr": "DAILY DRILLING REPORT",
    "eowr": "END OF WELL REPORT",
    "incident": "WELL OPERATIONS INCIDENT REPORT",
}

# --------------------------------------------------------------------------
# CSV ledger columns
# --------------------------------------------------------------------------
# `[csv.columns]` (+ optional `date_format`): maps a logical ledger column
# (`readers.csvledger.LOGICAL_COLUMNS`) to the header a site's own NPT ledger
# CSV uses. Empty by default: `read_ledger` already looks up every unmapped
# logical column under its own name (`well`, `date`, `code`, `hours`, ...).
DEFAULT_CSV_COLUMNS: dict[str, str] = {}
DEFAULT_CSV_DATE_FORMAT: str | None = None

# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------
# Spread rate = rig day rate + all services on location, in USD per day.
# Hours convert to cost at spread rate / 24. The default is illustrative;
# pass --spread-rate to use a different figure.

DEFAULT_SPREAD_RATE_USD_PER_DAY = 48_000.0


def hours_to_usd(hours: float, spread_rate_per_day: float = DEFAULT_SPREAD_RATE_USD_PER_DAY) -> float:
    return hours / 24.0 * spread_rate_per_day


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
EMBED_DIM = 512
RRF_K = 60          # rank-fusion damping, standard value from the TREC work
DEFAULT_TOP_K = 8
# A depth in a question ("around 2,650 m") is read as this many metres either
# side. It ranks reports with NPT entries in that band higher; it only filters
# when the question says "at 2,650 m".
DEPTH_BAND_M = 100.0
# Reports cited in an answer's text behind each computed figure, largest
# contributors first. The JSON output lists every contributing report.
FIGURE_SOURCE_LIMIT = 5

# A risk has to show up on at least this share of comparable offset wells
# before it goes in the brief. Chosen on the default seed (20260731): the
# four planted interval patterns (the three stuck-pipe-family codes in the
# Keldra Salt hole section, plus lost circulation in the Vessra Carbonate
# hole section) sit well above this line on affected-well share; the field's
# background noise (BOP test and cement blips that also clear `min_lift`,
# since a code confined to one casing point always looks concentrated by the
# day) sits at or below it except for one corner case. 0.30 keeps every
# planted pattern and cuts all but that one background risk, which is what
# the brief-precision target (>= 0.75 of listed risks planted) is measured
# against.
RISK_MIN_SUPPORT = 0.3
RISK_MIN_WELLS = 2

# An interval pattern (code, hole section, formation) also has to run hotter
# than the code's field-wide rate: its NPT hours per drilling day must be at
# least this many times the field's hours per drilling day for the same code.
# "Drilling day" = one DDR. Corresponds to `[risk] min_lift` once TOML
# configuration lands.
RISK_MIN_LIFT = 2.0

# Equipment patterns (code, rig) or (code, MWD tool) qualify on three tests at
# once: hours per drilling day on the category vs. the rest of the same
# field's fleet (`equipment_min_ratio`), the share of the category's own wells
# that were affected (`equipment_min_rate`), and a minimum number of affected
# wells (`equipment_min_wells`), so a two-well fleet cannot become a pattern.
# A share-of-wells-affected rule alone misses a category that fails harder,
# not more often: on the reference corpus the rig behind the repeated mud pump
# failures has 7 of 7 wells affected against 4 of 7 on the other rig (ratio
# 1.75, below a 2x share threshold), but the hours-per-day ratio is about 12x.
EQUIPMENT_MIN_RATIO = 2.0
EQUIPMENT_MIN_RATE = 0.4
EQUIPMENT_MIN_WELLS = 3

# Default cap on the number of risks a brief lists, ranked by expected cost.
RISK_MAX_RISKS = 8

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# A document longer than CHUNK_THRESHOLD_CHARS is split into chunks of at
# most CHUNK_MAX_CHARS, breaking at a heading or a blank line where one falls
# inside the limit; a shorter document stays one chunk. Indexes are built
# over chunks (`workspace.FieldIndex`), and retrieval keeps the best chunk
# per document before ranking documents.
CHUNK_THRESHOLD_CHARS = 4_000
CHUNK_MAX_CHARS = 1_500

# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
# `offline` needs no network and no model download. It is the default and, in
# this version, the only backend for both embeddings and narration.

EMBED_BACKEND = os.environ.get("WELLBRIEF_EMBED_BACKEND", "offline")
LLM_BACKEND = os.environ.get("WELLBRIEF_LLM_BACKEND", "offline")
