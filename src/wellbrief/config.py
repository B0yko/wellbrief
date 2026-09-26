"""Static configuration: NPT taxonomy, cost model, retrieval knobs.

The values are illustrative defaults. Taxonomy, thresholds and retrieval
knobs are constants in this version; the spread rate is set with
--spread-rate and the data directory with WELLBRIEF_DATA_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("WELLBRIEF_DATA_DIR", Path.cwd() / "data"))
CORPUS_DIR = DATA_DIR / "corpus"
DB_PATH = DATA_DIR / "wellbrief.db"

# --------------------------------------------------------------------------
# NPT taxonomy
# --------------------------------------------------------------------------
# The codes are a simplified taxonomy modelled on common industry practice,
# not an official code list. `avoidable` marks the classes an engineering
# change can realistically prevent; weather and third party standby cannot be
# engineered away, so they are kept out of the avoidable totals on purpose.

NPT_CODES: dict[str, dict] = {
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

HOLE_SECTIONS = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']

# Every document the synthetic corpus generator writes ends with this line.
# The parsers end every section at it, so it is never read as an NPT
# description, a lesson, a recommendation or a corrective action.
SYNTHETIC_FOOTER = (
    "Synthetic demonstration document. Operator, fields, wells, rigs and vendors are fictional."
)

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

# A risk has to show up on at least this share of comparable offset wells
# before it goes in the brief. Set low enough to catch a repeat-twice problem
# on a 14-well field, high enough to keep one-off events out.
RISK_MIN_SUPPORT = 0.12
RISK_MIN_WELLS = 2

# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
# `offline` needs no network and no model download. It is the default and, in
# this version, the only backend for both embeddings and narration.

EMBED_BACKEND = os.environ.get("WELLBRIEF_EMBED_BACKEND", "offline")
LLM_BACKEND = os.environ.get("WELLBRIEF_LLM_BACKEND", "offline")
