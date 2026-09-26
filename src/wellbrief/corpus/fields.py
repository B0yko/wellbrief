"""Geometry, naming, mud programme and campaign schedule of the two synthetic fields.

Every name here is invented: the operator, both fields, their wells, rigs,
formations and the MWD vendor. Depths are measured depths in metres; the
formation and hole-section intervals are half-open, `[top, base)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

OPERATOR = "Quillfen Energy"
MWD_VENDOR = "Parvane Downhole"
MWD_TOOLS = ("PJ-3", "PJ-5")

# The first well of each field spuds on FIRST_SPUD plus the field's offset;
# the first well of every other rig follows FIRST_WELL_STAGGER_DAYS later.
# After that a rig spuds its next well a few days after it releases the
# previous one (the rig move), so no rig drills two wells at once and none
# sits idle between wells.
FIRST_SPUD = date(2022, 1, 17)
FIRST_WELL_STAGGER_DAYS = 20
RIG_MOVE_DAYS = (3, 7)

# Rig names keep a single trailing digit, so the query planner's well-id
# pattern (letters, a dash, two to four digits) never reads a rig as a well.
MAX_RIGS_PER_FIELD = 9

SPUD_MUD = "Spud mud (bentonite)"
SALT_MUD = "Salt-saturated polymer"
KCL_MUD = "KCl-polymer"


@dataclass(frozen=True)
class Formation:
    name: str
    top_m: int
    base_m: int
    rop_m_per_day: float
    lithology: str


@dataclass(frozen=True)
class SectionPlan:
    """One hole section of the well design.

    `mw_range_sg` is the mud weight the section is drilled with (drawn once
    per well); in 17 1/2" it is the weight above Keldra Salt, and the mud is
    weighted up at the salt top. `fit_range_sg` is the formation integrity
    test run below the previous casing shoe, and `casing_speed_m_per_h` the
    running speed of the section's casing string.
    """

    size: str
    top_m: int
    base_m: int
    casing: str
    mud_system: str
    mw_range_sg: tuple[float, float]
    fit_range_sg: tuple[float, float] | None
    casing_speed_m_per_h: float


@dataclass(frozen=True)
class FieldSpec:
    name: str
    prefix: str
    first_no: int
    well_count: int
    rigs: tuple[str, ...]
    td_m: int
    spud_offset_days: int
    surface_temp_c: float
    geothermal_c_per_m: float
    formations: tuple[Formation, ...]
    sections: tuple[SectionPlan, ...]
    # Planted patterns this field carries (see the package docstring).
    patterns: frozenset[str]
    # Mud weight every well holds across Keldra Salt when the field has no
    # salt pattern; the salt pattern draws it per well instead.
    salt_mw_range_sg: tuple[float, float] | None = None

    def formation_at(self, depth: float) -> Formation:
        for f in self.formations:
            if f.top_m <= depth < f.base_m:
                return f
        return self.formations[-1]

    def formation(self, name: str) -> Formation:
        for f in self.formations:
            if f.name == name:
                return f
        raise KeyError(name)

    def bht_c(self, depth: float) -> int:
        """Static bottom-hole temperature in whole degrees C, as the reports quote it."""
        return round(self.surface_temp_c + self.geothermal_c_per_m * depth)

    def rigs_at_scale(self, scale: int) -> tuple[str, ...]:
        """Rigs working the field: two per scale step, at most nine."""
        return self.rigs[: min(MAX_RIGS_PER_FIELD, 2 * scale)]


ORRINDALE = FieldSpec(
    name="Orrindale",
    prefix="ORD",
    first_no=101,
    well_count=28,
    rigs=("Orrin-1", "Orrin-2", "Orrin-3", "Orrin-4", "Orrin-5", "Orrin-6", "Orrin-7", "Orrin-8",
          "Orrin-9"),
    td_m=3180,
    spud_offset_days=0,
    surface_temp_c=22.0,
    geothermal_c_per_m=0.040,
    formations=(
        Formation("Hesk Overburden", 0, 620, 310, "unconsolidated sand and clay"),
        Formation("Keldra Salt", 620, 1480, 195, "halite with anhydrite stringers"),
        Formation("Dovrin Shale", 1480, 2640, 145, "reactive shale"),
        Formation("Orrindale Sand", 2640, 3180, 88, "fine-grained sandstone reservoir"),
    ),
    sections=(
        SectionPlan('26"', 0, 480, '20" surface casing', SPUD_MUD, (1.05, 1.10), None, 140),
        SectionPlan('17 1/2"', 480, 1850, '13 3/8" intermediate casing', SALT_MUD, (1.20, 1.24),
                    (1.62, 1.68), 180),
        SectionPlan('12 1/4"', 1850, 2780, '9 5/8" production casing', KCL_MUD, (1.32, 1.38),
                    (1.70, 1.76), 220),
        SectionPlan('8 1/2"', 2780, 3180, '7" liner', KCL_MUD, (1.26, 1.32), (1.55, 1.62), 300),
    ),
    patterns=frozenset({"G1", "G3"}),
)

# Vessra-3 stays the first rig at every scale: it is the rig with the
# recurring mud pump problem (pattern G4).
VESSRA_SOUTH = FieldSpec(
    name="Vessra South",
    prefix="VSS",
    first_no=201,
    well_count=14,
    rigs=("Vessra-3", "Vessra-5", "Vessra-7", "Vessra-9", "Vessra-1", "Vessra-2", "Vessra-4",
          "Vessra-6", "Vessra-8"),
    td_m=3050,
    spud_offset_days=23,
    surface_temp_c=21.0,
    geothermal_c_per_m=0.041,
    formations=(
        Formation("Tessivar Marl", 0, 1120, 285, "soft marl"),
        Formation("Keldra Salt", 1120, 1760, 190, "halite"),
        Formation("Ulvent Claystone", 1760, 2640, 138, "claystone with silt laminae"),
        Formation("Vessra Carbonate", 2640, 3050, 72, "fractured and vuggy limestone"),
    ),
    sections=(
        SectionPlan('26"', 0, 400, '20" surface casing', SPUD_MUD, (1.05, 1.10), None, 140),
        SectionPlan('17 1/2"', 400, 1780, '13 3/8" intermediate casing', SALT_MUD, (1.20, 1.24),
                    (1.62, 1.68), 180),
        SectionPlan('12 1/4"', 1780, 2620, '9 5/8" production casing', KCL_MUD, (1.28, 1.34),
                    (1.70, 1.76), 220),
        # The fractured carbonate is drilled with the lightest mud of the well.
        SectionPlan('8 1/2"', 2620, 3050, '7" liner', KCL_MUD, (1.10, 1.14), (1.50, 1.56), 300),
    ),
    patterns=frozenset({"G2", "G3", "G4"}),
    salt_mw_range_sg=(1.43, 1.47),
)

FIELDS = (ORRINDALE, VESSRA_SOUTH)

# Planted-pattern parameters. The generator uses them; the retrieval and
# analytics code never sees them.
SALT = "Keldra Salt"
CARBONATE = "Vessra Carbonate"
G1_MUD_WEIGHT_LIMIT_SG = 1.38
G2_CROSSING_M = 2660
G2_PARTIAL_BASE_M = 2780
G3_BHT_LIMIT_C = 118
G3_SECTION = '12 1/4"'
G3_TOOL = "PJ-3"
G3_TOOL_SHARE = 0.55
G4_RIG = "Vessra-3"
