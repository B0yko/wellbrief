"""Synthetic well-file corpus.

All data here is synthetic and every name is fictional: the operator, fields,
wells, rigs, formations and vendors are invented. The generated field history
is shaped like a real one: rates of penetration follow formation,
non-productive time clusters around physical causes, and the paperwork
follows the layout of a daily drilling report, an end of well report and an
incident report.

Four causal patterns are planted in the data. Nothing in the retrieval or
analytics code knows about them; they have to be rediscovered from the
documents, which is the task the tool is built for.

  G1  Orrindale, 17 1/2" through Keldra Salt. Wells run below 1.38 sg wash
      the salt out and pack off on the trip to casing.
  G2  Vessra South, 8 1/2" entering Vessra Carbonate near 2,660 m. Total
      losses unless the hole is conditioned before the top of the carbonate.
  G3  Both fields, 12 1/4" below roughly 2,400 m where BHT passes 118 C.
      One MWD generation fails at temperature.
  G4  Rig Vessra-3 carries far more rig-repair time than the others because
      of a mud pump fluid end.

The generator is seeded per field and well index, so a fixed SEED always
produces the same corpus.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .config import CORPUS_DIR
from .models import Document, Well

SEED = 20260731


@dataclass(frozen=True)
class Formation:
    name: str
    top_m: float
    base_m: float
    rop_m_per_day: float
    lithology: str


@dataclass(frozen=True)
class SectionPlan:
    size: str
    top_m: float
    base_m: float
    casing: str


@dataclass(frozen=True)
class FieldSpec:
    name: str
    prefix: str
    first_no: int
    well_count: int
    rigs: tuple[str, ...]
    td_m: float
    surface_temp_c: float
    geothermal_c_per_m: float
    formations: tuple[Formation, ...]
    sections: tuple[SectionPlan, ...]
    mud_system: str


ORRINDALE = FieldSpec(
    name="Orrindale",
    prefix="ORD",
    first_no=101,
    well_count=28,
    rigs=("Orrin-1", "Orrin-2"),
    td_m=3180,
    surface_temp_c=22.0,
    geothermal_c_per_m=0.040,
    formations=(
        Formation("Hesk Overburden", 0, 620, 310, "unconsolidated sand and clay"),
        Formation("Keldra Salt", 620, 1480, 195, "halite with anhydrite stringers"),
        Formation("Dovrin Shale", 1480, 2640, 145, "reactive shale"),
        Formation("Orrindale Sand", 2640, 3180, 88, "fine-grained sandstone reservoir"),
    ),
    sections=(
        SectionPlan('26"', 0, 480, '20" conductor'),
        SectionPlan('17 1/2"', 480, 1850, '13 3/8" intermediate casing'),
        SectionPlan('12 1/4"', 1850, 2780, '9 5/8" production casing'),
        SectionPlan('8 1/2"', 2780, 3180, '7" liner'),
    ),
    mud_system="KCl-polymer",
)

VESSRA_SOUTH = FieldSpec(
    name="Vessra South",
    prefix="VSS",
    first_no=201,
    well_count=14,
    rigs=("Vessra-3", "Vessra-5"),
    td_m=3050,
    surface_temp_c=21.0,
    geothermal_c_per_m=0.041,
    formations=(
        Formation("Tessivar Marl", 0, 1120, 285, "soft marl"),
        Formation("Keldra Salt", 1120, 1760, 190, "halite"),
        Formation("Ulvent Claystone", 1760, 2640, 138, "claystone with silt laminae"),
        Formation("Vessra Carbonate", 2640, 3050, 72, "fractured and vuggy limestone"),
    ),
    sections=(
        SectionPlan('26"', 0, 400, '20" conductor'),
        SectionPlan('17 1/2"', 400, 1780, '13 3/8" intermediate casing'),
        SectionPlan('12 1/4"', 1780, 2620, '9 5/8" production casing'),
        SectionPlan('8 1/2"', 2620, 3050, '7" liner'),
    ),
    mud_system="KCl-polymer / salt saturated",
)

FIELDS = (ORRINDALE, VESSRA_SOUTH)

OPERATOR = "Quillfen Energy"

NOISE_CODES = [
    ("WAIT_ON_WEATHER", 2.0, 9.0, "Storm warning, operations suspended. Wind gusting above 22 m/s."),
    ("WAIT_ON_MATERIALS", 2.0, 11.0, "Waiting on delivery of barite from the shore base."),
    ("RIG_REPAIR", 1.5, 7.0, "Repaired drawworks brake linkage."),
    ("THIRD_PARTY_STANDBY", 1.5, 6.0, "Wireline crew standby, unit mobilising from base."),
    ("HSE_STOP", 1.0, 3.5, "Stop-work called on rig floor, lifting plan reviewed before restart."),
    ("CEMENT_ISSUE", 3.0, 10.0, "Cement head seal leak, bumped plug not confirmed on first attempt."),
    ("BOP_TEST_FAILURE", 2.5, 8.0, "Annular preventer failed low pressure test, element replaced."),
    ("HOLE_CLEANING", 2.0, 8.0, "Circulated and back-reamed to clear cuttings bed."),
]


def _fmt_depth(v: float) -> str:
    return f"{int(round(v)):,}"


def _formation_at(spec: FieldSpec, depth: float) -> Formation:
    for f in spec.formations:
        if f.top_m <= depth < f.base_m:
            return f
    return spec.formations[-1]


def _section_at(spec: FieldSpec, depth: float) -> SectionPlan:
    for s in spec.sections:
        if s.top_m <= depth < s.base_m:
            return s
    return spec.sections[-1]


def _bht(spec: FieldSpec, depth: float) -> float:
    return round(spec.surface_temp_c + spec.geothermal_c_per_m * depth, 0)


class _WellPlan:
    """Per-well choices that decide which planted pattern the well hits."""

    def __init__(self, spec: FieldSpec, index: int, rng: random.Random):
        self.spec = spec
        self.name = f"{spec.prefix}-{spec.first_no + index}"
        self.index = index
        self.rig = spec.rigs[index % len(spec.rigs)]
        self.spud = date(2022, 1, 17) + timedelta(days=index * 41 + (0 if spec is ORRINDALE else 23))

        # Campaign learning. Early wells in each field mostly get it wrong.
        # The fix is written down in one end-of-well report and adoption
        # climbs after it: the lesson exists in the archive well before every
        # well applies it.
        learning = min(0.85, max(0.0, (index - 6) / 14.0))
        self.salt_mw_ok = spec is not ORRINDALE or rng.random() < 0.30 + learning * 0.62
        self.lcm_pretreat = spec is not VESSRA_SOUTH or rng.random() < 0.18 + learning * 0.70

        self.mwd_vendor = "Parvane Downhole"
        self.mwd_gen = "PJ-3" if rng.random() < 0.55 else "PJ-5"
        self.salt_mw = round(rng.uniform(1.30, 1.37), 2) if not self.salt_mw_ok else round(rng.uniform(1.42, 1.48), 2)
        # Never below the base of the last section, or the well would never
        # reach a stopping point.
        self.td = spec.td_m - rng.randint(0, 90)


def _npt_for_day(
    plan: _WellPlan,
    rng: random.Random,
    depth_start: float,
    depth_end: float,
    section: SectionPlan,
    formation: Formation,
    leaving_formation: bool,
) -> list[tuple[str, float, str]]:
    """Return (code, hours, description) for one 24-hour report period.

    `leaving_formation` marks the day the bit drills out of the formation it
    started in, which is when the string gets pulled and when a washed-out
    interval above finally causes trouble.
    """
    spec = plan.spec
    events: list[tuple[str, float, str]] = []

    # G1 - salt washout, Orrindale 17 1/2"
    if spec is ORRINDALE and section.size == '17 1/2"' and formation.name == "Keldra Salt" and not plan.salt_mw_ok:
        if rng.random() < 0.34:
            h = round(rng.uniform(2.0, 7.0), 1)
            events.append((
                "WELLBORE_INSTABILITY",
                h,
                f"Tight hole and overpull up to 28 t on connections at {_fmt_depth(depth_end)} m. "
                f"Caliper trend indicates washout across the salt. Mud weight held at {plan.salt_mw} sg.",
            ))
        if leaving_formation and rng.random() < 0.80:
            h = round(rng.uniform(14.0, 38.0), 1)
            events.append((
                "STUCK_PIPE",
                h,
                f"String packed off at {_fmt_depth(depth_end - rng.randint(40, 260))} m while pulling out of hole "
                f"for a bit change. Salt creep and washed-out interval across Keldra Salt. "
                f"Worked pipe, jarred and circulated free after {h:.1f} h.",
            ))
            if rng.random() < 0.28:
                events.append((
                    "FISHING",
                    round(rng.uniform(8.0, 22.0), 1),
                    "Backed off above the jars and ran a fishing assembly to recover the BHA.",
                ))

    # G2 - carbonate total losses, Vessra South 8 1/2"
    if (
        spec is VESSRA_SOUTH
        and section.size == '8 1/2"'
        and depth_start < 2660 <= depth_end
        and not plan.lcm_pretreat
    ):
        h = round(rng.uniform(18.0, 52.0), 1)
        events.append((
            "LOST_CIRCULATION",
            h,
            f"Total losses on entering Vessra Carbonate at {_fmt_depth(depth_end)} m. Returns lost, "
            f"no fluid to surface. Pumped LCM pills and cement plug. Lost {rng.randint(180, 640)} bbl mud.",
        ))
    if (
        spec is VESSRA_SOUTH
        and section.size == '8 1/2"'
        and 2660 < depth_start < 2780
        and not plan.lcm_pretreat
        and rng.random() < 0.55
    ):
        events.append((
            "LOST_CIRCULATION",
            round(rng.uniform(6.0, 14.0), 1),
            "Partial losses continued below the carbonate top. Drilled ahead blind with a sacrificial sweep.",
        ))

    # G3 - MWD failure above 118 C
    bht = _bht(spec, depth_end)
    if section.size == '12 1/4"' and bht > 118 and plan.mwd_gen == "PJ-3" and rng.random() < 0.22:
        h = round(rng.uniform(12.0, 28.0), 1)
        events.append((
            "DOWNHOLE_TOOL_FAILURE",
            h,
            f"MWD signal lost at {_fmt_depth(depth_end)} m, BHT {bht:.0f} C. Pulled out of hole and replaced "
            f"{plan.mwd_vendor} {plan.mwd_gen} directional module. Vendor report cites thermal derating above 115 C.",
        ))

    # G4 - Vessra-3 mud pump
    if plan.rig == "Vessra-3" and rng.random() < 0.16:
        events.append((
            "RIG_REPAIR",
            round(rng.uniform(3.0, 13.0), 1),
            "Mud pump #2 fluid end module changed out after washed valve seat. Recurring on this unit.",
        ))

    # Background noise
    if rng.random() < 0.18:
        code, lo, hi, desc = NOISE_CODES[rng.randrange(len(NOISE_CODES))]
        events.append((code, round(rng.uniform(lo, hi), 1), desc))

    total = sum(e[1] for e in events)
    if total > 23.0:  # a day only has 24 hours
        scale = 23.0 / total
        events = [(c, round(h * scale, 1), d) for c, h, d in events]
    return events


def _render_ddr(
    plan: _WellPlan,
    report_no: int,
    day: date,
    depth_start: float,
    depth_end: float,
    section: SectionPlan,
    formation: Formation,
    mud_weight: float,
    npt: list[tuple[str, float, str]],
    ops_lines: list[str],
    rng: random.Random,
) -> str:
    spec = plan.spec
    npt_hours = sum(e[1] for e in npt)
    bht = _bht(spec, depth_end)
    ecd = round(mud_weight + rng.uniform(0.05, 0.11), 2)
    lines = [
        "DAILY DRILLING REPORT",
        f"Operator: {OPERATOR}    Field: {spec.name}    Well: {plan.name}",
        f"Rig: {plan.rig}    Report No: {report_no:03d}    Date: {day.isoformat()}",
        "Report period: 06:00 - 06:00",
        "",
        "DEPTH",
        f"  Depth at start        : {_fmt_depth(depth_start)} m MD",
        f"  Depth at end          : {_fmt_depth(depth_end)} m MD",
        f"  Progress              : {_fmt_depth(depth_end - depth_start)} m",
        f'  Hole section          : {section.size}',
        f"  Formation at TD       : {formation.name} ({formation.lithology})",
        "",
        "MUD",
        f"  System                : {spec.mud_system}",
        f"  Weight                : {mud_weight} sg",
        f"  PV / YP               : {rng.randint(16, 32)} cP / {rng.randint(12, 26)} lb/100ft2",
        f"  ECD at shoe           : {ecd} sg",
        "",
        "BHA AND PARAMETERS",
        f"  Bit                   : {section.size} PDC, IADC {rng.choice(['M323', 'M433', 'S123', 'M223'])}",
        f"  MWD                   : {plan.mwd_vendor} {plan.mwd_gen}",
        f"  WOB / RPM / Flow      : {rng.randint(8, 22)} t / {rng.randint(90, 160)} rpm / {rng.randint(420, 900)} gpm",
        f"  BHT max circulating   : {bht:.0f} C",
        "",
        "OPERATIONS SUMMARY (24 h)",
    ]
    lines += [f"  {l}" for l in ops_lines]
    lines += [
        "",
        "TIME BREAKDOWN",
        f"  Productive time       : {24.0 - npt_hours:.1f} h",
        f"  Non-productive time   : {npt_hours:.1f} h",
        "",
    ]
    if npt:
        lines.append("NPT DETAIL")
        for code, hours, desc in npt:
            lines.append(f"  Code                  : {code}")
            lines.append(f"  Hours                 : {hours:.1f}")
            lines.append(f"  Description           : {desc}")
            lines.append("")
    else:
        lines.append("NPT DETAIL")
        lines.append("  None reported this period.")
        lines.append("")
    lines.append("REMARKS")
    lines.append(f"  Casing programme for this section: {section.casing}.")
    lines.append(f"  Next 24 h: continue {section.size} hole in {formation.name}.")
    return "\n".join(lines)


def _render_eowr(plan: _WellPlan, docs: list[Document], npt_rows: list[tuple[str, float]], rng: random.Random) -> str:
    spec = plan.spec
    by_code: dict[str, float] = {}
    for code, hours in npt_rows:
        by_code[code] = by_code.get(code, 0.0) + hours
    total_npt = sum(by_code.values())
    days = len(docs)
    lines = [
        "END OF WELL REPORT",
        f"Operator: {OPERATOR}    Field: {spec.name}    Well: {plan.name}",
        f"Rig: {plan.rig}    Spud: {plan.spud.isoformat()}    Days on well: {days}",
        f"Total depth: {_fmt_depth(plan.td)} m MD",
        "",
        "1. WELL SUMMARY",
        f"  {plan.name} was drilled as a development well on {spec.name} to {_fmt_depth(plan.td)} m MD.",
        f"  Four hole sections were drilled: " + ", ".join(s.size for s in spec.sections) + ".",
        f"  Mud system throughout: {spec.mud_system}.",
        "",
        "2. TIME ANALYSIS",
        f"  Total time on well    : {days * 24:.1f} h",
        f"  Non-productive time   : {total_npt:.1f} h ({total_npt / (days * 24) * 100:.1f} %)",
        "",
        "3. NPT BREAKDOWN BY CODE",
    ]
    for code, hours in sorted(by_code.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {code:<24} {hours:>7.1f} h")
    lines += ["", "4. LESSONS LEARNED"]

    n = 1
    if spec is ORRINDALE and not plan.salt_mw_ok:
        lines.append(
            f"  {n}. The 17 1/2\" section was drilled through Keldra Salt at {plan.salt_mw} sg. Caliper and "
            f"cement volumes both indicate significant washout. The string packed off on the trip out for "
            f"{spec.sections[1].casing}. Mud weight in the salt was the controlling factor."
        )
        n += 1
    if spec is ORRINDALE and plan.salt_mw_ok:
        lines.append(
            f"  {n}. Keldra Salt was drilled at {plan.salt_mw} sg with a salt saturated brine sweep every 250 m. "
            f"Hole held gauge and the trip out for {spec.sections[1].casing} was uneventful. This is the "
            f"practice recommended in the ORD-119 end of well report and it continues to hold."
        )
        n += 1
    if spec is VESSRA_SOUTH and not plan.lcm_pretreat:
        lines.append(
            f"  {n}. Total losses were taken on entering Vessra Carbonate near 2,660 m. The 8 1/2\" section "
            f"was drilled into the carbonate top at full flow rate with no pre-treatment. Recovery took "
            f"several days of LCM and a cement plug."
        )
        n += 1
    if spec is VESSRA_SOUTH and plan.lcm_pretreat:
        lines.append(
            f"  {n}. Flow rate was cut to 430 gpm and a 30 bbl fibrous LCM pill was spotted 20 m above the "
            f"Vessra Carbonate top before drilling in. Only seepage losses were recorded. This follows the "
            f"VSS-207 recommendation."
        )
        n += 1
    if plan.mwd_gen == "PJ-3":
        lines.append(
            f"  {n}. {plan.mwd_vendor} {plan.mwd_gen} directional tools were run in the 12 1/4\" section where "
            f"BHT reaches {_bht(spec, spec.sections[2].base_m):.0f} C. Reliability at temperature should be "
            f"reviewed with the vendor before the next well."
        )
        n += 1
    if plan.rig == "Vessra-3":
        lines.append(
            f"  {n}. Mud pump #2 on {plan.rig} required repeated fluid end work. The unit should be inspected "
            f"between wells rather than repaired on the critical path."
        )
        n += 1
    lines.append(f"  {n}. Cementing and casing running went to programme with no notable deviation.")

    lines += ["", "5. RECOMMENDATIONS FOR FUTURE WELLS"]
    if spec is ORRINDALE:
        lines.append("  - Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m.")
        lines.append("  - Run a caliper on the intermediate section before running casing.")
    else:
        lines.append("  - Condition the hole and spot an LCM pill before entering Vessra Carbonate.")
        lines.append("  - Reduce flow rate to below 450 gpm for the first 60 m of carbonate.")
    lines.append("  - Confirm MWD temperature rating against the offset BHT profile at the planning stage.")
    return "\n".join(lines)


def _render_incident(plan: _WellPlan, day: date, code: str, hours: float, desc: str, depth: float, rng: random.Random) -> str:
    spec = plan.spec
    formation = _formation_at(spec, depth)
    section = _section_at(spec, depth)
    severity = "High" if hours >= 30 else "Medium"
    root = {
        "STUCK_PIPE": "Mud weight below the value required to hold the salt in gauge, combined with an "
                      "extended open hole exposure time.",
        "LOST_CIRCULATION": "The carbonate top was entered at full flow rate without hole conditioning or "
                            "a pre-emptive LCM pill.",
        "DOWNHOLE_TOOL_FAILURE": "Directional tool operated above its demonstrated continuous temperature "
                                 "rating for an extended period.",
        "RIG_REPAIR": "Fluid end module on mud pump #2 reached end of service life and was not replaced "
                      "during the previous maintenance window.",
        "WELLBORE_INSTABILITY": "Hole enlargement in the salt interval reduced annular velocity and allowed "
                                "cuttings to accumulate.",
    }.get(code, "Equipment or process deviation, see sequence of events.")
    return "\n".join([
        "WELL OPERATIONS INCIDENT REPORT",
        f"Operator: {OPERATOR}    Field: {spec.name}    Well: {plan.name}",
        f"Rig: {plan.rig}    Date: {day.isoformat()}    Severity: {severity}",
        f"Classification: {code}    Lost time: {hours:.1f} h",
        "",
        "1. LOCATION",
        f"  Depth: {_fmt_depth(depth)} m MD    Hole section: {section.size}    Formation: {formation.name}",
        f"  Bottom hole temperature: {_bht(spec, depth):.0f} C",
        "",
        "2. SEQUENCE OF EVENTS",
        f"  {desc}",
        f"  Operations were suspended for {hours:.1f} h. No injuries and no environmental release.",
        "",
        "3. IMMEDIATE CAUSE",
        f"  {desc.split('.')[0]}.",
        "",
        "4. ROOT CAUSE",
        f"  {root}",
        "",
        "5. CORRECTIVE ACTIONS",
        "  - Update the drilling programme for the next well in this field with the mitigation above.",
        "  - Brief the incoming crew during handover.",
        f"  - Review with the service provider at the next {rng.choice(['weekly', 'monthly'])} performance meeting.",
    ])


def generate(out_dir: Path | None = None, write_files: bool = True) -> tuple[list[Well], list[Document]]:
    """Build the whole corpus. Deterministic for a fixed SEED."""
    out_dir = Path(out_dir) if out_dir else CORPUS_DIR
    wells: list[Well] = []
    docs: list[Document] = []

    for spec in FIELDS:
        for i in range(spec.well_count):
            rng = random.Random(f"{SEED}:{spec.name}:{i}")
            plan = _WellPlan(spec, i, rng)
            wells.append(Well(
                name=plan.name,
                field_name=spec.name,
                rig=plan.rig,
                spud_date=plan.spud.isoformat(),
                td_m=float(plan.td),
                sections=[s.size for s in spec.sections],
                formations=[f.name for f in spec.formations],
            ))

            depth = 0.0
            day_no = 0
            well_docs: list[Document] = []
            npt_rows: list[tuple[str, float]] = []
            incident_seeds: list[tuple[date, str, float, str, float]] = []

            def emit_day(
                day_index: int,
                depth_start: float,
                depth_end: float,
                section: SectionPlan,
                formation: Formation,
                mud_weight: float,
                npt: list[tuple[str, float, str]],
                ops: list[str],
            ) -> None:
                day = plan.spud + timedelta(days=day_index - 1)
                for code, hours, desc in npt:
                    ops.append(f"NPT {hours:.1f} h  {code}: {desc}")
                well_docs.append(Document(
                    doc_id=f"DDR-{plan.name}-{day_index:03d}",
                    doc_type="ddr",
                    well=plan.name,
                    field_name=spec.name,
                    date=day.isoformat(),
                    title=f"Daily drilling report {day_index:03d} - {plan.name}",
                    text=_render_ddr(plan, day_index, day, depth_start, depth_end, section,
                                     formation, mud_weight, npt, ops, rng),
                    meta={},
                ))
                npt_rows.extend((c, h) for c, h, _ in npt)
                for c, h, d in npt:
                    if h >= 16.0:
                        incident_seeds.append((day, c, h, d, depth_end))

            for section in spec.sections:
                section_base = min(section.base_m, plan.td)
                if depth >= plan.td:
                    break
                depth = max(depth, section.top_m)

                while depth < section_base and day_no < 120:
                    day_no += 1
                    formation = _formation_at(spec, depth)
                    mud_weight = (
                        plan.salt_mw if formation.name == "Keldra Salt" and spec is ORRINDALE
                        else round(1.06 + depth / 3000.0 * 0.42 + rng.uniform(-0.02, 0.03), 2)
                    )
                    rop = formation.rop_m_per_day * rng.uniform(0.78, 1.22)
                    reach = min(depth + rop, section_base)
                    leaving_formation = depth < formation.base_m <= reach

                    npt = _npt_for_day(plan, rng, depth, reach, section, formation, leaving_formation)
                    npt_hours = sum(e[1] for e in npt)
                    # A minimum of 2 m guarantees the loop terminates even on a
                    # day that was almost entirely non-productive.
                    drilled = max(2.0, rop * (24.0 - npt_hours) / 24.0)
                    new_depth = min(depth + drilled, section_base)

                    mid = depth + (new_depth - depth) * 0.6
                    emit_day(day_no, depth, new_depth, section, formation, mud_weight, npt, [
                        f"06:00-13:30  Drilled {section.size} hole from {_fmt_depth(depth)} m to {_fmt_depth(mid)} m.",
                        f"13:30-15:00  Circulated bottoms up, surveyed at {_fmt_depth(mid)} m.",
                        f"15:00-06:00  Drilled ahead to {_fmt_depth(new_depth)} m in {formation.name}.",
                    ])
                    depth = new_depth

                if depth >= plan.td:
                    break

                day_no += 1
                formation = _formation_at(spec, depth - 1)
                mud_weight = round(1.06 + depth / 3000.0 * 0.42 + rng.uniform(-0.02, 0.03), 2)
                npt = _npt_for_day(plan, rng, depth, depth, section, formation, False)
                emit_day(day_no, depth, depth, section, formation, mud_weight, npt, [
                    f"06:00-14:00  Circulated and conditioned mud, pulled out of hole from {_fmt_depth(depth)} m.",
                    f"14:00-24:00  Ran {section.casing} to {_fmt_depth(depth)} m.",
                    "24:00-06:00  Cemented casing, waiting on cement.",
                ])

            docs.extend(well_docs)

            eowr_date = plan.spud + timedelta(days=day_no + 12)
            docs.append(Document(
                doc_id=f"EOWR-{plan.name}",
                doc_type="eowr",
                well=plan.name,
                field_name=spec.name,
                date=eowr_date.isoformat(),
                title=f"End of well report - {plan.name}",
                text=_render_eowr(plan, well_docs, npt_rows, rng),
                meta={
                    "rig": plan.rig,
                    "days_on_well": day_no,
                    "td_m": plan.td,
                    "total_npt_hours": round(sum(h for _, h in npt_rows), 1),
                    "salt_mw_ok": plan.salt_mw_ok,
                    "lcm_pretreat": plan.lcm_pretreat,
                    "mwd": f"{plan.mwd_vendor} {plan.mwd_gen}",
                },
            ))

            for k, (day, code, hours, desc, dep) in enumerate(incident_seeds[:3], start=1):
                docs.append(Document(
                    doc_id=f"INC-{plan.name}-{k:02d}",
                    doc_type="incident",
                    well=plan.name,
                    field_name=spec.name,
                    date=day.isoformat(),
                    title=f"Incident report {k} - {plan.name} - {code}",
                    text=_render_incident(plan, day, code, hours, desc, dep, rng),
                    meta={"rig": plan.rig, "code": code, "hours": hours, "depth_m": round(dep, 1)},
                ))

    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        for d in docs:
            (out_dir / f"{d.doc_id}.txt").write_text(d.text, encoding="utf-8")
        (out_dir / "_wells.json").write_text(
            json.dumps([w.to_dict() for w in wells], indent=2), encoding="utf-8"
        )
    return wells, docs
