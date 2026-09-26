"""The synthetic drilling campaign: wells, schedule, write-up and rendered documents.

Each field's wells are simulated one by one (`simulate.py`), then put on the
rig schedule, then written up (`writeup.py`), then rendered to lines
(`render.py`).

Randomness comes from one `random.Random` per well, keyed on the seed, the
field prefix and the well index (never the display name, so renaming a field
does not change its history). Two more per-well streams, keyed the same way
with a suffix, draw the cosmetic report values (mud rheology, ECD, bit type,
WOB, RPM, flow rate) and the choice of incident corrective actions, so that
changing either never shifts the simulated events. One stream per field,
keyed on the prefix, picks which wells run the PJ-3 MWD generation: a fixed
share of each field's wells, at random positions.
"""

from __future__ import annotations

import random
from datetime import timedelta

from .fields import (
    FIELDS,
    FIRST_SPUD,
    FIRST_WELL_STAGGER_DAYS,
    G1_MUD_WEIGHT_LIMIT_SG,
    G3_SECTION,
    G3_TOOL,
    G3_TOOL_SHARE,
    MWD_TOOLS,
    RIG_MOVE_DAYS,
    FieldSpec,
)
from .records import Corpus, RenderedDocument, WellRecord
from .render import render_ddr, render_eowr, render_incident
from .simulate import WellSim
from .writeup import originators, pattern_status, write_incidents, write_lessons

SEED = 20260731
MAX_SCALE = 10
EOWR_AFTER_RELEASE_DAYS = 13


def _tools(spec: FieldSpec, seed: int, count: int) -> set[int]:
    """Indices of the wells that run the PJ-3 generation in the 12 1/4" section."""
    return set(random.Random(f"{seed}:{spec.prefix}:mwd").sample(range(count), round(G3_TOOL_SHARE * count)))


def _simulate_well(spec: FieldSpec, index: int, seed: int, rig: str, pj3: bool) -> WellRecord:
    rng = random.Random(f"{seed}:{spec.prefix}:{index}")
    trng = random.Random(f"{seed}:{spec.prefix}:{index}:text")

    # Campaign learning: early wells mostly get the salt and the carbonate
    # wrong; adoption of the written practice climbs with the well index.
    learning = min(0.85, max(0.0, (index - 6) / 14.0))
    salt_mw_ok = "G1" not in spec.patterns or rng.random() < 0.30 + learning * 0.62
    lcm_pretreat = "G2" not in spec.patterns or rng.random() < 0.18 + learning * 0.70
    if "G1" in spec.patterns:
        salt_mw = round(rng.uniform(1.42, 1.48), 2) if salt_mw_ok else round(rng.uniform(1.30, 1.37), 2)
        if salt_mw_ok != (salt_mw >= G1_MUD_WEIGHT_LIMIT_SG):
            raise RuntimeError("salt mud weight disagrees with the practice flag")
    else:
        assert spec.salt_mw_range_sg is not None
        salt_mw = round(rng.uniform(*spec.salt_mw_range_sg), 2)
    td = spec.td_m - rng.randint(0, 90)
    move_days = rng.randint(*RIG_MOVE_DAYS)

    # A PJ-3 well changes to PJ-5 below the 12 1/4" section: the tool is not
    # run in the hotter 8 1/2" hole.
    tool = G3_TOOL if pj3 else MWD_TOOLS[1]
    g3_index = [s.size for s in spec.sections].index(G3_SECTION)
    by_section = {s.size: tool if i <= g3_index else MWD_TOOLS[1] for i, s in enumerate(spec.sections)}

    well = WellRecord(
        spec=spec, index=index, name=f"{spec.prefix}-{spec.first_no + index}", rig=rig, td_m=td, mwd=tool,
        mwd_by_section=by_section, salt_mw_sg=salt_mw, salt_mw_ok=salt_mw_ok, lcm_pretreat=lcm_pretreat,
        move_days=move_days,
    )
    WellSim(well, rng, trng).run()
    return well


def _schedule(spec: FieldSpec, wells: list[WellRecord]) -> None:
    """Spud each well when its rig is free; date every report and the end-of-well report."""
    free: dict[str, WellRecord] = {}
    for w in wells:
        prev = free.get(w.rig)
        if prev is None:
            spud = FIRST_SPUD + timedelta(days=spec.spud_offset_days + w.index * FIRST_WELL_STAGGER_DAYS)
        else:
            spud = prev.release + timedelta(days=w.move_days)
        w.spud = spud
        for d in w.days:
            d.day = spud + timedelta(days=d.report_no - 1)
        w.eowr_date = w.release + timedelta(days=EOWR_AFTER_RELEASE_DAYS)
        free[w.rig] = w


def build_corpus(seed: int = SEED, scale: int = 1) -> Corpus:
    """Simulate the whole campaign and render every document to lines."""
    if not 1 <= scale <= MAX_SCALE:
        raise ValueError(f"scale must be between 1 and {MAX_SCALE}")
    wells: list[WellRecord] = []
    rigs: dict[str, tuple[str, ...]] = {}
    for spec in FIELDS:
        field_rigs = spec.rigs_at_scale(scale)
        rigs[spec.name] = field_rigs
        count = spec.well_count * scale
        pj3 = _tools(spec, seed, count)
        field_wells = [_simulate_well(spec, i, seed, field_rigs[i % len(field_rigs)], i in pj3)
                       for i in range(count)]
        _schedule(spec, field_wells)
        for w in field_wells:
            w.patterns = pattern_status(w)
        origins = originators(field_wells)
        for w in field_wells:
            write_lessons(w, origins)
            write_incidents(w, random.Random(f"{seed}:{spec.prefix}:{w.index}:incident"))
        wells.extend(field_wells)

    documents: list[RenderedDocument] = []
    for w in wells:
        for report in w.days:
            documents.append(RenderedDocument(
                doc_id=report.doc_id, doc_type="ddr", well=w.name, field_name=w.spec.name,
                date=report.day.isoformat(), title=f"Daily drilling report {report.report_no:03d} - {w.name}",
                lines=render_ddr(w, report),
            ))
        documents.append(RenderedDocument(
            doc_id=w.eowr_id, doc_type="eowr", well=w.name, field_name=w.spec.name,
            date=w.eowr_date.isoformat(), title=f"End of well report - {w.name}", lines=render_eowr(w),
        ))
        for inc in w.incidents:
            start = w.report(inc.reports[0])
            documents.append(RenderedDocument(
                doc_id=inc.doc_id, doc_type="incident", well=w.name, field_name=w.spec.name,
                date=start.day.isoformat(),
                title=f"Incident report {inc.number} - {w.name} - {inc.event.code}",
                lines=render_incident(w, inc),
            ))
    return Corpus(seed=seed, scale=scale, fields=FIELDS, rigs=rigs, wells=wells, documents=documents)
