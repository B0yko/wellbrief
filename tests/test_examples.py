"""`examples/`: the alternative report template and the non-default NPT ledger CSV.

`examples/alt-template/` (three DDRs and one EOWR in a different label vocabulary, site NPT
codes, and its own `wellbrief.toml`) must ingest to the same ledger rows and yield the same
mitigation miner candidates as the same events written in the canonical template.
`examples/npt-ledger.csv` (a non-default column layout, `;` delimiter, and
`examples/npt-ledger.toml`'s mapping) must ingest cleanly through `[csv.columns]`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from wellbrief.ingest import ingest_folder
from wellbrief.miner import MinerScope, mine_mitigations
from wellbrief.store import Store

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ALT_TEMPLATE = EXAMPLES / "alt-template"

FOOTER = "Synthetic demonstration document. Operator, fields, wells, rigs and vendors are fictional."

FIELD = "Marrow Deep"
FORMATION = "Ashgrove Shale"
SECTION = '12 1/4"'
CODE = "STUCK_PIPE"
CLEAN_WELL = "MRD-204"

# The same three events and the same end-of-well report as `examples/alt-template/`'s own
# DDR-MRD-2{01,02,03}-001.txt and EOWR-MRD-204.txt, written in the canonical template instead.
_CANONICAL_DDR = """DAILY DRILLING REPORT
Operator: Fenwick Resources    Field: Marrow Deep    Well: {well}
Rig: Marrow-1    Report No: 001    Date: {date}

DEPTH
  Depth at start        : {depth_start} m MD
  Depth at end          : {depth_end} m MD
  Progress              : 100 m
  Hole section          : 12 1/4"
  Formation at TD       : Ashgrove Shale

MUD
  Weight                : 1.38 sg
  ECD                   : 1.44 sg

DIRECTIONAL
  MWD                   : Fenwick Downhole FX-2

TIME
  Productive time       : {productive} h
  Non-productive time   : {npt_hours} h

NPT DETAIL
  Code                  : STUCK_PIPE
  Hours                 : {npt_hours}
  Depth                 : {depth_end} m MD
  Formation             : Ashgrove Shale
  Description           : {description}

""" + FOOTER + "\n"

_CANONICAL_EOWR = """END OF WELL REPORT
Operator: Fenwick Resources    Field: Marrow Deep    Well: MRD-204
Rig: Marrow-1    Report No: 012    Date: 2026-04-20

Days on well          : 11
Total depth           : 2,180 m MD
Non-productive time   : 4.0 h

3. NPT BREAKDOWN BY CODE
  WAIT_ON_WEATHER          4.0 h

4. LESSONS LEARNED
  1. Hold at least 1.42 sg across Ashgrove Shale and monitor trip speed to avoid pack-off.

5. RECOMMENDATIONS FOR FUTURE WELLS
  - Run a wiper trip through Ashgrove Shale before tripping out for casing.

""" + FOOTER + "\n"

# (well, date, depth_start, depth_end, productive, npt_hours, description) -- must match the
# figures baked into examples/alt-template/DDR-MRD-2{01,02,03}-001.txt exactly.
_EVENTS = [
    ("MRD-201", "2026-04-10", "2,050", "2,150", "6.0", "18.0",
     "String packed off at 2,150 m while pulling out of hole through Ashgrove Shale."),
    ("MRD-202", "2026-04-12", "2,100", "2,205", "2.5", "21.5",
     "String packed off at 2,205 m on connection while backreaming through Ashgrove Shale."),
    ("MRD-203", "2026-04-14", "1,990", "2,090", "8.0", "16.0",
     "String packed off at 2,090 m while pulling out of hole through Ashgrove Shale."),
]


def _write_canonical_fixtures(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for well, date, depth_start, depth_end, productive, npt_hours, description in _EVENTS:
        text = _CANONICAL_DDR.format(well=well, date=date, depth_start=depth_start, depth_end=depth_end,
                                     productive=productive, npt_hours=npt_hours, description=description)
        (folder / f"DDR-{well}-001.txt").write_text(text, encoding="utf-8")
    (folder / "EOWR-MRD-204.txt").write_text(_CANONICAL_EOWR, encoding="utf-8")


def _ledger_rows(store: Store) -> list[tuple]:
    return sorted(
        (e.well, e.date, e.code, e.hours, e.hole_section, e.formation, e.depth_m, e.rig, e.description)
        for e in store.npt(field_name=FIELD)
    )


def test_alt_template_files_exist_and_are_fictional() -> None:
    files = sorted(p.name for p in ALT_TEMPLATE.iterdir())
    assert files == ["DDR-MRD-201-001.txt", "DDR-MRD-202-001.txt", "DDR-MRD-203-001.txt",
                     "EOWR-MRD-204.txt", "wellbrief.toml"]
    for name in files:
        if name.endswith(".txt"):
            assert FOOTER in (ALT_TEMPLATE / name).read_text(encoding="utf-8")


def test_alt_template_ingests_to_the_same_ledger_rows_as_the_canonical_template(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical"
    _write_canonical_fixtures(canonical_dir)

    canonical_store = Store(tmp_path / "canonical.db")
    canonical_coverage = ingest_folder(canonical_store, canonical_dir)
    assert canonical_coverage.npt_events == 3
    assert not canonical_coverage.skipped

    alt_store = Store(tmp_path / "alt.db")
    alt_coverage = ingest_folder(alt_store, ALT_TEMPLATE)
    assert alt_coverage.npt_events == 3
    assert not alt_coverage.skipped

    canonical_rows = _ledger_rows(canonical_store)
    alt_rows = _ledger_rows(alt_store)
    assert alt_rows == canonical_rows
    # every event resolved to the taxonomy code, not the site's own "SP" spelling
    assert {row[2] for row in alt_rows} == {CODE}


def test_alt_templates_own_wellbrief_toml_is_picked_up_without_a_config_flag(tmp_path: Path) -> None:
    """`ingest` finds a folder's own `wellbrief.toml` on its own (no `--config` needed), and
    records the settings it parsed with on every file, so a later re-parse is reproducible."""
    store = Store(tmp_path / "alt.db")
    ingest_folder(store, ALT_TEMPLATE)
    files = {f.path: f for f in store.files()}
    ddr_file = files["DDR-MRD-201-001.txt"]
    assert ddr_file.parse_config["parse"]["ddr_labels"]["weight"] == "MW"
    assert ddr_file.parse_config["taxonomy_aliases"] == {"SP": "STUCK_PIPE"}


def test_alt_template_yields_the_same_mitigation_candidates_as_the_canonical_template(
    tmp_path: Path,
) -> None:
    canonical_dir = tmp_path / "canonical"
    _write_canonical_fixtures(canonical_dir)

    canonical_store = Store(tmp_path / "canonical.db")
    ingest_folder(canonical_store, canonical_dir)

    alt_store = Store(tmp_path / "alt.db")
    ingest_folder(alt_store, ALT_TEMPLATE)

    # No settings passed to `mine_mitigations` for either store: lessons, recommendations and
    # corrective actions come from each document's own `meta`, already parsed correctly at
    # ingest time with that folder's own `wellbrief.toml` -- proving the
    # miner does not need the alt template's section headings at query time at all.
    scope = MinerScope(code=CODE, field_name=FIELD, hole_section=SECTION, formation=FORMATION,
                       clean_wells=frozenset({CLEAN_WELL}))
    canonical_mitigations = mine_mitigations(canonical_store, scope)
    alt_mitigations = mine_mitigations(alt_store, scope)

    assert [(m.text, m.well) for m in alt_mitigations] == [(m.text, m.well) for m in canonical_mitigations]
    assert len(canonical_mitigations) == 2   # the lesson and the recommendation both qualify


def test_alt_template_without_its_own_toml_would_not_resolve_the_site_codes(tmp_path: Path) -> None:
    """Sanity check that the equivalence above is doing real work: dropping the alt template's
    own `wellbrief.toml` (so ingestion falls back to the canonical labels) leaves the site's own
    labels and "SP" code unrecognised."""
    stripped = tmp_path / "stripped"
    stripped.mkdir()
    for name in ("DDR-MRD-201-001.txt", "DDR-MRD-202-001.txt", "DDR-MRD-203-001.txt", "EOWR-MRD-204.txt"):
        shutil.copy(ALT_TEMPLATE / name, stripped / name)
    store = Store(tmp_path / "stripped.db")
    ingest_folder(store, stripped)
    events = store.npt(field_name=FIELD)
    assert events == []   # "Lost-time code: SP" was never read as a Code label at all


# ---------------------------------------------------------------------------
# examples/npt-ledger.csv + examples/npt-ledger.toml
# ---------------------------------------------------------------------------


def test_npt_ledger_csv_ingests_with_its_column_mapping(tmp_path: Path) -> None:
    folder = tmp_path / "ledger"
    folder.mkdir()
    shutil.copy(EXAMPLES / "npt-ledger.csv", folder / "npt-ledger.csv")

    store = Store(tmp_path / "db.sqlite")
    coverage = ingest_folder(store, folder, config=EXAMPLES / "npt-ledger.toml", field="Thornmere Ridge")

    assert not coverage.skipped
    assert not coverage.csv_row_errors
    assert not coverage.csv_row_warnings
    assert coverage.npt_events == 20

    events = store.npt(field_name="Thornmere Ridge")
    assert len(events) == 20
    stuck_pipe = [e for e in events if e.code == "STUCK_PIPE"]
    assert len(stuck_pipe) == 3
    first = next(e for e in events if e.well == "THR-101" and e.date == "2026-02-10")
    assert first.code == "STUCK_PIPE"
    assert first.hours == 14.5
    assert first.hole_section == '12 1/4"'
    assert first.formation == "Dunraven Marl"
    assert first.depth_m == 1402.0
    assert first.rig == "Thornmere-2"
