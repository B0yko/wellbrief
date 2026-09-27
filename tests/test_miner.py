"""Mitigation miner v2: the practice/failure/neutral classifier, relevance,
clean-well sourcing per scope, near-duplicate removal and the 3-mitigation cap.

The heading `qa.ask` shows above what the miner finds is tested in `test_ask.py`, next to
the rest of `ask`'s behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief.miner import (
    MinerScope,
    classify_many,
    classify_sentence,
    exposed_clean_wells,
    mine_mitigations,
)
from wellbrief.models import Document, NptEvent
from wellbrief.store import Store

FIELD = "Testfield"
SECTION = '17 1/2"'
FORMATION = "Keldra Salt"
OTHER_SECTION = '12 1/4"'
OTHER_FORMATION = "Dovrin Shale"
CODE = "STUCK_PIPE"

# ---------------------------------------------------------------------------
# classify_sentence: practice | failure | neutral
# ---------------------------------------------------------------------------

PRACTICE_EXAMPLES = [
    # Imperative recommendations (source: RECOMMENDATIONS / CORRECTIVE_ACTIONS).
    "Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m.",
    "Inspect and change mud pump fluid ends between wells rather than on the critical path.",
    "Run a caliper across the salt before the trip out for casing.",
    # Explicit success phrases in an end-of-well lesson.
    "Keldra Salt was drilled at 1.45 sg with a saturated brine sweep every 250 m; hole held gauge.",
    ("Flow rate was cut to 430 gpm and a 30 bbl fibrous LCM pill was spotted just above the Vessra "
     "Carbonate top before drilling in. Only seepage losses were recorded."),
    'Ran Parvane Downhole PJ-5 rated to 150 C in the 12 1/4" section; no MWD failures above 118 C.',
    "Mud pump fluid ends on Orrin-2 were inspected and changed between wells; no pump NPT.",
    # Tricky: an instruction that names the very failure word it prevents (a
    # real corrective action from the synthetic demonstration corpus, and a good example
    # of an explicit practice signal beating a failure word in the same
    # sentence).
    "Place the jars so that the string can be backed off above the stuck point in the salt.",
]


@pytest.mark.parametrize("text", PRACTICE_EXAMPLES)
def test_practice_examples_classify_as_practice(text: str) -> None:
    assert classify_sentence(text) == "practice"


FAILURE_EXAMPLES = [
    ('Keldra Salt was drilled at 1.30 sg in the 17 1/2" section; the string packed off at 1,402 m '
     "on the trip out for the 13-3/8 in. casing."),
    ('Total losses were taken on entering Vessra Carbonate at 2,662 m; the 8 1/2" section was '
     "drilled into the carbonate top at full flow rate with no pre-treatment."),
    ('The Parvane Downhole PJ-3 MWD failed twice in the 12 1/4" section, at BHT up to 128 C; the '
     "tool was run above its demonstrated temperature rating."),
    ('Keldra Salt was drilled at 1.35 sg in the 17 1/2" section; tight hole from salt creep was '
     "recorded on 3 daily reports."),
    ("The cement head seal leaked during the cement job on the 9 5/8 in. casing and was replaced "
     "before displacement."),
    # Tricky: a failure narrative that carries a number next to a word that
    # could pass for a success claim ("held") without being the exact
    # practice phrase this classifier looks for ("held gauge").
    ("The string packed off at 1,402 m while pulling out of hole; the crew held 210 psi circulating "
     "pressure for six hours trying to work it free."),
]


@pytest.mark.parametrize("text", FAILURE_EXAMPLES)
def test_failure_examples_classify_as_failure(text: str) -> None:
    assert classify_sentence(text) == "failure"


NEUTRAL_EXAMPLES = [
    "Cement jobs went to programme with no notable deviation.",
    'Parvane Downhole PJ-3 directional tools were run in the 12 1/4" section, where BHT reached 110 C.',
    "Mud pumps on Vessra-3 were run on the rig's standard maintenance schedule.",
    # Tricky: a neutral closing lesson -- wraps the section up, prescribes
    # nothing, and does not describe a failure either.
    "The section was completed and the rig moved to the next location.",
]


@pytest.mark.parametrize("text", NEUTRAL_EXAMPLES)
def test_neutral_examples_classify_as_neutral(text: str) -> None:
    assert classify_sentence(text) == "neutral"


def test_classify_many_labels_each_sentence_in_order() -> None:
    labels = classify_many([PRACTICE_EXAMPLES[0], FAILURE_EXAMPLES[0], NEUTRAL_EXAMPLES[0]])
    assert labels == ["practice", "failure", "neutral"]


def test_classify_never_needs_a_label_argument() -> None:
    """The classifier's only input is the sentence text: it never reads the
    truth file."""
    import inspect
    assert list(inspect.signature(classify_sentence).parameters) == ["text"]


# ---------------------------------------------------------------------------
# A small hand-built store: one interval risk (STUCK_PIPE, 17 1/2", Keldra
# Salt) with an affected well, three exposed clean wells and one unexposed
# well, plus incident reports for relevance and clean-well tests.
# ---------------------------------------------------------------------------


def _eowr(doc_id: str, well: str, text: str) -> Document:
    return Document(doc_id, "eowr", well, FIELD, "2024-03-01", "END OF WELL REPORT",
                    "END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. " + text + "\n")


def _ddr(doc_id: str, well: str, section: str, formation: str) -> Document:
    return Document(doc_id, "ddr", well, FIELD, "2024-01-01", "DAILY DRILLING REPORT",
                    "DAILY DRILLING REPORT\n", meta={"hole_section": section, "formation": formation})


def _incident(doc_id: str, well: str, code: str, section: str, formation: str,
              actions: list[str]) -> Document:
    numbered = "\n".join(f"  - {a}" for a in actions)
    text = (
        "INCIDENT REPORT\n"
        f"Operator: Quillfen Energy    Field: {FIELD}    Well: {well}\n"
        f"Classification: {code}   Lost time: 20.0 h\n"
        f"Depth: 1,400 m MD  Hole section: {section}  Formation: {formation}\n"
        "4. ROOT CAUSE\nUnder investigation.\n"
        f"5. CORRECTIVE ACTIONS\n{numbered}\n"
    )
    return Document(doc_id, "incident", well, FIELD, "2024-02-01", "INCIDENT REPORT", text)


AFFECTED_WELL = "WB-101"          # exposed, has the STUCK_PIPE event: excluded
CLEAN_WELL = "WB-102"             # exposed, no event, practice lesson: included
NEUTRAL_WELL = "WB-103"          # exposed, no event, neutral lesson: excluded (not practice)
UNEXPOSED_WELL = "WB-104"        # not exposed at all: excluded even though its lesson is practice
DUPLICATE_WELL = "WB-105"        # exposed, no event, a near-duplicate of WB-102's lesson

# A distinct valid practice sentence for the affected well, so a test of
# unrestricted mode (`clean_wells=None`) can tell its result apart from
# CLEAN_WELL's instead of the two colliding as an exact duplicate.
AFFECTED_WELL_LESSON = "Raise mud weight to at least 1.42 sg before drilling into Keldra Salt."
CLEAN_LESSON = "Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m."
# A near-duplicate: same words, reordered/reworded lightly (token Jaccard >= 0.8).
CLEAN_LESSON_DUPLICATE = ("Hold at least 1.42 sg across Keldra Salt, sweeping with saturated brine "
                          "every 250 m.")
NEUTRAL_LESSON = 'Keldra Salt was drilled at 1.35 sg in the 17 1/2" section.'
# Textually a valid, relevant practice sentence -- but its well was never
# exposed to the section and formation at all, so it must not qualify.
UNEXPOSED_LESSON = CLEAN_LESSON


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        _ddr(f"DDR-{AFFECTED_WELL}-001", AFFECTED_WELL, SECTION, FORMATION),
        _ddr(f"DDR-{CLEAN_WELL}-001", CLEAN_WELL, SECTION, FORMATION),
        _ddr(f"DDR-{NEUTRAL_WELL}-001", NEUTRAL_WELL, SECTION, FORMATION),
        _ddr(f"DDR-{UNEXPOSED_WELL}-001", UNEXPOSED_WELL, OTHER_SECTION, OTHER_FORMATION),
        _ddr(f"DDR-{DUPLICATE_WELL}-001", DUPLICATE_WELL, SECTION, FORMATION),
        _eowr(f"EOWR-{AFFECTED_WELL}", AFFECTED_WELL, AFFECTED_WELL_LESSON),   # practice text, but affected
        _eowr(f"EOWR-{CLEAN_WELL}", CLEAN_WELL, CLEAN_LESSON),
        _eowr(f"EOWR-{NEUTRAL_WELL}", NEUTRAL_WELL, NEUTRAL_LESSON),
        _eowr(f"EOWR-{UNEXPOSED_WELL}", UNEXPOSED_WELL, UNEXPOSED_LESSON),
        _eowr(f"EOWR-{DUPLICATE_WELL}", DUPLICATE_WELL, CLEAN_LESSON_DUPLICATE),
        # Same code, a different section/formation: excluded.
        _incident("INC-WB-201-01", "WB-201", CODE, OTHER_SECTION, OTHER_FORMATION,
                 ["Raise mud weight to at least 1.42 sg before drilling into Keldra Salt."]),
        # A different code entirely: excluded regardless of wording.
        _incident("INC-WB-202-01", "WB-202", "LOST_CIRCULATION", SECTION, FORMATION,
                 ["Hold at least 1.42 sg across Keldra Salt before the next trip."]),
    ])
    store.put_npt([
        NptEvent(f"DDR-{AFFECTED_WELL}-001", AFFECTED_WELL, FIELD, "2024-01-05", CODE, 23.5,
                 SECTION, FORMATION, 1402.0, 1.30, "Orrin-1", "String packed off."),
    ])
    return store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return _build_store(tmp_path)


def _scope(store: Store, **kwargs: object) -> MinerScope:
    defaults: dict[str, object] = {"code": CODE, "field_name": FIELD, "hole_section": SECTION,
                                   "formation": FORMATION}
    defaults.update(kwargs)
    return MinerScope(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# exposed_clean_wells: interval clean-well selection
# ---------------------------------------------------------------------------


def test_exposed_clean_wells_excludes_the_affected_well(store: Store) -> None:
    clean = exposed_clean_wells(store, FIELD, CODE, SECTION, FORMATION)
    assert AFFECTED_WELL not in clean


def test_exposed_clean_wells_excludes_a_well_never_exposed_to_the_section_and_formation(store: Store) -> None:
    clean = exposed_clean_wells(store, FIELD, CODE, SECTION, FORMATION)
    assert UNEXPOSED_WELL not in clean


def test_exposed_clean_wells_includes_every_exposed_well_with_no_event_there(store: Store) -> None:
    clean = exposed_clean_wells(store, FIELD, CODE, SECTION, FORMATION)
    assert clean == {CLEAN_WELL, NEUTRAL_WELL, DUPLICATE_WELL}


def test_exposed_clean_wells_is_empty_with_neither_section_nor_formation(store: Store) -> None:
    assert exposed_clean_wells(store, FIELD, CODE, "", "") == set()


# ---------------------------------------------------------------------------
# mine_mitigations: clean-well sourcing (a discovered Pattern's own population,
# via an explicit MinerScope.clean_wells, the way riskbrief.build_risk calls it)
# ---------------------------------------------------------------------------


def test_only_the_clean_wells_own_practice_sentences_qualify(store: Store) -> None:
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL, NEUTRAL_WELL, DUPLICATE_WELL}))
    found = mine_mitigations(store, scope)
    assert {m.well for m in found} == {CLEAN_WELL}   # NEUTRAL_WELL's lesson is not practice;
    # DUPLICATE_WELL's is a near-duplicate of CLEAN_WELL's, so it is dropped, not counted separately.
    assert AFFECTED_WELL not in {m.well for m in found}


def test_the_affected_well_is_excluded_even_though_its_lesson_reads_as_practice(store: Store) -> None:
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope)
    assert all(m.well != AFFECTED_WELL for m in found)


def test_incident_corrective_actions_are_never_restricted_to_the_clean_wells(store: Store) -> None:
    """An incident's corrective action is never restricted to the clean wells: a same-code,
    same-scope incident's corrective action counts regardless of whether its own well is in
    `clean_wells` -- here the affected well's own incident report."""
    store.put_documents([_incident(f"INC-{AFFECTED_WELL}-01", AFFECTED_WELL, CODE, SECTION, FORMATION,
                                  ["Run a caliper across the salt before the trip out for casing."])])
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope, limit=3)
    assert any(m.doc_id == f"INC-{AFFECTED_WELL}-01" for m in found)


def test_an_incident_of_a_different_code_never_qualifies(store: Store) -> None:
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope, limit=10)
    assert all(m.doc_id != "INC-WB-202-01" for m in found)


def test_an_incident_of_a_different_section_and_formation_never_qualifies(store: Store) -> None:
    """Relevance for an interval risk's corrective actions is structural (the incident's own
    recorded section and formation), not the risk's text alone."""
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope, limit=10)
    assert all(m.doc_id != "INC-WB-201-01" for m in found)


def test_unrestricted_scope_draws_on_every_wells_end_of_well_report(store: Store) -> None:
    """`clean_wells=None`: the miner still classifies and scores, it just does not
    require a known-avoided population (`qa.ask`'s fallback, see `test_ask.py`)."""
    scope = _scope(store, clean_wells=None)
    found = mine_mitigations(store, scope, limit=10)
    wells_found = {m.well for m in found}
    assert CLEAN_WELL in wells_found
    assert AFFECTED_WELL in wells_found            # its lesson reads as practice too


# ---------------------------------------------------------------------------
# Relevance: the code's keywords, and (for an interval risk) the formation or section
# ---------------------------------------------------------------------------


def test_a_practice_sentence_of_an_unrelated_code_is_not_relevant(store: Store) -> None:
    """A well can write a valid practice sentence for a different code; it must not be
    offered as a STUCK_PIPE mitigation."""
    store.put_documents([_eowr("EOWR-WB-301", "WB-301",
                              "Inspect and change mud pump fluid ends between wells; no pump NPT.")])
    store.put_documents([_ddr("DDR-WB-301-001", "WB-301", SECTION, FORMATION)])
    scope = _scope(store, clean_wells=frozenset({"WB-301"}))
    assert mine_mitigations(store, scope) == []


def test_an_interval_sentence_naming_neither_the_formation_nor_the_section_is_not_relevant(
        store: Store) -> None:
    store.put_documents([_ddr("DDR-WB-302-001", "WB-302", SECTION, FORMATION),
                        _eowr("EOWR-WB-302", "WB-302",
                             "Hold at least 1.42 sg and sweep with saturated brine every 250 m.")])
    scope = _scope(store, clean_wells=frozenset({"WB-302"}))
    assert mine_mitigations(store, scope) == []


def test_equipment_scope_needs_only_the_keyword_not_a_formation_or_section(store: Store) -> None:
    """For an equipment risk, the keyword match alone is enough, because the
    clean-well population is already scoped by rig or tool, not by a place in the well."""
    store.put_documents([
        _eowr("EOWR-WB-201", "WB-201",
             "Mud pump fluid ends on Orrin-2 were inspected and changed between wells; no pump NPT."),
    ])
    scope = MinerScope(code="RIG_REPAIR", clean_wells=frozenset({"WB-201"}))
    found = mine_mitigations(store, scope)
    assert [m.well for m in found] == ["WB-201"]


def test_equipment_scope_still_excludes_a_well_outside_the_clean_population(store: Store) -> None:
    store.put_documents([
        _eowr("EOWR-WB-201", "WB-201",
             "Mud pump fluid ends on Orrin-2 were inspected and changed between wells; no pump NPT."),
        _eowr("EOWR-WB-202", "WB-202",
             "Inspect and change mud pump fluid ends between wells rather than on the critical path."),
    ])
    scope = MinerScope(code="RIG_REPAIR", clean_wells=frozenset({"WB-201"}))
    found = mine_mitigations(store, scope)
    assert {m.well for m in found} == {"WB-201"}


# ---------------------------------------------------------------------------
# Near-duplicate removal and the 3-mitigation cap
# ---------------------------------------------------------------------------


def test_a_near_duplicate_lesson_does_not_count_as_a_second_mitigation(store: Store) -> None:
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL, DUPLICATE_WELL}))
    found = mine_mitigations(store, scope, limit=10)
    assert len([m for m in found if "sweep" in m.text.lower()]) == 1


def test_at_most_three_mitigations_are_ever_returned(store: Store) -> None:
    wells = {
        "WB-401": "Hold at least 1.42 sg across Keldra Salt on every connection through the section.",
        "WB-402": "Reduce flow rate across Keldra Salt and monitor pit volumes on every trip.",
        "WB-403": "Condition the mud across Keldra Salt before every trip out of hole.",
        "WB-404": "Weight up across Keldra Salt and record overpull on every connection.",
    }
    docs = []
    for well, text in wells.items():
        docs.append(_ddr(f"DDR-{well}-001", well, SECTION, FORMATION))
        docs.append(_eowr(f"EOWR-{well}", well, text))
    store.put_documents(docs)
    scope = _scope(store, clean_wells=frozenset(wells))
    found = mine_mitigations(store, scope, limit=3)
    assert len(found) == 3


def test_the_limit_argument_controls_the_cap(store: Store) -> None:
    wells = {
        "WB-501": "Hold at least 1.42 sg across Keldra Salt on every connection through the section.",
        "WB-502": "Reduce flow rate across Keldra Salt and monitor pit volumes on every trip.",
    }
    docs = []
    for well, text in wells.items():
        docs.append(_ddr(f"DDR-{well}-001", well, SECTION, FORMATION))
        docs.append(_eowr(f"EOWR-{well}", well, text))
    store.put_documents(docs)
    scope = _scope(store, clean_wells=frozenset(wells))
    assert len(mine_mitigations(store, scope, limit=1)) == 1
    assert len(mine_mitigations(store, scope, limit=2)) == 2


# ---------------------------------------------------------------------------
# Mitigation.to_dict: the shape qa.py and riskbrief.py both depend on
# ---------------------------------------------------------------------------


def test_mitigation_to_dict_carries_text_doc_id_well_and_source(store: Store) -> None:
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope)
    assert found
    d = found[0].to_dict()
    assert set(d) == {"text", "doc_id", "well", "source"}
    assert d["well"] == CLEAN_WELL
    assert d["doc_id"] == f"EOWR-{CLEAN_WELL}"
    assert d["source"] == "eowr"


# ---------------------------------------------------------------------------
# Mitigation.source: which candidate pool a sentence came from -- the caller (`qa.ask`) needs
# this to know which of its two headings a mitigation may truthfully sit under (see
# `test_ask.py`'s mitigation-heading tests).
# ---------------------------------------------------------------------------


def test_an_end_of_well_report_lesson_is_sourced_as_eowr(store: Store) -> None:
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope)
    assert [m.source for m in found] == ["eowr"]


def test_an_incident_corrective_action_is_sourced_as_incident(store: Store) -> None:
    store.put_documents([_incident(f"INC-{AFFECTED_WELL}-01", AFFECTED_WELL, CODE, SECTION, FORMATION,
                                  ["Run a caliper across the salt before the trip out for casing."])])
    scope = _scope(store, clean_wells=frozenset({CLEAN_WELL}))
    found = mine_mitigations(store, scope, limit=10)
    by_doc = {m.doc_id: m.source for m in found}
    assert by_doc[f"EOWR-{CLEAN_WELL}"] == "eowr"
    assert by_doc[f"INC-{AFFECTED_WELL}-01"] == "incident"
