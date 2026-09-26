"""The query planner: hole-section spellings, word-boundary synonyms, field phrasing, wells,
formations, depth, document types, unknown names and the candidate filter without fall-back."""

from __future__ import annotations

from pathlib import Path

import pytest

import mini_workspace as mini
from wellbrief.search import QueryPlan, Unmatched, candidate_documents, match_codes, plan_query
from wellbrief.store import Store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Store:
    return mini.store(tmp_path_factory.mktemp("planner"))


def plan(store: Store, question: str) -> QueryPlan:
    return plan_query(question, store)


# ---------------------------------------------------------------------------
# Hole sections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("spelling", "canonical"), [
    ('12 1/4"', '12 1/4"'),
    ("12-1/4 in", '12 1/4"'),
    ('12.25"', '12 1/4"'),
    ('12¼"', '12 1/4"'),
    ("12 1/4 in", '12 1/4"'),
    ('12-1/4"', '12 1/4"'),
    ("12.25 in", '12 1/4"'),
    ("12 1/4 inch", '12 1/4"'),
    ("12¼ in", '12 1/4"'),
    ("12 1/4”", '12 1/4"'),
    ('17-1/2"', '17 1/2"'),
    ("17 1/2 in", '17 1/2"'),
    ('8.5"', '8 1/2"'),
    ('8 1/2"', '8 1/2"'),
    ('26"', '26"'),
    ("26 in", '26"'),
])
def test_every_section_spelling_is_canonicalised(store: Store, spelling: str, canonical: str) -> None:
    p = plan(store, f"Stuck pipe in the {spelling} section on Orrindale")
    assert p.hole_sections == [canonical]
    assert p.unmatched == []


@pytest.mark.parametrize("question", [
    "Stuck pipe in the 12 1/4 section on Orrindale",
    "Stuck pipe in the 12-1/4 hole on Orrindale",
    "Stuck pipe in the 12.25 open hole on Orrindale",
    'Stuck pipe in the 12 1/4" hole below the 13 3/8" shoe on Orrindale',
    'Below the 13 3/8" shoe in the 12 1/4" hole on Orrindale',
    'Losses at the 12 1/4" casing point on Orrindale',
    'Hole section: 12 1/4", stuck pipe on Orrindale',
    'Stuck pipe in the 12 1/4" with casing wear on Orrindale',
])
def test_a_hole_section_among_other_sizes_and_without_a_unit(store: Store, question: str) -> None:
    p = plan(store, question)
    assert p.hole_sections == ['12 1/4"']
    assert p.unmatched == []


@pytest.mark.parametrize("question", [
    'Any cementing issues on the 9 5/8" casing on Orrindale?',
    'Problems running the 13 3/8" casing on Vessra South',
    'Problems running the 13 3/8" intermediate casing on Vessra South',
    'Losses while running 7" liner on Vessra South',
    'Stuck pipe with 5" drill pipe on Orrindale',
    'Washout in the 6 3/4" drill collars on Orrindale',
    'Leak on the 4 1/2" tubing on Orrindale',
    'Cementing the 26" conductor on Orrindale',
    'Stuck pipe with the 9 5/8" on Orrindale',
    "Which 2 in 10 wells had stuck pipe on Orrindale?",
    "Stuck pipe on 26 in Orrindale and 3 in Vessra South",
    "Stuck pipe in 12 wells on Orrindale",
])
def test_casing_liner_and_pipe_sizes_are_not_hole_sections(store: Store, question: str) -> None:
    p = plan(store, question)
    assert p.hole_sections == []
    assert p.unmatched == []


def test_a_hole_size_next_to_stuck_pipe_is_still_a_section(store: Store) -> None:
    assert plan(store, 'What were the 8 1/2" stuck pipe events on Vessra South?').hole_sections == ['8 1/2"']


def test_several_sections_are_kept_in_the_order_they_are_named(store: Store) -> None:
    p = plan(store, 'Stuck pipe in the 26" and 12 1/4" sections on Orrindale')
    assert p.hole_sections == ['26"', '12 1/4"']
    assert p.ledger_filters()["hole_section"] == ['26"', '12 1/4"']


@pytest.mark.parametrize("question", [
    'NPT in the 6" section on Orrindale',
    "NPT in the 6 in section on Orrindale",
    'NPT in the 6" hole on Orrindale',
    'NPT with hole size 6" on Orrindale',
    "NPT in the 5 7/8 section on Orrindale",
])
def test_a_hole_section_the_workspace_does_not_hold_is_unmatched(store: Store, question: str) -> None:
    p = plan(store, question)
    assert len(p.unmatched) == 1 and p.unmatched[0].kind == "section"
    assert p.hole_sections == [p.unmatched[0].value]


# ---------------------------------------------------------------------------
# NPT codes: synonyms on word boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "codes"), [
    ("Pack-off events on Orrindale wells", ["STUCK_PIPE"]),
    ("pack off in the salt", ["STUCK_PIPE"]),
    ("the string packed off twice", ["STUCK_PIPE"]),
    ("packoff while tripping", ["STUCK_PIPE"]),
    ("stuck_pipe hours", ["STUCK_PIPE"]),
    ("Lost returns in the carbonate", ["LOST_CIRCULATION"]),
    ("losses around 2,650 m", ["LOST_CIRCULATION"]),
    ("Mud pump problems", ["RIG_REPAIR"]),
    ("fluid end repairs on the rig", ["RIG_REPAIR"]),
    ("MWD failure in 12 1/4", ["DOWNHOLE_TOOL_FAILURE"]),
    ("MWD tool failures", ["DOWNHOLE_TOOL_FAILURE"]),
    ("HSE stops", ["HSE_STOP"]),
    ("Rig repairs in the 12.25 hole", ["RIG_REPAIR"]),
    ("the cement job", ["CEMENT_ISSUE"]),
    ("the plug was bumped late, a bumped plug", ["CEMENT_ISSUE"]),
    ("waited on high winds", ["WAIT_ON_WEATHER"]),
])
def test_synonyms_name_their_codes(text: str, codes: list[str]) -> None:
    assert match_codes(text) == codes


@pytest.mark.parametrize("text", [
    "Sidetrack through the casing window at 2,400 m",   # "window" is not "wind"
    "A plugged nozzle slowed the rate of penetration",  # "plugged" is not cementing
    "Did the plug hold?",                                # a bare "plug" is not cementing
    "How much time was lost on ORD-101?",                # "lost" is not lost circulation
    "Unstuck and fishless",                              # no partial words
])
def test_words_that_only_contain_a_synonym_name_no_code(text: str) -> None:
    assert match_codes(text) == []


def test_time_lost_to_weather_is_weather_only(store: Store) -> None:
    p = plan(store, "How much time was lost to weather on Vessra South?")
    assert p.codes == ["WAIT_ON_WEATHER"]
    assert p.npt


def test_a_code_with_no_entry_anywhere_in_the_workspace_is_unmatched(store: Store) -> None:
    p = plan(store, "HSE stops on Orrindale")
    assert p.codes == ["HSE_STOP"]
    assert p.unmatched == [Unmatched("code", "HSE_STOP")]


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("question", "field"), [
    ("Stuck pipe in the Orrindale field", "Orrindale"),
    ("Lost returns for field Vessra South", "Vessra South"),
    ("Field: Vessra South, lost returns", "Vessra South"),
    ("stuck pipe on orrindale", "Orrindale"),
    ("The Orrindale field, stuck pipe", "Orrindale"),
    ("Losses on Vessra South wells", "Vessra South"),
    ('stuck pipe on the "orrindale" field', "Orrindale"),
    ("stuck pipe on the 'Orrindale' field", "Orrindale"),
    ("Stuck pipe in the 'Vessra' field", "Vessra South"),
    ("Stuck pipe in the \u2018Orrindale\u2019 field", "Orrindale"),
])
def test_field_phrasing(store: Store, question: str, field: str) -> None:
    p = plan(store, question)
    assert p.fields == [field]
    assert p.unmatched == []


def test_every_named_field_is_kept(store: Store) -> None:
    p = plan(store, "Compare stuck pipe on Orrindale and Vessra South")
    assert p.fields == ["Orrindale", "Vessra South"]
    assert p.ledger_filters()["field_name"] == ["Orrindale", "Vessra South"]


def test_a_field_name_inside_a_formation_name_is_not_a_field(store: Store) -> None:
    p = plan(store, "MWD failures in the Orrindale Sand")
    assert p.fields == [] and p.formations == ["Orrindale Sand"]


@pytest.mark.parametrize("question", [
    "What NPT did we see in the field?",
    "Stuck pipe across the whole field",
    "Oil field practice for stuck pipe",
    "The field average for stuck pipe",
    "Which field had the most stuck pipe?",
    "Stuck pipe in Keldra Salt field-wide",
    "Stuck pipe in the Keldra Salt field interval",
])
def test_ordinary_words_before_field_are_not_field_names(store: Store, question: str) -> None:
    p = plan(store, question)
    assert p.fields == []
    assert p.unmatched == []


@pytest.mark.parametrize("question", [
    "What did the Field Engineer report on ORD-101?",
    "Stuck pipe per the Field Engineer's report on ORD-101",
    "What happened on the Orrindale field Monday?",
    "Orrindale field Wellbore Instability events",
    "Orrindale field Stuck Pipe summary",
    "The Field Service Engineer on Orrindale reported losses",
    "Drilling Salt problems on Orrindale",
    "Reactive Shale problems on Orrindale",
    "Highly Reactive Shale on Orrindale",
    "Formation Integrity results on Orrindale",
    "Orrindale field Rigsite Visit notes",
    "What did the driller's report say about the 'Keldra' formation on Orrindale?",
])
def test_capitalised_ordinary_words_are_not_unknown_names(store: Store, question: str) -> None:
    p = plan(store, question)
    assert p.unmatched == []
    assert set(p.fields) <= {"Orrindale"} and set(p.formations) <= {"Keldra Salt"}


@pytest.mark.parametrize(("question", "name"), [
    ("How much NPT was recorded on the Yarrowvane field?", "Yarrowvane"),
    ("Stuck pipe on Zelmar field", "Zelmar"),
    ("Stuck pipe for field Zelmar", "Zelmar"),
    ('stuck pipe in the "zelmar" field', "zelmar"),
    ("Stuck pipe in the 'Zelmar' field", "Zelmar"),
    ("Stuck pipe on field Zelmar, 12 1/4\" section", "Zelmar"),
    ("Stuck pipe on field Zelmar wells", "Zelmar"),
    ("Stuck pipe on Zelmar's field", "Zelmar"),
])
def test_an_unknown_field_name_is_unmatched(store: Store, question: str, name: str) -> None:
    p = plan(store, question)
    assert p.fields == []
    assert p.unmatched == [Unmatched("field", name)]


# ---------------------------------------------------------------------------
# Wells, formations, depth, document types, NPT questions
# ---------------------------------------------------------------------------

def test_well_ids_in_any_case_are_upper_cased(store: Store) -> None:
    p = plan(store, "What non-productive time was booked on ord-102 and Vss-201?")
    assert p.wells == ["ORD-102", "VSS-201"]
    assert p.unmatched == []


def test_an_unknown_well_is_unmatched(store: Store) -> None:
    p = plan(store, "How much NPT did ORD-199 have?")
    assert p.wells == ["ORD-199"]
    assert p.unmatched == [Unmatched("well", "ORD-199")]


def test_rig_and_tool_names_are_not_wells(store: Store) -> None:
    p = plan(store, "Rig repairs on Orrin-1 and Vessra-3 with the PJ-3 tool")
    assert p.wells == [] and p.unmatched == []


@pytest.mark.parametrize("question", [
    "What were the top-10 NPT events on Orrindale?",
    "Top-10 NPT events on Orrindale",
    "Covid-19 rules on Orrindale",
])
def test_an_id_in_running_text_is_not_a_well(store: Store, question: str) -> None:
    p = plan(store, question)
    assert p.wells == [] and p.unmatched == []


@pytest.mark.parametrize(("question", "well"), [
    ("How much NPT did ord-199 have?", "ORD-199"),     # a prefix the workspace holds, any case
    ("How much NPT did XYZ-101 have?", "XYZ-101"),     # another prefix, written in capitals
])
def test_an_unknown_well_id_that_names_a_well_is_unmatched(store: Store, question: str, well: str) -> None:
    p = plan(store, question)
    assert p.wells == [well] and p.unmatched == [Unmatched("well", well)]


@pytest.mark.parametrize(("question", "formation"), [
    ("Show wellbore instability in Keldra Salt on Orrindale", "Keldra Salt"),
    ("instability in keldra salt", "Keldra Salt"),
    ("tight hole in the Keldra formation", "Keldra Salt"),
    ("Lost returns in Vessra Carbonate", "Vessra Carbonate"),
])
def test_known_formations(store: Store, question: str, formation: str) -> None:
    p = plan(store, question)
    assert p.formations == [formation]
    assert p.unmatched == []


@pytest.mark.parametrize(("question", "name"), [
    ("Stuck pipe in the Qelvit Dolomite on Orrindale", "Qelvit Dolomite"),
    ("Stuck pipe in the Qelvit formation on Orrindale", "Qelvit"),
    ("Stuck pipe in the 'Qelvit' formation on Orrindale", "Qelvit"),
    ("Losses in the Fractured Qelvit Limestone on Orrindale", "Qelvit Limestone"),
])
def test_an_unknown_formation_is_unmatched(store: Store, question: str, name: str) -> None:
    p = plan(store, question)
    assert p.formations == []
    assert p.unmatched == [Unmatched("formation", name)]


def test_a_capitalised_ordinary_word_before_a_rock_word_is_no_formation(store: Store) -> None:
    p = plan(store, "Show Salt creep problems on Orrindale")
    assert p.formations == [] and p.unmatched == []


def test_every_named_formation_is_kept(store: Store) -> None:
    p = plan(store, "Tight hole in Dovrin Shale and Keldra Salt on Orrindale")
    assert p.formations == ["Dovrin Shale", "Keldra Salt"]


@pytest.mark.parametrize(("question", "depth", "exact"), [
    ("Losses at 2,650 m on Vessra South", 2650.0, True),
    ("Losses at 2650m on Vessra South", 2650.0, True),
    ("Losses around 2,650 m on Vessra South", 2650.0, False),
    ("Losses near 2,650 m on Vessra South", 2650.0, False),
    ("Losses 2650 m on Vessra South", 2650.0, False),
    ("Losses at around 2,650 m on Vessra South", 2650.0, False),
])
def test_depth_is_a_preference_unless_the_question_says_at(store: Store, question: str, depth: float,
                                                           exact: bool) -> None:
    p = plan(store, question)
    assert p.depth_m == depth and p.depth_exact is exact
    assert p.depth_band == (2550.0, 2750.0)
    assert ("depth_min" in p.ledger_filters()) is exact


@pytest.mark.parametrize(("question", "types"), [
    ("What do the lessons say about Keldra Salt?", ["eowr"]),
    ("What do the end of well reports recommend?", ["eowr"]),
    ("EOWR lessons on stuck pipe", ["eowr"]),
    ("Incident reports on stuck pipe", ["incident"]),
    ("What was the root cause of the pack-off?", ["incident"]),
    ("Show the daily reports for ORD-101", ["ddr"]),
    ("DDR entries for ORD-101", ["ddr"]),
    ("What caused stuck pipe on Orrindale?", []),
])
def test_document_type_cues(store: Store, question: str, types: list[str]) -> None:
    assert plan(store, question).doc_types == types


@pytest.mark.parametrize(("question", "npt"), [
    ("What NPT was booked around 1,850 m on Orrindale?", True),
    ("What non-productive time was booked on ORD-101?", True),
    ("How much downtime did ORD-101 have?", True),
    ("What do the end of well reports recommend for Keldra Salt?", False),
])
def test_general_npt_questions(store: Store, question: str, npt: bool) -> None:
    assert plan(store, question).npt is npt


def test_the_plan_is_serialisable(store: Store) -> None:
    d = plan(store, 'Stuck pipe at 1,400 m in the 17 1/2" section of ord-101 on Orrindale').to_dict()
    assert d["fields"] == ["Orrindale"] and d["wells"] == ["ORD-101"] and d["hole_sections"] == ['17 1/2"']
    assert d["codes"] == ["STUCK_PIPE"] and d["depth_mode"] == "filter" and d["depth_band_m"] == [1300, 1500]
    assert d["unmatched"] == [] and "field=Orrindale" in d["summary"]


# ---------------------------------------------------------------------------
# Structured filters before retrieval
# ---------------------------------------------------------------------------

def test_ledger_filters_keep_eowrs_and_matching_incidents_of_the_matched_wells(store: Store) -> None:
    allowed = candidate_documents(plan(store, "Stuck pipe on Orrindale"), store)
    assert allowed == {
        "DDR-ORD-101-005", "DDR-ORD-101-006", "DDR-ORD-102-004", "DDR-ORD-102-006", "DDR-ORD-103-004",
        "DDR-ORD-103-007", "DDR-ORD-103-008", "EOWR-ORD-101", "EOWR-ORD-102", "EOWR-ORD-103",
        "INC-ORD-101-01",
    }


def test_an_incident_of_another_code_is_not_a_candidate(store: Store) -> None:
    allowed = candidate_documents(plan(store, "How much time was lost to weather on Orrindale?"), store)
    assert allowed == {"DDR-ORD-101-006", "EOWR-ORD-101"}


def test_a_filter_that_matches_nothing_does_not_fall_back_to_everything(store: Store) -> None:
    # Cementing entries exist on Orrindale, incident reports exist, but no cementing incident report.
    p = plan(store, "Incident reports on cementing on Orrindale")
    assert p.doc_types == ["incident"] and p.codes == ["CEMENT_ISSUE"]
    assert candidate_documents(p, store) == set()


def test_a_general_npt_question_about_a_well_keeps_its_reports_with_npt(store: Store) -> None:
    allowed = candidate_documents(plan(store, "What NPT was booked on ord-102?"), store)
    assert allowed == {"DDR-ORD-102-004", "DDR-ORD-102-005", "DDR-ORD-102-006", "DDR-ORD-102-009",
                       "EOWR-ORD-102"}


def test_an_at_depth_filters_and_an_around_depth_does_not(store: Store) -> None:
    at = candidate_documents(plan(store, "NPT at 2,660 m on Vessra South"), store)
    around = candidate_documents(plan(store, "NPT around 2,660 m on Vessra South"), store)
    assert at == {"DDR-VSS-201-012", "DDR-VSS-201-013", "DDR-VSS-202-011", "EOWR-VSS-201", "EOWR-VSS-202",
                  "INC-VSS-201-01"}
    assert around is not None and "DDR-VSS-202-003" in around and at < around


def test_a_general_npt_question_about_a_well_without_npt_keeps_its_documents(store: Store) -> None:
    # ORD-104 has daily reports and an end-of-well report but no NPT entry.
    for question in ("What NPT was booked on ORD-104?", "How much downtime did ORD-104 have?"):
        assert candidate_documents(plan(store, question), store) == {
            "DDR-ORD-104-001", "DDR-ORD-104-003", "DDR-ORD-104-011", "EOWR-ORD-104"}


def test_a_code_on_a_well_without_that_code_still_matches_nothing(store: Store) -> None:
    assert candidate_documents(plan(store, "Stuck pipe on ORD-104"), store) == set()


def test_an_at_depth_without_npt_there_keeps_the_reports_that_drilled_through_it(store: Store) -> None:
    at_3000 = candidate_documents(plan(store, "What happened at 3,000 m on ORD-104?"), store)
    assert at_3000 == {"DDR-ORD-104-011"}
    assert candidate_documents(plan(store, "What NPT was booked at 1,300 m on ORD-104?"), store) == {
        "DDR-ORD-104-003"}
    # The reports of ORD-101 carry no drilled interval, and none of its NPT lies near 3,000 m.
    assert candidate_documents(plan(store, "What happened at 3,000 m on ORD-101?"), store) == set()


def test_without_structured_filters_every_document_is_a_candidate(tmp_path: Path) -> None:
    s = mini.store(tmp_path)
    assert candidate_documents(plan_query("What went wrong?", s), s) is None
