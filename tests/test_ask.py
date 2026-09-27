"""`qa.ask`: abstention, figures from SQL over the NPT ledger, and figure sources."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import mini_workspace as mini
from wellbrief.models import Answer, Document, NptEvent
from wellbrief.narrate import NO_MATCH
from wellbrief.qa import (
    FIGURE_SOURCE_LIMIT,
    MITIGATIONS_HEADING_AVOIDED,
    MITIGATIONS_HEADING_GENERAL,
    MITIGATIONS_HEADING_INCIDENT,
    ask,
)
from wellbrief.search import Searcher
from wellbrief.store import Store


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Store, Searcher]:
    store = mini.store(tmp_path_factory.mktemp("ask"))
    return store, mini.searcher(store)


def answer(workspace: tuple[Store, Searcher], question: str, **kwargs: float) -> Answer:
    store, searcher = workspace
    return ask(question, store, searcher, **kwargs)


def sql(store: Store, query: str, *params: object) -> float:
    return float(store.conn.execute(query, params).fetchone()[0])


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_the_no_match_sentence_is_exactly_the_specified_one() -> None:
    assert NO_MATCH == "No records in this workspace match that question."


@pytest.mark.parametrize(("question", "line", "unmatched"), [
    ("How much NPT did ORD-199 have?", "Unmatched: well ORD-199", {"well": ["ORD-199"]}),
    ("How much NPT was recorded on the Yarrowvane field?", "Unmatched: field Yarrowvane",
     {"field": ["Yarrowvane"]}),
    ("Stuck pipe in the Qelvit Dolomite on Orrindale", "Unmatched: formation Qelvit Dolomite",
     {"formation": ["Qelvit Dolomite"]}),
    ('NPT in the 6" section on Orrindale', 'Unmatched: section 6"', {"section": ['6"']}),
    ("HSE stops on Orrindale", "Unmatched: code HSE_STOP", {"code": ["HSE_STOP"]}),
])
def test_an_unknown_name_abstains_with_the_exact_sentence(workspace: tuple[Store, Searcher], question: str,
                                                          line: str, unmatched: dict[str, list[str]]) -> None:
    a = answer(workspace, question)
    assert a.text == f"No records in this workspace match that question.\n{line}"
    assert a.abstained and a.unmatched == unmatched
    assert a.figures is None and a.figure_sources == []
    assert a.citations == [] and a.hits == [] and a.mitigations == []


@pytest.mark.parametrize(("question", "line"), [
    # Lost circulation exists in the workspace, but not on Orrindale.
    ("Lost circulation on Orrindale", "field Orrindale, code LOST_CIRCULATION"),
    # Stuck pipe exists on Orrindale, but not in the 8 1/2" section.
    ('Stuck pipe in the 8 1/2" section on Orrindale', 'field Orrindale, section 8 1/2", code STUCK_PIPE'),
    ("Stuck pipe at 2,650 m on Orrindale", "field Orrindale, code STUCK_PIPE, depth 2,550-2,750 m"),
    # ORD-104 has reports but no stuck pipe.
    ("Stuck pipe on ORD-104", "well ORD-104, code STUCK_PIPE"),
    # ORD-101 has no NPT near 3,000 m, and its reports carry no drilled interval.
    ("What happened at 3,000 m on ORD-101?", "well ORD-101, depth 2,900-3,100 m"),
])
def test_filters_that_match_nothing_abstain(workspace: tuple[Store, Searcher], question: str,
                                            line: str) -> None:
    a = answer(workspace, question)
    assert a.text.splitlines() == ["No records in this workspace match that question.",
                                   f"Unmatched: no document or NPT event matches {line}"]
    assert a.abstained and a.unmatched is not None and "filters" in a.unmatched
    assert a.figures is None and a.citations == [] and a.mitigations == []


def test_the_unmatched_filters_are_the_ones_applied(workspace: tuple[Store, Searcher]) -> None:
    a = answer(workspace, "Lost circulation on Orrindale")
    assert a.unmatched == {"filters": {"fields": ["Orrindale"], "codes": ["LOST_CIRCULATION"]}}


def test_the_abstention_json(workspace: tuple[Store, Searcher]) -> None:
    d = json.loads(json.dumps(answer(workspace, "How much NPT did ORD-199 have?").to_dict()))
    assert d["abstained"] is True
    assert d["unmatched"] == {"well": ["ORD-199"]}
    assert d["figures"] is None and d["figure_sources"] == [] and d["citations"] == []
    assert d["query_plan"]["unmatched"] == [{"kind": "well", "value": "ORD-199"}]


@pytest.mark.parametrize(("question", "line"), [
    ('Any cementing issues on the 9 5/8" casing on Orrindale?', None),
    ('Stuck pipe with 5" drill pipe on Orrindale', None),
    ('Losses while running 7" liner on Vessra South', None),
    ('Below the 13 3/8" shoe in the 12 1/4" hole on Orrindale, what stuck pipe?', '12 1/4"'),
])
def test_a_casing_size_does_not_make_the_answer_abstain(workspace: tuple[Store, Searcher], question: str,
                                                         line: str | None) -> None:
    a = answer(workspace, question)
    assert not a.abstained and a.citations and a.figures is not None and a.figures["event_count"] > 0
    assert a.query_plan["hole_sections"] == ([line] if line else [])


@pytest.mark.parametrize("question", [
    "What NPT was booked on ORD-104?",
    "How much downtime did ORD-104 have?",
])
def test_a_well_with_reports_but_no_npt_gets_zero_figures(workspace: tuple[Store, Searcher],
                                                          question: str) -> None:
    a = answer(workspace, question)
    assert not a.abstained and a.unmatched is None
    assert a.figures is not None
    assert (a.figures["event_count"], a.figures["total_hours"], a.figures["total_cost_usd"]) == (0, 0.0, 0.0)
    assert a.figure_sources == []
    assert a.text.splitlines()[0] == ("No NPT entry is recorded for well ORD-104: "
                                      "0.0 h of non-productive time.")
    assert a.hits and {h.document.well for h in a.hits} == {"ORD-104"}
    assert {c.doc_id for c in a.citations} <= {"DDR-ORD-104-001", "DDR-ORD-104-003", "DDR-ORD-104-011",
                                               "EOWR-ORD-104"}


def test_an_at_depth_with_no_npt_answers_from_the_reports_that_drilled_through(
        workspace: tuple[Store, Searcher]) -> None:
    a = answer(workspace, "What happened at 3,000 m on ORD-104?")
    assert not a.abstained and [h.doc_id for h in a.hits] == ["DDR-ORD-104-011"]
    assert a.figures is not None and a.figures["event_count"] == 0
    assert a.text.splitlines()[0] == ("No NPT entry is recorded for well ORD-104, depth 2,900-3,100 m: "
                                      "0.0 h of non-productive time.")


def test_documents_that_do_not_match_but_events_that_do_give_figures_without_abstaining(
        workspace: tuple[Store, Searcher]) -> None:
    # No cementing incident report exists, but the cementing ledger entry does.
    a = answer(workspace, "Incident reports on cementing on Orrindale")
    assert not a.abstained and a.hits == []
    assert a.figures is not None and a.figures["total_hours"] == 5.5
    assert "DDR-ORD-102-009" in {c.doc_id for c in a.citations}


# ---------------------------------------------------------------------------
# Figures from SQL
# ---------------------------------------------------------------------------

def test_code_figures_match_a_hand_computed_total_and_sql(workspace: tuple[Store, Searcher]) -> None:
    store, _ = workspace
    a = answer(workspace, "What did stuck pipe cost on Orrindale?", spread_rate=60_000.0)
    assert a.figures is not None
    # 23.5 + 6.0 + 10.2 + 0.8 + 2.5 + 1.5 + 4.0 over 7 reports on 3 wells.
    hand = 48.5
    expected = sql(store, "SELECT ROUND(SUM(hours), 1) FROM npt_events WHERE field_name = ? AND code = ?",
                   "Orrindale", "STUCK_PIPE")
    assert a.figures["total_hours"] == hand == expected
    assert a.figures["total_cost_usd"] == 121_250 == pytest.approx(hand / 24 * 60_000)
    assert (a.figures["event_count"], a.figures["well_count"], a.figures["report_count"]) == (7, 3, 7)
    assert a.figures["avoidable_share"] == 1.0
    assert a.figures["spread_rate"] == 60_000
    assert "by_code" not in a.figures          # one code asked: no breakdown by code
    assert a.figures["by_section"] == [
        {"section": '17 1/2"', "hours": 43.0, "events": 5},
        {"section": '26"', "hours": 4.0, "events": 1},
        {"section": '12 1/4"', "hours": 1.5, "events": 1},
    ]


def test_field_figures_break_down_by_code_well_and_section(workspace: tuple[Store, Searcher]) -> None:
    store, _ = workspace
    a = answer(workspace, "What share of the NPT on Orrindale was avoidable?")
    assert a.figures is not None
    total = sql(store, "SELECT SUM(hours) FROM npt_events WHERE field_name = 'Orrindale'")
    avoidable = sql(store, "SELECT SUM(hours) FROM npt_events WHERE field_name = 'Orrindale' "
                           "AND code NOT IN ('WAIT_ON_WEATHER', 'THIRD_PARTY_STANDBY', 'HSE_STOP')")
    # 48.5 stuck pipe + 4.0 weather + 12.0 tool + 3.1 instability + 5.5 cementing.
    assert a.figures["total_hours"] == 73.1 == round(total, 1)
    assert a.figures["avoidable_hours"] == 69.1 == round(avoidable, 1)
    assert a.figures["avoidable_share"] == round(69.1 / 73.1, 3) == round(avoidable / total, 3)
    assert [(r["code"], r["hours"]) for r in a.figures["by_code"]] == [
        ("STUCK_PIPE", 48.5), ("DOWNHOLE_TOOL_FAILURE", 12.0), ("CEMENT_ISSUE", 5.5),
        ("WAIT_ON_WEATHER", 4.0), ("WELLBORE_INSTABILITY", 3.1)]
    assert [(r["well"], r["hours"]) for r in a.figures["by_well"]] == [
        ("ORD-101", 45.5), ("ORD-102", 19.6), ("ORD-103", 8.0)]


def test_section_and_formation_figures(workspace: tuple[Store, Searcher]) -> None:
    section = answer(workspace, 'Total NPT hours in the 12.25" section on Orrindale?').figures
    formation = answer(workspace, "How many NPT hours were recorded in Keldra Salt on Orrindale?").figures
    assert section is not None and formation is not None
    assert section["total_hours"] == 19.0           # 12.0 + 5.5 + 1.5
    assert formation["total_hours"] == 50.1         # 23.5 + 6.0 + 4.0 + 10.2 + 3.1 + 0.8 + 2.5


def test_a_depth_around_ranks_but_does_not_filter_the_figures(workspace: tuple[Store, Searcher]) -> None:
    around = answer(workspace, "NPT around 2,660 m on Vessra South")
    at = answer(workspace, "NPT at 2,660 m on Vessra South")
    assert around.figures is not None and at.figures is not None
    assert around.figures["total_hours"] == 40.0    # 18.0 + 7.0 + 9.0 + 6.0 (the weather stop at 300 m)
    assert at.figures["total_hours"] == 34.0
    ranked = [h.doc_id for h in around.hits]
    assert ranked.index("DDR-VSS-202-003") > max(ranked.index(d) for d in
                                                 ("DDR-VSS-201-012", "DDR-VSS-202-011", "DDR-VSS-201-013"))


# ---------------------------------------------------------------------------
# Figure sources
# ---------------------------------------------------------------------------

def test_figure_sources_list_every_report_and_the_text_cites_the_five_largest(
        workspace: tuple[Store, Searcher]) -> None:
    a = answer(workspace, "How many hours of stuck pipe on Orrindale?")
    assert [(s["doc_id"], s["hours"]) for s in a.figure_sources] == [
        ("DDR-ORD-101-005", 23.5), ("DDR-ORD-102-004", 10.2), ("DDR-ORD-101-006", 6.0),
        ("DDR-ORD-103-008", 4.0), ("DDR-ORD-103-004", 2.5), ("DDR-ORD-103-007", 1.5),
        ("DDR-ORD-102-006", 0.8)]
    assert all(s["codes"] == ["STUCK_PIPE"] for s in a.figure_sources)
    line = next(ln for ln in a.text.splitlines() if ln.startswith("Largest contributing reports"))
    cited = re.findall(r"\[([A-Z0-9-]+)\]", line)
    assert len(cited) == FIGURE_SOURCE_LIMIT == 5
    assert cited == [s["doc_id"] for s in a.figure_sources[:5]]
    assert "(5 of 7)" in line
    # The cited ones are in the evidence pack with a verbatim quote; the others are not cited.
    store, _ = workspace
    quotes = {c.doc_id: c.quote for c in a.citations}
    for doc_id in cited:
        doc = store.get_document(doc_id)
        assert doc is not None and " ".join(quotes[doc_id].split()) in " ".join(doc.text.split())
    assert not a.citation_warnings


def test_a_figure_source_is_quoted_by_the_entry_the_figure_counts(workspace: tuple[Store, Searcher]) -> None:
    # DDR-ORD-101-006 carries a 6.0 h stuck pipe entry and a 4.0 h weather entry.
    a = answer(workspace, "How much time was lost to weather on Orrindale?")
    quotes = [c.quote for c in a.citations if c.doc_id == "DDR-ORD-101-006"]
    assert "Storm warning, operations suspended." in quotes
    assert "Continued to work the stuck string and jarred free." not in quotes


def test_the_ask_json_carries_plan_figures_sources_citations(workspace: tuple[Store, Searcher]) -> None:
    d = answer(workspace, "Stuck pipe in the 17 1/2\" section on ord-101").to_dict()
    assert {"query_plan", "figures", "figure_sources", "citations", "abstained", "unmatched",
           "mitigation_groups"} <= set(d)
    assert d["abstained"] is False and d["unmatched"] is None
    assert d["query_plan"]["wells"] == ["ORD-101"] and d["query_plan"]["hole_sections"] == ['17 1/2"']
    assert d["figures"]["total_hours"] == 29.5
    assert d["figures"]["filters"] == {"fields": [], "wells": ["ORD-101"], "codes": ["STUCK_PIPE"],
                                       "hole_sections": ['17 1/2"'], "formations": [], "depth_band_m": None}
    assert [s["doc_id"] for s in d["figure_sources"]] == ["DDR-ORD-101-005", "DDR-ORD-101-006"]


def test_two_named_fields_are_both_counted(workspace: tuple[Store, Searcher]) -> None:
    store, _ = workspace
    a = answer(workspace, "Compare the NPT on Orrindale and Vessra South")
    assert a.figures is not None and a.figures["filters"]["fields"] == ["Orrindale", "Vessra South"]
    total = sql(store, "SELECT ROUND(SUM(hours), 1) FROM npt_events")
    assert a.figures["total_hours"] == 113.1 == total     # 73.1 on Orrindale + 40.0 on Vessra South
    assert [(r["field"], r["hours"]) for r in a.figures["by_field"]] == [("Orrindale", 73.1),
                                                                          ("Vessra South", 40.0)]
    assert "By field: Orrindale 73.1 h, Vessra South 40.0 h." in a.text.splitlines()
    one = answer(workspace, "Compare the NPT on Orrindale").figures
    assert one is not None and "by_field" not in one


def test_no_citation_comes_from_another_field(workspace: tuple[Store, Searcher]) -> None:
    store, _ = workspace
    a = answer(workspace, "Lost returns on Vessra South")
    assert a.citations
    docs = [store.get_document(c.doc_id) for c in a.citations]
    assert {d.field_name for d in docs if d is not None} == {"Vessra South"} and None not in docs


# ---------------------------------------------------------------------------
# The mitigations heading: "Recorded mitigations on wells that
# avoided it" only when the miner drew them from a population it could
# compute as having avoided the named problem; "Lessons and recommendations
# from end-of-well reports" otherwise. Neither the shared mini_workspace
# fixture's daily reports carry parsed hole-section/formation metadata (they
# are hand-built Documents, not run through the real parser), so these use
# their own small stores where that metadata is set explicitly.
# ---------------------------------------------------------------------------

def _clean_population_store(tmp_path: Path) -> tuple[Store, Searcher]:
    """One field, one affected well and one clean well, exposed to the same
    section and formation: enough for `qa._mitigations_for` to compute a
    clean population directly from the ledger."""
    field, section, formation = "Mirren", '17 1/2"', "Keldra Salt"
    store = Store(tmp_path / "wellbrief.db")

    def ddr(well: str) -> Document:
        return Document(
            f"DDR-{well}-001", "ddr", well, field, "2024-01-05", "DAILY DRILLING REPORT",
            f"DAILY DRILLING REPORT\nOperator: Quillfen Energy    Field: {field}    Well: {well}\n",
            meta={"hole_section": section, "formation": formation},
        )

    store.put_documents([
        ddr("MRN-101"), ddr("MRN-102"),
        Document(
            "EOWR-MRN-102", "eowr", "MRN-102", field, "2024-03-01", "END OF WELL REPORT",
            "END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. Hold at least 1.42 sg across Keldra Salt "
            "and sweep with saturated brine every 250 m.\n",
        ),
    ])
    store.put_npt([NptEvent(
        "DDR-MRN-101-001", "MRN-101", field, "2024-01-05", "STUCK_PIPE", 23.5, section, formation,
        1402.0, 1.30, "Orrin-1", "String packed off at 1,402 m while pulling out of hole.",
    )])
    return store, mini.searcher(store)


def test_ask_uses_the_avoided_heading_when_a_clean_population_is_known(tmp_path: Path) -> None:
    store, searcher = _clean_population_store(tmp_path)
    a = ask("Stuck pipe in Keldra Salt on Mirren", store, searcher)
    assert not a.abstained
    assert [m["well"] for m in a.mitigations] == ["MRN-102"]
    assert f"{MITIGATIONS_HEADING_AVOIDED}:" in a.text
    assert f"{MITIGATIONS_HEADING_GENERAL}:" not in a.text


def _codeonly_store(tmp_path: Path) -> tuple[Store, Searcher]:
    """A code named with no hole section or formation: there is no place in
    the well to define an "avoided" population against (the clean-well rule
    needs a section or formation to know who was exposed), so `_mitigations_for`
    falls back to any relevant end of well report."""
    field, section, formation = "Kelso", '17 1/2"', "Keldra Salt"
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        Document(
            "DDR-KLS-101-001", "ddr", "KLS-101", field, "2024-01-05", "DAILY DRILLING REPORT",
            f"DAILY DRILLING REPORT\nOperator: Quillfen Energy    Field: {field}    Well: KLS-101\n",
            meta={"hole_section": section, "formation": formation},
        ),
        Document(
            "EOWR-KLS-101", "eowr", "KLS-101", field, "2024-03-01", "END OF WELL REPORT",
            "END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. Hold at least 1.42 sg across Keldra Salt "
            "and sweep with saturated brine every 250 m.\n",
        ),
    ])
    store.put_npt([NptEvent(
        "DDR-KLS-101-001", "KLS-101", field, "2024-01-05", "STUCK_PIPE", 23.5, section, formation,
        1402.0, 1.30, "Orrin-1", "String packed off at 1,402 m while pulling out of hole.",
    )])
    return store, mini.searcher(store)


def test_ask_uses_the_general_heading_when_the_question_names_no_place_in_the_well(tmp_path: Path) -> None:
    store, searcher = _codeonly_store(tmp_path)
    a = ask("What NPT was booked for stuck pipe on Kelso?", store, searcher)
    assert not a.abstained
    assert [m["well"] for m in a.mitigations] == ["KLS-101"]
    assert f"{MITIGATIONS_HEADING_GENERAL}:" in a.text
    assert f"{MITIGATIONS_HEADING_AVOIDED}:" not in a.text


def test_no_mitigations_means_no_mitigations_block_at_all(workspace: tuple[Store, Searcher]) -> None:
    a = answer(workspace, "What happened on ORD-104?")
    assert a.mitigations == []
    assert f"{MITIGATIONS_HEADING_AVOIDED}:" not in a.text
    assert f"{MITIGATIONS_HEADING_GENERAL}:" not in a.text


def _four_clean_wells_store(tmp_path: Path) -> tuple[Store, Searcher]:
    """One affected well and four clean wells, each with its own distinct
    qualifying practice lesson, so that more than 3 candidates qualify and
    the "at most 3" cap is the only thing holding the count
    down -- both in `Answer.mitigations` and in the rendered text."""
    field, section, formation = "Sarnov", '17 1/2"', "Keldra Salt"
    store = Store(tmp_path / "wellbrief.db")

    def ddr(well: str) -> Document:
        return Document(
            f"DDR-{well}-001", "ddr", well, field, "2024-01-05", "DAILY DRILLING REPORT",
            f"DAILY DRILLING REPORT\nOperator: Quillfen Energy    Field: {field}    Well: {well}\n",
            meta={"hole_section": section, "formation": formation},
        )

    def eowr(well: str, sentence: str) -> Document:
        return Document(
            f"EOWR-{well}", "eowr", well, field, "2024-03-01", "END OF WELL REPORT",
            f"END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. {sentence}\n",
        )

    clean_wells = ["SRN-102", "SRN-103", "SRN-104", "SRN-105"]
    sentences = [
        "Hold at least 1.42 sg across Keldra Salt; the string ran clean to bottom.",
        "Reduce flow rate on connections through Keldra Salt to control the wellbore below 1.40 sg.",
        "Spot a barite pill before tripping out of Keldra Salt at 1.41 sg.",
        "Sweep the hole with saturated brine every stand while drilling Keldra Salt at 1.43 sg.",
    ]
    store.put_documents([
        ddr("SRN-101"), *(ddr(w) for w in clean_wells),
        *(eowr(w, s) for w, s in zip(clean_wells, sentences, strict=True)),
    ])
    store.put_npt([NptEvent(
        "DDR-SRN-101-001", "SRN-101", field, "2024-01-05", "STUCK_PIPE", 23.5, section, formation,
        1402.0, 1.30, "Orrin-1", "String packed off at 1,402 m while pulling out of hole.",
    )])
    return store, mini.searcher(store)


def test_ask_caps_mitigations_at_three_even_when_more_qualify(tmp_path: Path) -> None:
    store, searcher = _four_clean_wells_store(tmp_path)
    a = ask("Stuck pipe in Keldra Salt on Sarnov", store, searcher)
    assert not a.abstained
    assert f"{MITIGATIONS_HEADING_AVOIDED}:" in a.text
    # Four clean wells each wrote a qualifying, non-duplicate practice lesson;
    # only 3 make it into the answer and into its rendered text.
    assert len(a.mitigations) == 3
    assert len({m["well"] for m in a.mitigations}) == 3
    mitigations_block = a.text.split(f"{MITIGATIONS_HEADING_AVOIDED}:")[1]
    assert mitigations_block.count("[EOWR-SRN-") == 3


def test_ask_shows_no_mitigations_when_the_question_names_two_codes(
        workspace: tuple[Store, Searcher]) -> None:
    # A single shared miner keys relevance on one code's keywords
    # (`miner._relevant`); a question naming more than one code leaves
    # `_mitigations_for`'s `code` empty and so yields no mitigations, a
    # documented scope limit (qa._mitigations_for), not a crash or a
    # silent partial answer -- the rest of the answer (figures, citations)
    # is unaffected.
    a = answer(workspace, "Compare stuck pipe and lost circulation NPT on Orrindale")
    assert not a.abstained
    assert len(a.query_plan["codes"]) == 2
    assert a.mitigations == []
    assert f"{MITIGATIONS_HEADING_AVOIDED}:" not in a.text
    assert f"{MITIGATIONS_HEADING_GENERAL}:" not in a.text


# ---------------------------------------------------------------------------
# An incident's corrective action is written by the well that HAD the
# problem, never one that avoided it, so it must never appear under
# `MITIGATIONS_HEADING_AVOIDED` -- even when the miner's clean-well scope also
# turns up a genuine end-of-well-report practice sentence in the same call.
# `MITIGATIONS_HEADING_INCIDENT` is its own heading for exactly that reason.
# ---------------------------------------------------------------------------

def _incident_text(field: str, well: str, section: str, formation: str, action: str) -> str:
    return (
        "INCIDENT REPORT\n"
        f"Operator: Quillfen Energy    Field: {field}    Well: {well}\n"
        "Classification: STUCK_PIPE   Lost time: 20.0 h\n"
        f"Depth: 1,400 m MD  Hole section: {section}  Formation: {formation}\n"
        "4. ROOT CAUSE\nUnder investigation.\n"
        f"5. CORRECTIVE ACTIONS\n  - {action}\n"
    )


def _mixed_source_store(tmp_path: Path) -> tuple[Store, Searcher]:
    """One affected well (a STUCK_PIPE event, and its own incident report with a same-code
    corrective action) and one clean well (a genuine end-of-well-report practice sentence): the
    miner finds one mitigation of each kind for the same scope."""
    field, section, formation = "Kestrel", '17 1/2"', "Keldra Salt"
    store = Store(tmp_path / "wellbrief.db")

    def ddr(well: str) -> Document:
        return Document(
            f"DDR-{well}-001", "ddr", well, field, "2024-01-05", "DAILY DRILLING REPORT",
            f"DAILY DRILLING REPORT\nOperator: Quillfen Energy    Field: {field}    Well: {well}\n",
            meta={"hole_section": section, "formation": formation},
        )

    store.put_documents([
        ddr("KST-101"), ddr("KST-102"),
        Document(
            "EOWR-KST-102", "eowr", "KST-102", field, "2024-03-01", "END OF WELL REPORT",
            "END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. Hold at least 1.42 sg across Keldra Salt "
            "and sweep with saturated brine every 250 m.\n",
        ),
        Document(
            "INC-KST-101-01", "incident", "KST-101", field, "2024-01-06", "INCIDENT REPORT",
            _incident_text(field, "KST-101", section, formation,
                           "Place the jars so that the string can be backed off above the stuck "
                           "point in the salt."),
        ),
    ])
    store.put_npt([NptEvent(
        "DDR-KST-101-001", "KST-101", field, "2024-01-05", "STUCK_PIPE", 23.5, section, formation,
        1402.0, 1.30, "Orrin-1", "String packed off at 1,402 m while pulling out of hole.",
    )])
    return store, mini.searcher(store)


def test_an_incident_corrective_action_never_shares_the_avoided_heading(tmp_path: Path) -> None:
    store, searcher = _mixed_source_store(tmp_path)
    a = ask("Stuck pipe in Keldra Salt on Kestrel", store, searcher)
    assert not a.abstained

    sources = {m["doc_id"]: m["source"] for m in a.mitigations}
    assert sources == {"EOWR-KST-102": "eowr", "INC-KST-101-01": "incident"}

    assert [g["heading"] for g in a.mitigation_groups] == [
        MITIGATIONS_HEADING_AVOIDED, MITIGATIONS_HEADING_INCIDENT,
    ]
    avoided, incident = a.mitigation_groups
    assert [m["doc_id"] for m in avoided["mitigations"]] == ["EOWR-KST-102"]
    assert [m["doc_id"] for m in incident["mitigations"]] == ["INC-KST-101-01"]

    # Reproduces the reported defect directly in the rendered text: everything between the two
    # headings is the avoided-heading's own group, and the incident's own document id must not
    # appear there.
    avoided_block = a.text.split(f"{MITIGATIONS_HEADING_AVOIDED}:", 1)[1]
    avoided_block = avoided_block.split(f"{MITIGATIONS_HEADING_INCIDENT}:", 1)[0]
    assert "INC-KST-101-01" not in avoided_block
    assert "EOWR-KST-102" in avoided_block
    incident_block = a.text.split(f"{MITIGATIONS_HEADING_INCIDENT}:", 1)[1]
    assert "INC-KST-101-01" in incident_block


def test_incident_only_mitigations_get_their_own_heading(tmp_path: Path) -> None:
    """No clean well exists in this scope at all (a single well in the store); the only
    mitigation the miner finds is that well's own incident corrective action, which must still
    get its own heading rather than folding into either end-of-well-report heading."""
    field, section, formation = "Petrel", '17 1/2"', "Keldra Salt"
    store = Store(tmp_path / "wellbrief.db")
    store.put_documents([
        Document(
            "DDR-PTL-101-001", "ddr", "PTL-101", field, "2024-01-05", "DAILY DRILLING REPORT",
            f"DAILY DRILLING REPORT\nOperator: Quillfen Energy    Field: {field}    Well: PTL-101\n",
            meta={"hole_section": section, "formation": formation},
        ),
        Document(
            "INC-PTL-101-01", "incident", "PTL-101", field, "2024-01-06", "INCIDENT REPORT",
            _incident_text(field, "PTL-101", section, formation,
                           "Place the jars so that the string can be backed off above the stuck "
                           "point in the salt."),
        ),
    ])
    store.put_npt([NptEvent(
        "DDR-PTL-101-001", "PTL-101", field, "2024-01-05", "STUCK_PIPE", 23.5, section, formation,
        1402.0, 1.30, "Orrin-1", "String packed off at 1,402 m while pulling out of hole.",
    )])
    a = ask("Stuck pipe in Keldra Salt on Petrel", store, mini.searcher(store))
    assert not a.abstained
    assert len(a.mitigation_groups) == 1
    assert a.mitigation_groups[0]["heading"] == MITIGATIONS_HEADING_INCIDENT
    assert [m["doc_id"] for m in a.mitigation_groups[0]["mitigations"]] == ["INC-PTL-101-01"]
    assert f"{MITIGATIONS_HEADING_AVOIDED}:" not in a.text
    assert f"{MITIGATIONS_HEADING_GENERAL}:" not in a.text
    assert f"{MITIGATIONS_HEADING_INCIDENT}:" in a.text
