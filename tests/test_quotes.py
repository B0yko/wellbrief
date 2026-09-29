"""Evidence quotes: verbatim raw spans, the passage chosen per document type, and never a label line.

The last tests ask every question of the extended evaluation suite on the
default corpus and check the quotes of every answer: none is a header or
`Label : value` line (an earlier version quoted `Hole section : 17 1/2"` six times in
one answer), none is the footer, each is verbatim, and each comes from the
passage the rule names for its document type.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import mini_workspace as mini
from wellbrief import parse
from wellbrief.config import FIGURE_SOURCE_LIMIT, SYNTHETIC_FOOTER
from wellbrief.corpus import SEED, build_corpus, write_corpus
from wellbrief.evals.cases import load_suite
from wellbrief.ingest import ingest, load_corpus_dir
from wellbrief.models import Answer, Document, NptEvent
from wellbrief.qa import ask
from wellbrief.quotes import best_passage, evidence_quote, passages, raw_span, verbatim_quote
from wellbrief.search import Searcher
from wellbrief.store import Store
from wellbrief.text import tokenize
from wellbrief.workspace import build_searcher

CASES = Path(__file__).resolve().parents[1] / "evals" / "cases"

DDR = f"""DAILY DRILLING REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Rig: Orrin-2    Report No: 014    Date: 2023-05-02
Report period: 06:00 - 06:00

DEPTH
  Depth at start        : 1,610 m MD
  Depth at end          : 1,642 m MD
  Hole section          : 17 1/2"
  Formation at TD       : Dovrin Shale (reactive shale)

OPERATIONS SUMMARY (24 h)
  06:00-08:00  Drilled 17 1/2" hole from 1,610 m to 1,642 m.
  08:00-23:12  NPT STUCK_PIPE 15.2 h: String packed off at 1,402 m while pulling out of hole for the
               13 3/8" intermediate casing. Salt creep at 1.33 sg across Keldra Salt.
  23:12-03:36  NPT WAIT_ON_WEATHER 4.4 h: Storm warning, operations suspended.
  03:36-06:00  Circulated the hole clean at 1,642 m.

TIME BREAKDOWN
  Productive time       : 4.4 h
  Non-productive time   : 19.6 h

NPT DETAIL
  Code                  : STUCK_PIPE
  Hours                 : 15.2
  Depth                 : 1,402 m MD
  Formation             : Keldra Salt
  Description           : String packed off at 1,402 m while pulling out of hole for the
                          13 3/8" intermediate casing. Salt creep at 1.33 sg across Keldra Salt.

  Code                  : WAIT_ON_WEATHER
  Hours                 : 4.4
  Depth                 : 1,642 m MD
  Formation             : Dovrin Shale
  Description           : Storm warning, operations suspended.

REMARKS
  Casing programme for this section: 13 3/8" intermediate casing.
  Next 24 h: work the stuck string free.

{SYNTHETIC_FOOTER}
"""

EOWR = f"""END OF WELL REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Days on well: 35    Total depth: 3,154 m MD

1. WELL SUMMARY
  ORD-150 was drilled as a development well on Orrindale to 3,154 m MD.
  MWD: Parvane Downhole PJ-5 in every section.

2. TIME ANALYSIS
  Total time on well    : 840.0 h
  Non-productive time   : 89.6 h (10.7 %)

4. LESSONS LEARNED
  1. Keldra Salt was drilled at 1.33 sg in the 17 1/2" section; tight hole from salt creep was
     recorded on 2 daily reports.
  2. Cement jobs went to programme with no notable deviation.

5. RECOMMENDATIONS FOR FUTURE WELLS
  - Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m.

{SYNTHETIC_FOOTER}
"""

INCIDENT = f"""WELL OPERATIONS INCIDENT REPORT
Operator: Quillfen Energy    Field: Orrindale    Well: ORD-150
Classification: STUCK_PIPE    Lost time: 15.2 h

1. LOCATION
  Depth: 1,402 m MD    Hole section: 17 1/2"    Formation: Keldra Salt

2. SEQUENCE OF EVENTS
  String packed off at 1,402 m while pulling out of hole for the 13 3/8" intermediate casing. Salt
  creep at 1.33 sg across Keldra Salt.
  Lost time 15.2 h. No injuries and no environmental release.

4. ROOT CAUSE
  Mud weight below the value required to hold the salt in gauge.

5. CORRECTIVE ACTIONS
  - Raise mud weight to at least 1.42 sg before drilling into Keldra Salt.
  - Pump out of hole through the salt interval rather than pulling on elevators, and record the
    overpull at every stand.

{SYNTHETIC_FOOTER}
"""


def squash(text: str) -> str:
    return " ".join(text.split())


def verbatim(quote: str, source: str) -> bool:
    return bool(quote) and squash(quote) in squash(source)


def doc(text: str, doc_type: str, doc_id: str = "DOC-1") -> Document:
    return Document(doc_id=doc_id, doc_type=doc_type, well="ORD-150", field_name="Orrindale",
                    date="2023-05-02", title="", text=text)


def entry(code: str, hours: float, depth: float, description: str) -> NptEvent:
    return NptEvent("DOC-1", "ORD-150", "Orrindale", "2023-05-02", code, hours, '17 1/2"', "Keldra Salt",
                    depth, 1.33, "Orrin-2", description)


# ---------------------------------------------------------------------------
# Verbatim spans of the raw text
# ---------------------------------------------------------------------------

def test_a_normalised_sentence_is_quoted_as_its_raw_span() -> None:
    raw = ("4. LESSONS LEARNED\n"
           "  1. Keldra Salt held gauge at 1.45 sg in the 17\u00bd\" section \u2013 no pack\u2019off.\n")
    parsed = parse.parse_eowr(raw)["lessons"][0]
    assert parsed == "Keldra Salt held gauge at 1.45 sg in the 17 1/2\" section - no pack'off."
    assert parsed not in raw                      # the parser's copy is not in the source
    quote = verbatim_quote(raw, parsed)
    assert quote == "Keldra Salt held gauge at 1.45 sg in the 17\u00bd\" section \u2013 no pack\u2019off."
    assert quote in raw


def test_a_combining_accent_is_quoted_as_written() -> None:
    raw = "Remarks: cafe\u0301 area cleared.\n"      # "e" + combining acute; NFKC composes it
    assert verbatim_quote(raw, "caf\u00e9 area") == "cafe\u0301 area"


def test_continuation_lines_are_joined_into_one_line() -> None:
    quote = verbatim_quote(DDR, "String packed off at 1,402 m while pulling out of hole for the 13 3/8\" "
                                "intermediate casing.")
    assert quote == ("String packed off at 1,402 m while pulling out of hole for the 13 3/8\" "
                     "intermediate casing.")


def test_a_span_is_searched_at_or_after_the_given_position() -> None:
    needle = "Storm warning, operations suspended."
    first = raw_span(DDR, needle)
    later = raw_span(DDR, needle, DDR.index("NPT DETAIL"))
    assert first is not None and later is not None
    assert first[0] < DDR.index("NPT DETAIL") < later[0]
    assert DDR[later[0]:later[1]] == needle


def test_text_that_is_not_in_the_document_has_no_quote() -> None:
    assert verbatim_quote(DDR, "Hold at least 1.42 sg") is None
    assert verbatim_quote(DDR, "   ") is None
    assert raw_span("", "anything") is None


def test_a_long_passage_is_cut_at_a_word_boundary_and_stays_verbatim() -> None:
    raw = "REMARKS\n  " + " ".join(f"word{i}" for i in range(200)) + ".\n"
    quote = best_passage(raw, "ddr", [])
    assert len(quote) <= 400 and verbatim(quote, raw)
    assert quote.split()[-1] in raw.split()       # cut between words


# ---------------------------------------------------------------------------
# Layout: content passages, labels, headings, the footer
# ---------------------------------------------------------------------------

def test_ddr_passages_by_role() -> None:
    found = passages(DDR, "ddr")
    content = [(p.kind, p.text) for p in found if p.kind and p.role == "content"]
    assert content == [
        ("operations", '06:00-08:00 Drilled 17 1/2" hole from 1,610 m to 1,642 m.'),
        ("operations", "08:00-23:12 NPT STUCK_PIPE 15.2 h: String packed off at 1,402 m while pulling out of "
                       'hole for the 13 3/8" intermediate casing. Salt creep at 1.33 sg across Keldra Salt.'),
        ("operations", "23:12-03:36 NPT WAIT_ON_WEATHER 4.4 h: Storm warning, operations suspended."),
        ("operations", "03:36-06:00 Circulated the hole clean at 1,642 m."),
    ]
    labels = {p.text for p in found if p.role == "label"}
    assert {'Hole section : 17 1/2"', "Depth at end : 1,642 m MD", "Code : STUCK_PIPE",
            "Operator: Quillfen Energy Field: Orrindale Well: ORD-150",
            'Casing programme for this section: 13 3/8" intermediate casing.',
            "Next 24 h: work the stuck string free."} <= labels
    headings = {p.text for p in found if p.role == "heading"}
    assert headings >= {"DAILY DRILLING REPORT", "OPERATIONS SUMMARY (24 h)", "NPT DETAIL", "REMARKS"}
    assert all(SYNTHETIC_FOOTER not in p.text for p in found)
    assert all(verbatim(p.text, DDR) for p in found)


def test_a_label_line_is_not_quoted_when_a_content_line_exists() -> None:
    # `Hole section : 17 1/2"` shares three terms with the question, the drilling line two.
    quote = best_passage(DDR, "ddr", tokenize('hole section 17 1/2"'))
    assert quote == '06:00-08:00 Drilled 17 1/2" hole from 1,610 m to 1,642 m.'
    # "stuck string" is only in a label-shaped remark; the best operations line is quoted instead.
    quote = best_passage(DDR, "ddr", tokenize("work the stuck string free"))
    assert quote.startswith("08:00-23:12 NPT STUCK_PIPE")


def test_ties_go_to_the_earlier_passage() -> None:
    assert best_passage(DDR, "ddr", tokenize("nothing matches this")) == (
        '06:00-08:00 Drilled 17 1/2" hole from 1,610 m to 1,642 m.')
    assert best_passage(DDR, "ddr", tokenize("circulated or drilled")).startswith("06:00-08:00 Drilled")


def test_a_document_with_only_labels_falls_back_to_a_label_and_never_to_the_footer() -> None:
    text = (f"DEPTH\n  Hole section          : 12 1/4\"\n  Depth at end          : 2,100 m MD\n\n"
            f"{SYNTHETIC_FOOTER}\n")
    assert best_passage(text, "ddr", tokenize("depth at end")) == "Depth at end : 2,100 m MD"
    headings_only = f"NPT DETAIL\n{SYNTHETIC_FOOTER}\nText after the footer.\n"
    assert best_passage(headings_only, "ddr", []) == "NPT DETAIL"
    assert best_passage(f"{SYNTHETIC_FOOTER}\n", "ddr", ["synthetic", "fictional"]) == ""


def test_an_unknown_document_type_quotes_its_content_lines() -> None:
    memo = "The mud pump fluid end was changed twice this week."
    assert best_passage(f"SITE MEMO\nDate: 2023-05-02\n{memo}\n", "other", tokenize("mud pump")) == memo


def test_eowr_quotes_a_lesson_or_recommendation_without_its_marker() -> None:
    quote = best_passage(EOWR, "eowr", tokenize("Keldra Salt tight hole"))
    assert quote == ('Keldra Salt was drilled at 1.33 sg in the 17 1/2" section; tight hole from salt creep '
                     "was recorded on 2 daily reports.")
    # The question asks what the reports recommend: the recommendation, not the lesson.
    quote = best_passage(EOWR, "eowr", tokenize("What do the end of well reports recommend for Keldra Salt?"))
    assert quote == "Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m."
    # Never the summary or the time analysis while a lesson exists.
    quote = best_passage(EOWR, "eowr", tokenize("MWD Parvane PJ-5 non-productive time"))
    assert quote in {p.text for p in passages(EOWR, "eowr") if p.kind}


def test_incident_quotes_a_sentence_of_the_sequence_the_root_cause_or_a_corrective_action() -> None:
    kinds = {(p.kind, p.text) for p in passages(INCIDENT, "incident") if p.kind}
    assert ("sequence", "Salt creep at 1.33 sg across Keldra Salt.") in kinds
    assert ("sequence", "Lost time 15.2 h.") in kinds
    assert ("root_cause", "Mud weight below the value required to hold the salt in gauge.") in kinds
    assert ("corrective_action", "Pump out of hole through the salt interval rather than pulling on "
                                 "elevators, and record the overpull at every stand.") in kinds
    assert best_passage(INCIDENT, "incident", tokenize("What caused the stuck pipe?")) == (
        "Mud weight below the value required to hold the salt in gauge.")
    assert best_passage(INCIDENT, "incident", tokenize("pack-off pulling out of hole")) == (
        'String packed off at 1,402 m while pulling out of hole for the 13 3/8" intermediate casing.')
    location = next(p for p in passages(INCIDENT, "incident") if p.text.startswith("Depth: 1,402 m MD"))
    assert location.role == "label"


def test_a_topic_colon_line_inside_a_content_section_is_content() -> None:
    text = ("1. WELL SUMMARY\n  Drilled to TD without incident.\n\n"
            "4. LESSONS LEARNED\n  1. Keldra Salt: hold at least 1.42 sg across the salt.\n\n"
            "5. RECOMMENDATIONS FOR FUTURE WELLS\n  - Mud weight: raise to 1.42 sg before the salt.\n\n"
            f"{SYNTHETIC_FOOTER}\n")
    roles = {p.text: (p.kind, p.role) for p in passages(text, "eowr")}
    assert roles["Keldra Salt: hold at least 1.42 sg across the salt."] == ("lesson", "content")
    assert roles["Mud weight: raise to 1.42 sg before the salt."] == ("recommendation", "content")
    assert best_passage(text, "eowr", tokenize("Keldra Salt")) == (
        "Keldra Salt: hold at least 1.42 sg across the salt.")
    assert best_passage(text, "eowr", tokenize("raise the mud weight")) == (
        "Mud weight: raise to 1.42 sg before the salt.")
    # A sentence of the sequence of events shaped the same way is content too.
    incident = ("2. SEQUENCE OF EVENTS\n  Continued the round trip: replaced the MWD module and ran back to "
                "bottom.\n\n4. ROOT CAUSE\n  Tool operated above its temperature rating.\n")
    assert best_passage(incident, "incident", tokenize("round trip MWD module")) == (
        "Continued the round trip: replaced the MWD module and ran back to bottom.")


def test_the_planning_notes_of_the_remarks_stay_label_lines() -> None:
    labels = {p.text for p in passages(DDR, "ddr") if p.kind == "remarks" and p.role == "label"}
    assert labels == {'Casing programme for this section: 13 3/8" intermediate casing.',
                      "Next 24 h: work the stuck string free."}
    # An item is never a label line, even when it starts with a listed label.
    text = "REMARKS\n  - Next 24 h: rig down.\n"
    assert [(p.kind, p.role, p.text) for p in passages(text, "ddr") if p.kind] == [
        ("remarks", "content", "Next 24 h: rig down.")]


def test_an_item_runs_to_the_next_marker_as_the_parser_reads_it() -> None:
    text = ("5. RECOMMENDATIONS FOR FUTURE WELLS\n"
            "  - Hold at least 1.42 sg across Keldra Salt and sweep with\n"
            "  saturated brine every 250 m.\n"
            "  - Inspect mud pump fluid ends between wells.\n"
            f"{SYNTHETIC_FOOTER}\n")
    parsed = parse.parse_eowr(text)["recommendations"]
    found = [p.text for p in passages(text, "eowr") if p.kind]
    assert found == parsed == [
        "Hold at least 1.42 sg across Keldra Salt and sweep with saturated brine every 250 m.",
        "Inspect mud pump fluid ends between wells."]
    assert best_passage(text, "eowr", tokenize("saturated brine sweep")) == found[0]
    ops = ("OPERATIONS SUMMARY\n  06:00-08:00  Drilled 12 1/4\" hole from 2,100 m to\n  2,250 m.\n"
           "  08:00-06:00  Circulated.\n")
    assert [p.text for p in passages(ops, "ddr") if p.kind] == [
        '06:00-08:00 Drilled 12 1/4" hole from 2,100 m to 2,250 m.', "08:00-06:00 Circulated."]


def test_a_heading_may_end_with_a_colon() -> None:
    text = ("REMARKS:\n  Losses of 20 bbl/h at 2,650 m, pumped LCM pill.\n"
            "TIME BREAKDOWN\n  Productive time : 20 h\n")
    found = [(p.section, p.kind, p.role, p.text) for p in passages(text, "ddr")]
    assert found[:2] == [("REMARKS", "", "heading", "REMARKS:"),
                         ("REMARKS", "remarks", "content", "Losses of 20 bbl/h at 2,650 m, pumped LCM pill.")]
    assert best_passage(text, "ddr", tokenize("productive time")) == found[1][3]


def test_a_line_in_capitals_that_reads_as_a_sentence_is_content_not_a_heading() -> None:
    text = ("DAILY DRILLING REPORT\nOPERATIONS SUMMARY\nPOOH TO SHOE. TIGHT SPOT AT 1,400 M.\n"
            "CIRCULATED HOLE CLEAN.\nRemarks: none\n" + SYNTHETIC_FOOTER)
    found = [(p.kind, p.role, p.text) for p in passages(text, "ddr")]
    assert ("operations", "content", "TIGHT SPOT AT 1,400 M.") in found
    assert ("operations", "content", "CIRCULATED HOLE CLEAN.") in found
    assert best_passage(text, "ddr", tokenize("tight spot")) == "TIGHT SPOT AT 1,400 M."


def test_lines_without_final_punctuation_are_separate_passages_unless_they_wrap() -> None:
    text = ("REMARKS\n  Mud losses 20 bbl/h\n  Pumped LCM pill\n  Circulated bottoms up\n\n"
            "2. SEQUENCE OF EVENTS\n" + "  " + "x" * 60 + "\n" + SYNTHETIC_FOOTER)
    assert [p.text for p in passages(text, "ddr") if p.kind] == [
        "Mud losses 20 bbl/h", "Pumped LCM pill", "Circulated bottoms up"]
    # Wrapped at 70 characters (the longest line): "Keldra" would not have fitted on the first
    # line, so the break is a wrap and the lines join.
    first = "Mud weight below the value required to hold the salt in gauge across"
    wrapped = ("4. ROOT CAUSE\n  " + first + "\n  Keldra Salt.\n")
    assert [p.text for p in passages(wrapped, "incident") if p.kind] == [first + " Keldra Salt."]
    # The same lines in a document with longer lines are two statements.
    longer = "1. LOCATION\n  " + "y" * 90 + "\n\n" + wrapped
    assert [p.text for p in passages(longer, "incident") if p.kind] == [first, "Keldra Salt."]
    # The rows of a table in one column are separate passages.
    table = ("3. NPT BREAKDOWN BY CODE\n"
             "  STUCK_PIPE                  23.5 h\n"
             "  RIG_REPAIR                   7.0 h\n")
    assert [p.text for p in passages(table, "eowr")][1:] == ["STUCK_PIPE 23.5 h", "RIG_REPAIR 7.0 h"]


def test_an_abbreviation_does_not_end_a_sentence() -> None:
    text = "REMARKS\n  Pumped approx. 20 bbl LCM pill at 2,650 m. Regained returns.\n"
    assert [p.text for p in passages(text, "ddr") if p.kind] == [
        "Pumped approx. 20 bbl LCM pill at 2,650 m.", "Regained returns."]


def test_a_cue_word_decides_a_tie_for_its_kind_of_passage() -> None:
    # As in the corpus: the sequence shares "pipe" with the question, the root cause only the cue.
    text = ("2. SEQUENCE OF EVENTS\n  String packed off at 777 m while pulling out of hole. Worked pipe and "
            "jarred free after 20.8 h.\n\n4. ROOT CAUSE\n  Mud weight below the value required to hold the "
            "salt in gauge.\n")
    root = "Mud weight below the value required to hold the salt in gauge."
    question = "What caused the stuck pipe incidents on Orrindale?"
    assert best_passage(text, "incident", tokenize(question)) == root
    # Without the cue the earlier passage wins the tie; with more shared terms the sequence still wins.
    assert best_passage(text, "incident", tokenize("stuck pipe on Orrindale")) == (
        "Worked pipe and jarred free after 20.8 h.")
    assert best_passage(text, "incident", tokenize("What caused the pipe to be jarred free?")) == (
        "Worked pipe and jarred free after 20.8 h.")


# ---------------------------------------------------------------------------
# The NPT description rule
# ---------------------------------------------------------------------------

PACK_OFF = ('String packed off at 1,402 m while pulling out of hole for the 13 3/8" intermediate casing. '
            "Salt creep at 1.33 sg across Keldra Salt.")


def test_a_ddr_with_a_matching_entry_is_quoted_by_its_description() -> None:
    stuck = entry("STUCK_PIPE", 15.2, 1402, PACK_OFF)
    quote = evidence_quote(doc(DDR, "ddr"), tokenize("stuck pipe on Orrindale"), [stuck])
    assert quote == stuck.description             # the value only: no "Description :" label
    weather = entry("WAIT_ON_WEATHER", 4.4, 1642, "Storm warning, operations suspended.")
    assert evidence_quote(doc(DDR, "ddr"), tokenize("weather"), [weather]) == weather.description


def test_the_description_is_the_raw_text_not_the_parsed_copy() -> None:
    raw = DDR.replace('13 3/8" intermediate casing. Salt creep', "13\u215c\" intermediate casing. Salt creep")
    parsed = parse.parse_npt_blocks(raw)[0]["description"]
    assert '13 3/8"' in parsed
    quote = evidence_quote(doc(raw, "ddr"), [], [entry("STUCK_PIPE", 15.2, 1402, parsed)])
    assert "13\u215c\" intermediate casing" in quote and verbatim(quote, raw)


def test_of_several_entries_the_nearest_to_the_depth_then_the_best_match_then_the_longest_wins() -> None:
    stuck = entry("STUCK_PIPE", 15.2, 1402, PACK_OFF)
    weather = entry("WAIT_ON_WEATHER", 4.4, 1642, "Storm warning, operations suspended.")
    d = doc(DDR, "ddr")
    assert evidence_quote(d, [], [weather, stuck], depth_m=1650) == weather.description
    assert evidence_quote(d, [], [weather, stuck], depth_m=1400) == stuck.description
    assert evidence_quote(d, tokenize("storm"), [stuck, weather]) == weather.description
    assert evidence_quote(d, [], [weather, stuck]) == stuck.description
    # An entry without a depth ranks after every entry with one when the question gives a depth.
    no_depth = entry("WAIT_ON_WEATHER", 4.4, 0.0, weather.description)
    assert evidence_quote(d, [], [no_depth, stuck], depth_m=1650) == stuck.description


def test_an_entry_whose_description_is_not_in_the_text_falls_back_to_the_best_passage() -> None:
    missing = entry("STUCK_PIPE", 15.2, 1402, "A description the report does not contain.")
    quote = evidence_quote(doc(DDR, "ddr"), tokenize("circulated the hole clean"), [missing])
    assert quote == "03:36-06:00 Circulated the hole clean at 1,642 m."


def test_entries_only_apply_to_daily_reports() -> None:
    stuck = entry("STUCK_PIPE", 15.2, 1402, "Salt creep at 1.33 sg across Keldra Salt.")
    quote = evidence_quote(doc(INCIDENT, "incident"), tokenize("What caused it?"), [stuck])
    assert quote == "Mud weight below the value required to hold the salt in gauge."


# ---------------------------------------------------------------------------
# A CSV ledger row: the quote is the whole row, never a sub-sentence
# ---------------------------------------------------------------------------

def test_a_csv_row_is_quoted_whole_even_with_several_sentences_in_its_description() -> None:
    # Two sentences after the first period: the general sentence-splitting `best_passage`
    # falls back to would otherwise cut this comma-separated record apart at "casing.".
    row = ('ORD-103,2022-03-07,STUCK_PIPE,23.3,Orrindale,1062,"17 1/2""",Keldra Salt,Orrin-1,'
          '"String packed off at 1,062 m while pulling out of hole for the casing. '
          'Salt creep at 1.32 sg across Keldra Salt. Working pipe and jarring."')
    quote = evidence_quote(doc(row, "csv"), tokenize("stuck pipe on Orrindale"))
    assert quote == row


def test_a_csv_row_is_quoted_whole_regardless_of_matching_npt_entries() -> None:
    # `entries` only ever steers a "ddr"; a "csv" document is quoted whole either way.
    row = "ORD-501,2026-01-01,STUCK_PIPE,12.5,Orrindale,1400,17 1/2\",Keldra Salt,Orrin-1,Packed off."
    stuck = entry("STUCK_PIPE", 12.5, 1400, "Packed off.")
    assert evidence_quote(doc(row, "csv"), [], [stuck]) == row


# ---------------------------------------------------------------------------
# In answers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_workspace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Store, Searcher]:
    store = mini.store(tmp_path_factory.mktemp("quotes"))
    return store, mini.searcher(store)


def test_answer_quotes_the_entry_of_the_asked_code(mini_workspace: tuple[Store, Searcher]) -> None:
    store, searcher = mini_workspace
    a = ask("Stuck pipe on Orrindale", store, searcher)
    ddr_quotes = {c.doc_id: c.quote for c in a.citations if c.doc_type == "ddr"}
    assert ddr_quotes["DDR-ORD-102-004"] == ("Pack-off on the trip out in the salt; "
                                             "worked free after circulating.")
    # DDR-ORD-101-006 also has a weather entry; the stuck pipe one is quoted.
    assert ddr_quotes["DDR-ORD-101-006"] == "Continued to work the stuck string and jarred free."
    assert all(h.snippet == ddr_quotes.get(h.doc_id, h.snippet) for h in a.hits)


def test_answer_without_a_code_quotes_an_operations_line(mini_workspace: tuple[Store, Searcher]) -> None:
    store, searcher = mini_workspace
    a = ask("Did the hole hold gauge on ORD-104?", store, searcher)
    quotes = {c.doc_id: c.quote for c in a.citations}
    assert quotes["DDR-ORD-104-003"] == ('10:00-06:00 Drilled 17 1/2" hole through Keldra Salt at 1.45 sg; '
                                         "hole held gauge.")


def test_a_report_that_is_a_hit_and_a_figure_source_is_quoted_once(
        mini_workspace: tuple[Store, Searcher]) -> None:
    store, searcher = mini_workspace
    a = ask("What happened on ORD-101?", store, searcher)
    assert not a.query_plan["codes"] and a.figures is not None
    cited = [s["doc_id"] for s in a.figure_sources[:FIGURE_SOURCE_LIMIT]]
    ddr_quotes: dict[str, set[str]] = {}
    for c in a.citations:
        if c.doc_type == "ddr":
            ddr_quotes.setdefault(c.doc_id, set()).add(c.quote)
    assert all(len(q) == 1 for q in ddr_quotes.values())
    # A figure source is quoted by the description of the entry it contributes.
    assert ddr_quotes["DDR-ORD-101-005"] == {"String packed off at 1,402 m while pulling out of hole. "
                                             "Worked pipe free."}
    assert ddr_quotes["DDR-ORD-101-010"] == {"MWD pulses lost; tripped to change the tool."}
    hits = {h.doc_id: h.snippet for h in a.hits}
    assert set(cited) & set(hits)
    assert all(hits[d] in ddr_quotes[d] for d in cited if d in hits)
    # A hit that is not a figure source keeps its operations line.
    assert ddr_quotes["DDR-ORD-101-001"] == {"10:00-06:00 Drilled ahead."}


def test_a_mitigation_with_a_unicode_fraction_is_quoted_from_the_raw_text(tmp_path: Path) -> None:
    store = mini.store(tmp_path)
    lesson = "Keldra Salt was drilled at 1.45 sg in the 17\u00bd\u2033 section; the hole held gauge."
    store.put_documents([Document(
        "EOWR-ORD-104", "eowr", "ORD-104", "Orrindale", "2024-03-01", "END OF WELL REPORT",
        f"END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. {lesson}\n\n{SYNTHETIC_FOOTER}\n")])
    a = ask("Stuck pipe in Keldra Salt on Orrindale", store, mini.searcher(store))
    quoted = [m["text"] for m in a.mitigations if m["doc_id"] == "EOWR-ORD-104"]
    assert quoted == [lesson]                         # the raw sentence, not the parser's "17 1/2"" copy
    assert lesson in [c.quote for c in a.citations if c.doc_id == "EOWR-ORD-104"]


# ---------------------------------------------------------------------------
# Regression: every question of the extended suite on the default corpus
# ---------------------------------------------------------------------------

# A label at the start of a raw line: `  Hole section          : 17 1/2"`, `Operator: Quillfen Energy`.
_RAW_LABEL = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9 /().,#%'-]*?)[ \t]*:[ \t]")
_HEADING = re.compile(r"^(?:\d\.\s+)?[A-Z][A-Z0-9 ()/&.-]*(?:\([^)]*\))?$")
_INCIDENT_SECTIONS = ("SEQUENCE OF EVENTS", "ROOT CAUSE", "CORRECTIVE ACTIONS")


def _labels(text: str) -> set[str]:
    lines = parse.strip_footer(text).splitlines()
    return {squash(m.group(1)) for line in lines if (m := _RAW_LABEL.match(line))}


def _headings(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines()
            if line and not line[0].isspace() and _HEADING.match(line)}


def _incident_content(text: str) -> str:
    """The sequence of events, root cause and corrective actions of an incident report, squashed."""
    body = parse.strip_footer(text)
    parts = re.split(r"^\d\.\s+([A-Z ]+)$", body, flags=re.M)
    return " ".join(squash(parts[i + 1]) for i in range(1, len(parts) - 1, 2)
                    if parts[i].strip() in _INCIDENT_SECTIONS)


def _questions() -> list[str]:
    out: list[str] = []
    for case in load_suite(CASES, "extended"):
        out += [q for q in [case.get("question"), *case.get("questions", [])] if q]
    return list(dict.fromkeys(out))


@pytest.fixture(scope="module")
def corpus_answers(tmp_path_factory: pytest.TempPathFactory) -> tuple[Store, list[Answer]]:
    root = tmp_path_factory.mktemp("quote-regression")
    write_corpus(build_corpus(seed=SEED), root / "corpus")
    store = Store(root / "wellbrief.db")
    ingest(store, load_corpus_dir(root / "corpus"))
    searcher = build_searcher(store)
    return store, [ask(q, store, searcher) for q in _questions()]


def test_the_extended_questions_are_all_asked(corpus_answers: tuple[Store, list[Answer]]) -> None:
    _, answers = corpus_answers
    assert len(answers) >= 30
    assert sum(len(a.citations) for a in answers) > 300


def test_no_quote_is_a_label_line_a_heading_or_the_footer(corpus_answers: tuple[Store, list[Answer]]) -> None:
    store, answers = corpus_answers
    problems: list[str] = []
    for a in answers:
        for c in a.citations:
            d = store.get_document(c.doc_id)
            assert d is not None
            label = next((lb for lb in _labels(d.text) if re.match(re.escape(lb) + r"\s*:", c.quote)), None)
            if label is not None:
                problems.append(f"{a.question!r}: {c.doc_id} quotes the {label!r} label line: {c.quote!r}")
            if c.quote in _headings(d.text):
                problems.append(f"{a.question!r}: {c.doc_id} quotes a heading: {c.quote!r}")
            if "Synthetic demonstration document" in c.quote:
                problems.append(f"{a.question!r}: {c.doc_id} quotes the footer")
            if not verbatim(c.quote, d.text):
                problems.append(f"{a.question!r}: {c.doc_id} quote is not verbatim: {c.quote!r}")
    assert not problems, "\n".join(problems[:20])
    assert not any('Hole section : 17 1/2"' in squash(c.quote) for a in answers for c in a.citations)


def _uses_ledger(plan: dict[str, object]) -> bool:
    """Whether the answer chose its reports through the NPT ledger (its code, section, formation,
    an "at" depth or the general NPT cue)."""
    return bool(plan["codes"] or plan["hole_sections"] or plan["formations"] or plan["npt"]
                or plan["depth_mode"] == "filter")


def test_each_quote_comes_from_the_passage_its_document_type_calls_for(
        corpus_answers: tuple[Store, list[Answer]]) -> None:
    store, answers = corpus_answers
    problems: list[str] = []
    for a in answers:
        codes = a.query_plan["codes"]
        sources = {s["doc_id"] for s in a.figure_sources[:FIGURE_SOURCE_LIMIT]}
        for c in a.citations:
            d = store.get_document(c.doc_id)
            assert d is not None
            entries = store.npt(doc_id=d.doc_id)
            coded = [e for e in entries if e.code in codes]
            if d.doc_type == "ddr" and coded:
                # Ground truth: the description of an entry of the asked code.
                ok = c.quote in {squash(e.description) for e in coded}
            elif d.doc_type == "ddr" and entries and (_uses_ledger(a.query_plan) or c.doc_id in sources):
                ok = c.quote in {squash(e.description) for e in entries}
            elif d.doc_type == "ddr":
                ops = re.findall(r"^  (\d\d:\d\d-\d\d:\d\d  .*(?:\n {15}.*)*)", d.text, re.M)
                ok = c.quote in {squash(o) for o in ops}
            elif d.doc_type == "eowr":
                parsed = parse.parse_eowr(d.text)
                ok = c.quote in {squash(s) for s in parsed["lessons"] + parsed["recommendations"]}
            else:
                ok = c.quote in _incident_content(d.text)
            if not ok:
                problems.append(f"{a.question!r}: {c.doc_id} ({d.doc_type}) quotes {c.quote!r}")
    assert not problems, "\n".join(problems[:20])
