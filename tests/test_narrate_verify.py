"""The verifier: checks (a) cited ids, (b) formatted numbers, (c) well/field names, and the
fault-injection harness behind the "verifier catch rate" metric.

The corpus-backed property test (`test_the_offline_narrators_answer_always_verifies`) is the one
the design calls for explicitly: the offline narrator's own output must always pass, over every
question the extended suite asks, since there is nowhere else for it to fall back to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mini_workspace as mini
from wellbrief.evals.cases import default_dir, load_suite
from wellbrief.ingest import ingest_folder
from wellbrief.narrate import OfflineNarrator, verify as v
from wellbrief.qa import ask
from wellbrief.store import Store


def _evidence(doc_ids: frozenset[str] = frozenset(), wells: frozenset[str] = frozenset(),
             fields: frozenset[str] = frozenset(), numbers: tuple[float, ...] = ()) -> v.Evidence:
    return v.Evidence(doc_ids, wells, fields, numbers)


# ---------------------------------------------------------------------------
# Check (a): cited document ids
# ---------------------------------------------------------------------------

def test_a_cited_id_in_the_pack_passes() -> None:
    ev = _evidence(doc_ids=frozenset({"DDR-ORD-101-009"}))
    assert v.verify("Stuck pipe [DDR-ORD-101-009].", ev).ok


def test_a_cited_id_outside_the_pack_is_caught() -> None:
    ev = _evidence(doc_ids=frozenset({"DDR-ORD-101-009"}))
    result = v.verify("Stuck pipe [ZZZ-999].", ev)
    assert not result.ok
    assert "ZZZ-999" in result.reasons[0]


def test_several_unknown_ids_are_each_reported() -> None:
    result = v.verify("[A] and [B]", _evidence())
    assert len(result.reasons) == 2


# ---------------------------------------------------------------------------
# Check (b): formatted-number normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "evidence_number"), [
    ("Total cost was $1.49 M so far.", 1_490_000.0),
    ("Total cost was $1,487,650 so far.", 1_487_650.0),        # an exact plain dollar amount
    ("The rig booked 744.3 h of NPT.", 744.3),
    ("Avoidable share was 29 %.", 0.29),                       # fraction form
    ("Avoidable share was 29 %.", 29.0),                       # point form
    ("Depth reached 1,402 m.", 1402.0),
])
def test_a_formatted_number_traces_to_the_matching_evidence_number(text: str, evidence_number: float) -> None:
    ev = _evidence(numbers=(evidence_number,))
    assert v.verify(text, ev).ok


def test_a_dollar_millions_number_matches_at_its_own_ten_thousand_precision() -> None:
    # $1.49 M rounds to the nearest $10,000; an evidence value in that same bucket passes.
    ev = _evidence(numbers=(1_487_650.32,))
    assert v.verify("Total cost was $1.49 M so far.", ev).ok


def test_an_untraceable_number_is_caught() -> None:
    ev = _evidence(numbers=(1_490_000.0,))
    result = v.verify("Total cost was $9.99 M so far.", ev)
    assert not result.ok
    assert "9.99" in result.reasons[0]


def test_a_hole_size_is_never_checked() -> None:
    ev = _evidence()
    for text in ('Drilled the 12 1/4" section.', "Drilled the 12.25\" section.",
                'Drilled the 12-1/4 in section.', "Drilled the 12 1/4 in section."):
        assert v.verify(text, ev).ok, text


def test_a_number_from_the_question_is_exempt() -> None:
    ev = _evidence()
    result = v.verify("At 2,650 m nothing else is recorded.", ev, question="What happened at 2,650 m?")
    assert result.ok


def test_a_well_id_and_date_do_not_count_as_numbers() -> None:
    ev = _evidence(doc_ids=frozenset({"DDR-ORD-101-009"}), wells=frozenset({"ORD-101"}))
    assert v.verify("On 2026-03-18 well ORD-101 reported nothing new [DDR-ORD-101-009].", ev).ok


def test_a_document_id_outside_a_citation_bracket_leaves_no_orphan_number() -> None:
    # Only the bracketed form ("[DDR-ORD-101-009]") is a citation; this is a plain-prose
    # mention. Before the id-shaped token was stripped whole, the embedded well id
    # ("ORD-101") was removed but the trailing "-009" was left behind and misread as an
    # untraceable number.
    ev = _evidence(doc_ids=frozenset({"DDR-ORD-101-009"}), wells=frozenset({"ORD-101"}))
    result = v.verify("See also DDR-ORD-101-009 for detail.", ev)
    assert result.ok, result.reasons


def test_a_three_segment_document_id_is_also_stripped_whole() -> None:
    ev = _evidence(wells=frozenset({"ORD-101"}))
    assert v.verify("See the EOWR-ORD-101 report.", ev).ok


def test_a_percentile_label_fused_to_a_letter_is_not_read_as_a_number() -> None:
    # "P90" is a percentile label (a 90th-percentile figure), not the measurement 90; the
    # digits are fused directly to the letter with no separator. The genuine figure right
    # after it must still be checked against the evidence.
    ev = _evidence(numbers=(28.1, 37.7))
    assert v.verify("Mean 28.1 h when it happens, P90 37.7 h.", ev).ok


def test_a_percentile_label_does_not_exempt_the_number_after_it() -> None:
    ev = _evidence(numbers=(1.0,))  # 37.7 is nowhere in the evidence
    result = v.verify("P90 37.7 h.", ev)
    assert not result.ok
    assert "37.7" in result.reasons[0]


def test_an_empty_reply_fails_outright() -> None:
    result = v.verify("", _evidence())
    assert not result.ok
    assert result.reasons


def test_a_whitespace_only_reply_fails_outright() -> None:
    result = v.verify("   \n\t", _evidence())
    assert not result.ok
    assert result.reasons


# ---------------------------------------------------------------------------
# Check (c): well and field names
# ---------------------------------------------------------------------------

def test_a_well_named_in_the_pack_passes() -> None:
    ev = _evidence(wells=frozenset({"ORD-101"}))
    assert v.verify("ORD-101 had the problem.", ev).ok


def test_a_well_from_another_field_is_caught() -> None:
    ev = _evidence(wells=frozenset({"ORD-101"}))
    result = v.verify("This also affected VSS-207.", ev)
    assert not result.ok
    assert "VSS-207" in result.reasons[0]


def test_a_field_name_is_caught_only_when_the_caller_names_the_universe() -> None:
    ev = _evidence(fields=frozenset({"Orrindale"}))
    text = "Vessra South had a similar issue."
    assert v.verify(text, ev).ok  # no known_field_names: check (c) cannot see the field name
    result = v.verify(text, ev, known_field_names=frozenset({"Orrindale", "Vessra South"}))
    assert not result.ok


# ---------------------------------------------------------------------------
# Evidence builders
# ---------------------------------------------------------------------------

def test_evidence_from_pack_walks_figures_figure_sources_and_mitigations() -> None:
    pack = [{"doc_id": "DDR-A-1", "well": "ORD-101", "field": "Orrindale", "quote": "22 bbl lost."}]
    summary = {
        "figures": {"total_hours": 280.5, "by_well": [{"well": "ORD-107", "hours": 38.5}]},
        "figure_sources": [{"doc_id": "DDR-A-1", "hours": 24.0}],
        "report_count": 20,
        "mitigations": [{"doc_id": "EOWR-A", "text": "Raise mud weight to 1.42 sg."}],
    }
    ev = v.evidence_from_pack(pack, summary)
    assert ev.doc_ids == frozenset({"DDR-A-1"})
    assert ev.wells == frozenset({"ORD-101", "ORD-107"})
    assert ev.fields == frozenset({"Orrindale"})
    for n in (280.5, 38.5, 24.0, 20.0, 1.42, 22.0):
        assert n in ev.numbers


def test_evidence_from_brief_walks_risks_and_citation_quotes() -> None:
    from wellbrief.models import Citation, Risk, RiskBrief

    risk = Risk(risk_id="STUCK_PIPE-121", title="Stuck pipe", code="STUCK_PIPE", scope="interval",
               hole_section='17 1/2"', formation="Keldra Salt", rig="", mwd="",
               depth_window_m=(1300.0, 1450.0), wells_total=10, wells_affected=4, probability=0.4,
               mean_npt_hours=20.0, p90_npt_hours=30.0, expected_npt_hours=8.0, expected_cost_usd=16000.0,
               driver="", citations=[Citation("DDR-A-1", "ddr", "ORD-101", "2026-01-01", "22 bbl lost.")])
    brief = RiskBrief(well_name="ORD-NEXT", field_name="Orrindale", planned_td_m=3100.0,
                      generated_from_wells=["ORD-101", "ORD-107"], spread_rate_usd_per_day=48000.0,
                      risks=[risk])
    ev = v.evidence_from_brief(brief)
    assert ev.doc_ids == frozenset({"DDR-A-1"})
    assert ev.wells == frozenset({"ORD-101", "ORD-107"})
    assert ev.fields == frozenset({"Orrindale"})
    for n in (20.0, 30.0, 8.0, 16000.0, 22.0, 3100.0):
        assert n in ev.numbers


def test_evidence_from_brief_includes_the_unavoidable_hours_total() -> None:
    # The narrative states the total across codes ("Unavoidable background NPT: 346.2 h"),
    # a sum the narrator computes, not one of the dict's own per-code values.
    from wellbrief.models import RiskBrief

    brief = RiskBrief(well_name="ORD-NEXT", field_name="Orrindale", planned_td_m=3100.0,
                      generated_from_wells=["ORD-101"], spread_rate_usd_per_day=48000.0,
                      risks=[], unavoidable_hours={"HSE_STOP": 200.1, "WAIT_ON_WEATHER": 146.1})
    ev = v.evidence_from_brief(brief)
    assert 346.2 in ev.numbers
    assert v.verify("Unavoidable background NPT: 346.2 h (HSE_STOP, WAIT_ON_WEATHER).", ev).ok


def test_evidence_from_answer_reconstructs_the_same_shape_from_the_public_contract(
        tmp_path: Path) -> None:
    store = mini.store(tmp_path)
    searcher = mini.searcher(store)
    answer = ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=OfflineNarrator())
    ev = v.evidence_from_answer(answer, store)
    assert ev.doc_ids == {c.doc_id for c in answer.citations}
    result = v.verify(answer.text, ev, "What did stuck pipe cost on Orrindale?")
    assert result.ok, result.reasons


# ---------------------------------------------------------------------------
# Fault injection: the "verifier catch rate" harness
# ---------------------------------------------------------------------------

def test_fault_kinds_is_the_documented_four() -> None:
    assert v.FAULT_KINDS == ("quote_char", "id_outside_pack", "untraceable_number", "foreign_well")


def test_every_fault_kind_is_caught_on_a_rich_grounded_text() -> None:
    ev = _evidence(doc_ids=frozenset({"DDR-ORD-101-009"}), wells=frozenset({"ORD-101"}),
                   fields=frozenset({"Orrindale"}), numbers=(23.5, 561000.0))
    text = "Stuck pipe cost 23.5 h, $561,000 total [DDR-ORD-101-009]."
    assert v.verify(text, ev).ok
    for kind in v.FAULT_KINDS:
        kwargs = {"foreign_well": "VSS-207"} if kind == "foreign_well" else {}
        faulted = v.inject_fault(text, ev, kind, **kwargs)
        assert faulted is not None, kind
        result = v.verify(faulted, ev)
        assert not result.ok, f"{kind} was not caught: {faulted!r}"


def test_inject_fault_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown fault kind"):
        v.inject_fault("text", _evidence(), "not-a-kind")


def test_quote_char_returns_none_without_a_number_to_corrupt() -> None:
    ev = _evidence(doc_ids=frozenset({"A"}))
    assert v.inject_fault("No numbers here at all [A].", ev, "quote_char") is None


def test_foreign_well_returns_none_without_a_name() -> None:
    ev = _evidence()
    assert v.inject_fault("Some text.", ev, "foreign_well") is None


# ---------------------------------------------------------------------------
# Property test: the offline narrator always verifies
# ---------------------------------------------------------------------------

def _ask_questions() -> list[str]:
    cases = load_suite(default_dir(), "extended")
    return sorted({c["question"] for c in cases if c.get("question")})


@pytest.fixture(scope="module")
def real_workspace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Store, object]:
    from wellbrief.corpus import SEED, build_corpus, write_corpus
    from wellbrief.workspace import build_searcher

    work = tmp_path_factory.mktemp("verify-extended")
    write_corpus(build_corpus(seed=SEED), work / "corpus", "txt")
    store = Store(work / "wellbrief.db")
    ingest_folder(store, work / "corpus")
    return store, build_searcher(store, "offline")


@pytest.mark.parametrize("question", _ask_questions())
def test_the_offline_narrators_answer_always_verifies(
        real_workspace: tuple[Store, object], question: str) -> None:
    store, searcher = real_workspace
    answer = ask(question, store, searcher, narrator=OfflineNarrator())
    if answer.abstained:
        pytest.skip("an abstention carries no evidence to verify")
    ev = v.evidence_from_answer(answer, store)
    result = v.verify(answer.text, ev, question)
    assert result.ok, (question, result.reasons, answer.text)


@pytest.mark.parametrize("field_name", ["Orrindale", "Vessra South"])
def test_the_offline_narrators_risk_brief_always_verifies(
        real_workspace: tuple[Store, object], field_name: str) -> None:
    from wellbrief.riskbrief import build_brief

    store, _searcher = real_workspace
    assert field_name in store.field_names(), "the default seed's own field names changed"
    brief = build_brief(store, f"{field_name}-NEXT", field_name, 3000.0, narrator=OfflineNarrator())
    ev = v.evidence_from_brief(brief)
    result = v.verify(brief.narrative, ev)
    assert result.ok, (field_name, result.reasons, brief.narrative)
