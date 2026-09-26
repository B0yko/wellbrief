"""Brief-level behaviour built on top of the pattern statistics: the planned well's
`--rig`/`--mwd` marking a risk applies / does not apply / plan did not specify (and the
totals excluding the ones that do not), the planned-TD depth filter, and the unavoidable
background line reaching the brief.

Uses the same hand-built ledger as `test_analytics.py` (`analytics_ledger.py`): Rig-North
(RB-101..103) runs hot on RIG_REPAIR at 2,600 m; Rig-South (RB-104..106) does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import analytics_ledger as ledger
from wellbrief import riskbrief
from wellbrief.models import Document
from wellbrief.store import Store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Store:
    return ledger.store(tmp_path_factory.mktemp("riskbrief"))


def risk(brief: riskbrief.RiskBrief, code: str):
    matches = [r for r in brief.risks if r.code == code]
    assert matches, f"{code} not listed: {[r.code for r in brief.risks]}"
    return matches[0]


# ---------------------------------------------------------------------------
# --rig / --mwd: applies, does not apply, plan did not specify
# ---------------------------------------------------------------------------

def test_rig_repair_applies_when_the_plan_names_its_rig(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, plan_rig=ledger.RIG_NORTH)
    assert risk(brief, "RIG_REPAIR").applies == "applies"


def test_rig_repair_does_not_apply_to_a_different_planned_rig(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, plan_rig=ledger.RIG_SOUTH)
    assert risk(brief, "RIG_REPAIR").applies == "does_not_apply"


def test_rig_repair_is_plan_did_not_specify_with_no_rig_given(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    assert risk(brief, "RIG_REPAIR").applies == "not_specified"


def test_interval_risk_always_applies_regardless_of_the_plan(store: Store) -> None:
    for plan_rig in (None, ledger.RIG_NORTH, ledger.RIG_SOUTH):
        brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, plan_rig=plan_rig)
        assert risk(brief, "WELLBORE_INSTABILITY").applies == "applies"


def test_totals_exclude_a_risk_that_does_not_apply(store: Store) -> None:
    """`does_not_apply` (Rig-South planned) drops RIG_REPAIR's 8.0 expected hours from the
    brief's totals; the risk is still listed."""
    on_plan = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, plan_rig=ledger.RIG_NORTH)
    off_plan = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, plan_rig=ledger.RIG_SOUTH)
    rig_repair_hours = risk(on_plan, "RIG_REPAIR").expected_npt_hours
    assert rig_repair_hours == pytest.approx(8.0)   # probability 1.0 x mean 8.0 h
    assert on_plan.total_expected_npt_hours == pytest.approx(off_plan.total_expected_npt_hours
                                                             + rig_repair_hours)
    assert off_plan.total_exposure_usd < on_plan.total_exposure_usd
    # Still listed, just excluded from the totals.
    assert risk(off_plan, "RIG_REPAIR").applies == "does_not_apply"


def test_not_specified_counts_toward_the_totals_like_applies(store: Store) -> None:
    unspecified = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    on_plan = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, plan_rig=ledger.RIG_NORTH)
    assert unspecified.total_expected_npt_hours == pytest.approx(on_plan.total_expected_npt_hours)


# ---------------------------------------------------------------------------
# Planned TD filter: risks whose depth window starts below the planned TD are dropped
# ---------------------------------------------------------------------------

def test_risk_beyond_planned_td_is_dropped(store: Store) -> None:
    """Every event on this ledger is logged at 2,600 m. A well planned to 2,000 m never
    reaches it."""
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 2000.0)
    assert brief.risks == []


def test_risk_within_planned_td_is_kept(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    assert {r.code for r in brief.risks} == {"RIG_REPAIR", "WELLBORE_INSTABILITY"}


# ---------------------------------------------------------------------------
# Unavoidable background line
# ---------------------------------------------------------------------------

def test_brief_carries_the_unavoidable_background_line(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    assert brief.unavoidable_hours == {"WAIT_ON_WEATHER": 5.0}
    assert "WAIT_ON_WEATHER" not in {r.code for r in brief.risks}


def test_narrative_states_the_unavoidable_background_line(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    assert "Unavoidable background NPT" in brief.narrative
    assert "WAIT_ON_WEATHER" in brief.narrative


# ---------------------------------------------------------------------------
# Verification, and the max-risks cap
# ---------------------------------------------------------------------------

def test_verify_brief_passes_on_a_freshly_built_brief(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    check = riskbrief.verify_brief(brief, store)
    assert check["ok"] is True
    assert check["problems"] == []


def test_max_risks_caps_the_listed_risks_by_expected_cost(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, max_risks=1)
    assert len(brief.risks) == 1
    # RIG_REPAIR (8.0 expected h) outranks WELLBORE_INSTABILITY (2.0 expected h).
    assert brief.risks[0].code == "RIG_REPAIR"


# ---------------------------------------------------------------------------
# Provenance block and the Markdown rendering
# ---------------------------------------------------------------------------

def test_provenance_carries_version_hashes_thresholds_and_narrator(store: Store) -> None:
    provenance = riskbrief.build_provenance(store, ledger.FIELD, 48_000.0, "offline", {"ok": True})
    assert provenance["wellbrief_version"]
    assert provenance["generated_at"].endswith("Z")
    assert len(provenance["corpus_hash"]) == 64        # sha256 hex digest
    assert provenance["spread_rate_usd_per_day"] == 48_000.0
    assert provenance["narrator"] == "offline"
    assert provenance["thresholds"]["min_lift"] == pytest.approx(2.0)
    assert provenance["thresholds"]["equipment_min_ratio"] == pytest.approx(2.0)
    assert provenance["verification"] == {"ok": True}


def test_markdown_rendering_carries_the_disclaimer_and_every_risk(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    check = riskbrief.verify_brief(brief, store)
    text = riskbrief.render_markdown(brief, check)
    assert riskbrief.DISCLAIMER in text
    assert "NOT VERIFIED" not in text
    for r in brief.risks:
        assert r.title in text
    assert "Unavoidable background NPT" in text


def test_markdown_rendering_flags_a_failed_verification(store: Store) -> None:
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    failed_check = {"ok": False, "citations_checked": 1, "problems": ["made up for the test"]}
    text = riskbrief.render_markdown(brief, failed_check)
    assert "NOT VERIFIED" in text


def test_markdown_rendering_shows_the_real_provenance_block(store: Store) -> None:
    """The two tests above build a brief with no `provenance=`, so `render_markdown`'s
    `if prov:` branch (the actual provenance lines) never ran; this builds one the way
    `cli.cmd_brief` does, with `build_provenance`'s own output, and checks its content
    reaches the page instead of only checking the branch does not crash."""
    check = {"ok": True, "citations_checked": 7, "problems": []}
    provenance = riskbrief.build_provenance(store, ledger.FIELD, 48_000.0, "offline", check)
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, provenance=provenance)
    text = riskbrief.render_markdown(brief, check)
    assert "## Provenance" in text
    assert provenance["corpus_hash"][:16] in text
    assert "narrator: offline" in text
    assert "min_lift=2.0" in text
    assert "all citations verbatim" in text
    assert "7 checked" in text


# ---------------------------------------------------------------------------
# `risk_filters=False` (`eval --no-risk-filters`): drops the lift gate and
# lets unavoidable codes back in, changing which patterns a brief lists.
# ---------------------------------------------------------------------------

def test_risk_filters_false_admits_patterns_the_default_filters_drop(store: Store) -> None:
    """WAIT_ON_WEATHER (avoidable = false) never becomes a risk with the default
    filters; `risk_filters=False` (min_lift 0, unavoidable codes included) lets it
    through, in addition to whatever the filtered call already lists."""
    filtered = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    unfiltered = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0, risk_filters=False)
    assert "WAIT_ON_WEATHER" not in {r.code for r in filtered.risks}
    assert "WAIT_ON_WEATHER" in {r.code for r in unfiltered.risks}
    assert len(unfiltered.risks) > len(filtered.risks)


# ---------------------------------------------------------------------------
# Mitigations reach the risk's citations too (`miner.mine_mitigations`,
# wired through `build_risk`): a fresh store, so the shared, module-scoped
# `store` fixture above (no end of well reports at all) stays untouched.
# ---------------------------------------------------------------------------

def test_a_clean_wells_mitigation_is_quoted_and_cited(tmp_path: Path) -> None:
    store = ledger.store(tmp_path)
    # RB-106 (Rig-South): "clean on both counts" for RIG_REPAIR (analytics_ledger's own docstring).
    store.put_documents([Document(
        "EOWR-RB-106", "eowr", "RB-106", ledger.FIELD, "2024-03-01", "END OF WELL REPORT",
        "END OF WELL REPORT\n\n4. LESSONS LEARNED\n  1. Mud pump fluid ends on Rig-South were "
        "inspected and changed between wells; no pump NPT.\n",
    )])
    brief = riskbrief.build_brief(store, "RB-NEXT", ledger.FIELD, 3000.0)
    rig_repair = risk(brief, "RIG_REPAIR")
    assert rig_repair.mitigations == [
        "Mud pump fluid ends on Rig-South were inspected and changed between wells; no pump NPT."
    ]
    cited = {c.doc_id: c.quote for c in rig_repair.citations}
    assert cited["EOWR-RB-106"] == rig_repair.mitigations[0]
    check = riskbrief.verify_brief(brief, store)
    assert check["ok"] is True
