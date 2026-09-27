"""`LlmNarrator`: the OpenAI-compatible narrator behind the egress guard and the cost ledger,
against a fake server on `127.0.0.1:0` (no paid API is ever called).

Covers: the happy path, each hallucination kind falling back to the deterministic offline answer
with a banner and `last_rejection`, the budget stop refusing a call before it reaches the server,
a remote host refused without `WELLBRIEF_ALLOW_REMOTE`, the egress log's line shape, and cumulative
spend across two narrator instances that share one log.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mini_workspace as mini
from fake_openai import FakeOpenAIServer, FakeResponse, completion
from wellbrief.egress import EgressLog
from wellbrief.models import Citation, Risk, RiskBrief
from wellbrief.narrate import llm as narrate_llm
from wellbrief.qa import ask
from wellbrief.search import Searcher
from wellbrief.store import Store


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Store, Searcher]:
    store = mini.store(tmp_path)
    return store, mini.searcher(store)


def _narrator(base_url: str, egress_path: Path, **env: str) -> narrate_llm.LlmNarrator:
    full_env = {"WELLBRIEF_LLM_BASE_URL": base_url, "WELLBRIEF_LLM_MODEL": "fake-model", **env}
    return narrate_llm.narrator_from_env(full_env, egress_path)


def _brief() -> RiskBrief:
    risk = Risk(
        risk_id="STUCK_PIPE-121", title="Stuck pipe / pack-off in the 17 1/2\" section", code="STUCK_PIPE",
        scope="interval", hole_section='17 1/2"', formation="Keldra Salt", rig="", mwd="",
        depth_window_m=(1300.0, 1450.0), wells_total=3, wells_affected=2, probability=0.667,
        mean_npt_hours=20.0, p90_npt_hours=25.0, expected_npt_hours=13.3, expected_cost_usd=26670.0,
        driver="", mitigations=["Raise mud weight to 1.42 sg before drilling into Keldra Salt."],
        citations=[Citation("DDR-ORD-101-005", "ddr", "ORD-101", "2024-01-05",
                            "String packed off at 1,402 m while pulling out of hole."),
                  Citation("INC-ORD-101-01", "incident", "ORD-101", "2024-01-06",
                          "Raise mud weight to 1.42 sg before drilling into Keldra Salt.")],
    )
    return RiskBrief(well_name="ORD-NEXT", field_name="Orrindale", planned_td_m=3100.0,
                     generated_from_wells=["ORD-101", "ORD-102", "ORD-103"],
                     spread_rate_usd_per_day=48_000.0, risks=[risk])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_uses_the_servers_text_and_logs_one_ok_line(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    store, searcher = workspace
    fake_server.queue(FakeResponse(completion(
        "Stuck pipe on Orrindale cost 29.5 h [DDR-ORD-101-005]."
    )))
    egress_path = tmp_path / "egress.jsonl"
    narrator = _narrator(fake_server.base_url, egress_path)
    answer = ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    assert answer.text == "Stuck pipe on Orrindale cost 29.5 h [DDR-ORD-101-005]."
    assert answer.narrator_rejected is None
    assert len(fake_server.requests) == 1
    records = EgressLog(egress_path).records()
    assert len(records) == 1 and records[0]["status"] == "ok"


def test_the_request_carries_the_evidence_pack_doc_ids(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    store, searcher = workspace
    fake_server.queue(FakeResponse(completion("Stuck pipe [DDR-ORD-101-005].")))
    narrator = _narrator(fake_server.base_url, tmp_path / "egress.jsonl")
    ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    body = fake_server.requests[0].json()
    assert body["model"] == "fake-model"
    prompt = "\n".join(m["content"] for m in body["messages"])
    assert "DDR-ORD-101-005" in prompt


# ---------------------------------------------------------------------------
# Hallucinations, each caught and each falling back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("label", "hallucinated_text", "reason_snippet"), [
    ("an id outside the pack", "Stuck pipe cost 29.5 h [DDR-ZZZ-999-001].", "not in the evidence"),
    ("an untraceable number", "Stuck pipe cost 999999.9 h [DDR-ORD-101-005].", "not traceable"),
    ("a well from another field", "This also affected VSS-207 [DDR-ORD-101-005].", "not in the evidence"),
])
def test_a_hallucination_is_rejected_and_falls_back_to_the_offline_answer(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path,
        label: str, hallucinated_text: str, reason_snippet: str) -> None:
    store, searcher = workspace
    fake_server.queue(FakeResponse(completion(hallucinated_text)))
    narrator = _narrator(fake_server.base_url, tmp_path / "egress.jsonl")
    answer = ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    assert answer.narrator_rejected is not None, label
    assert answer.narrator_rejected["text"] == hallucinated_text
    assert any(reason_snippet in r for r in answer.narrator_rejected["reasons"]), label
    assert answer.text.startswith(narrate_llm.VERIFICATION_FAILED_BANNER)
    assert "29.5 h" in answer.text  # the deterministic offline answer, shown instead


def test_an_empty_reply_falls_back_instead_of_showing_a_blank_answer(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    # A realistic failure mode: reasoning ate the whole max_tokens budget, so the server
    # returns empty content with finish_reason "length". That must not verify vacuously.
    store, searcher = workspace
    fake_server.queue(FakeResponse(completion("", finish_reason="length")))
    narrator = _narrator(fake_server.base_url, tmp_path / "egress.jsonl")
    answer = ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    assert answer.narrator_rejected is not None
    assert answer.narrator_rejected["text"] == ""
    assert answer.text.startswith(narrate_llm.VERIFICATION_FAILED_BANNER)
    assert answer.text.strip() != narrate_llm.VERIFICATION_FAILED_BANNER.strip()


# ---------------------------------------------------------------------------
# Budget stop and remote refusal: no request reaches the server
# ---------------------------------------------------------------------------

def test_budget_stop_refuses_before_any_request_reaches_the_server(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    store, searcher = workspace
    egress_path = tmp_path / "egress.jsonl"
    EgressLog(egress_path).append(_spend_record(9.99))
    narrator = _narrator(fake_server.base_url, egress_path, WELLBRIEF_BUDGET_USD="10",
                         WELLBRIEF_LLM_PRICE_IN_PER_MTOK="5", WELLBRIEF_LLM_PRICE_OUT_PER_MTOK="15")
    answer = ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    assert len(fake_server.requests) == 0
    assert answer.narrator_rejected is not None
    assert "call failed" in answer.narrator_rejected["reasons"][0]
    assert "budget" in answer.narrator_rejected["reasons"][0].lower()
    records = EgressLog(egress_path).records()
    assert records[-1]["status"] == "refused-budget"


def test_a_remote_host_is_refused_without_allow_remote(
        workspace: tuple[Store, Searcher], tmp_path: Path) -> None:
    store, searcher = workspace
    egress_path = tmp_path / "egress.jsonl"
    narrator = _narrator("http://example.invalid:8080", egress_path)
    answer = ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    assert answer.narrator_rejected is not None
    assert "WELLBRIEF_ALLOW_REMOTE" in answer.narrator_rejected["reasons"][0]
    records = EgressLog(egress_path).records()
    assert records[-1]["status"] == "refused-remote"
    assert records[-1]["host"] == "example.invalid"


# ---------------------------------------------------------------------------
# Egress log shape and cumulative spend
# ---------------------------------------------------------------------------

def _spend_record(cost_usd: float):  # type: ignore[no-untyped-def]
    from wellbrief.egress import EgressRecord
    return EgressRecord("t", "127.0.0.1:0", 0, (), "m", 0, 0, cost_usd, "price-table", 0.0, "ok")


def test_egress_log_line_has_the_documented_fields(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    store, searcher = workspace
    fake_server.queue(FakeResponse(completion("Stuck pipe [DDR-ORD-101-005].",
                                              prompt_tokens=100, completion_tokens=20)))
    egress_path = tmp_path / "egress.jsonl"
    narrator = _narrator(fake_server.base_url, egress_path)
    ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    (record,) = EgressLog(egress_path).records()
    for key in ("ts", "host", "bytes_sent", "doc_ids", "model", "prompt_tokens", "completion_tokens",
               "cost_usd", "cost_source", "latency_ms", "status", "error", "estimate_usd",
               "tokens_estimated"):
        assert key in record
    assert record["prompt_tokens"] == 100 and record["completion_tokens"] == 20
    assert "DDR-ORD-101-005" in record["doc_ids"]


def test_cumulative_spend_across_two_narrator_instances_sharing_one_log(
        workspace: tuple[Store, Searcher], fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    store, searcher = workspace
    egress_path = tmp_path / "egress.jsonl"
    prices = {"WELLBRIEF_LLM_PRICE_IN_PER_MTOK": "5", "WELLBRIEF_LLM_PRICE_OUT_PER_MTOK": "15"}
    for _ in range(2):
        fake_server.queue(FakeResponse(completion("Stuck pipe [DDR-ORD-101-005].", cost=None)))
        narrator = _narrator(fake_server.base_url, egress_path, **prices)
        ask("What did stuck pipe cost on Orrindale?", store, searcher, narrator=narrator)
    records = EgressLog(egress_path).records()
    assert len(records) == 2
    total = EgressLog(egress_path).spent_usd()
    assert total == pytest.approx(sum(r["cost_usd"] for r in records))
    assert total > 0


# ---------------------------------------------------------------------------
# risk_brief: the same verify-then-fallback behaviour
# ---------------------------------------------------------------------------

def test_risk_brief_happy_path(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    brief = _brief()
    fake_server.queue(FakeResponse(completion(
        "ORD-NEXT is exposed to stuck pipe in the 17 1/2\" section [DDR-ORD-101-005]."
    )))
    narrator = _narrator(fake_server.base_url, tmp_path / "egress.jsonl")
    text = narrator.risk_brief(brief)
    assert text == "ORD-NEXT is exposed to stuck pipe in the 17 1/2\" section [DDR-ORD-101-005]."
    assert narrator.last_rejection is None


def test_risk_brief_hallucination_falls_back(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    brief = _brief()
    fake_server.queue(FakeResponse(completion("This risk also threatens VSS-207 badly.")))
    narrator = _narrator(fake_server.base_url, tmp_path / "egress.jsonl")
    text = narrator.risk_brief(brief)
    assert narrator.last_rejection is not None
    assert text.startswith(narrate_llm.VERIFICATION_FAILED_BANNER)
    assert "ORD-NEXT" in text  # the deterministic offline narrative, shown instead
