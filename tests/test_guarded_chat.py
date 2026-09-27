"""Tests for GuardedChat: egress rule, budget stop and egress log around a fake loopback server."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from fake_openai import FakeOpenAIServer, FakeResponse, completion
from wellbrief import egress
from wellbrief.egress import (
    BudgetExceeded,
    CostLedger,
    EgressLog,
    EgressLogError,
    EgressRecord,
    PricingUnknown,
    RemoteNotAllowed,
    estimate_request_tokens,
    utc_timestamp,
)
from wellbrief.narrate.llm import (
    ChatResult,
    GuardedChat,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMHTTPError,
    LLMResponseError,
    OpenAICompatibleClient,
    encode_body,
)

MESSAGES = [
    {"role": "system", "content": "Use only the evidence."},
    {"role": "user", "content": "Evidence [DDR-ORD-101-009]: pack-off at 1,402 m. Summarise."},
]
DOC_IDS = ["DDR-ORD-101-009", "EOWR-ORD-104", "DDR-ORD-101-009"]
REMOTE_URL = "https://llm.example.invalid/v1"


@pytest.fixture(autouse=True)
def isolated_reservations(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test its own in-flight reservation registry."""
    monkeypatch.setattr(egress, "_PENDING", {})
    yield


def lines(log: EgressLog) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]


def prompt_estimate(max_tokens: int = 800, extra: Mapping[str, Any] | None = None) -> int:
    """The prompt estimate GuardedChat makes for MESSAGES sent to model 'fake-model'."""
    http = OpenAICompatibleClient("http://127.0.0.1:1/v1", "fake-model")
    body = encode_body(http.build_body(MESSAGES, max_tokens=max_tokens, extra=extra))
    return estimate_request_tokens(body, len(MESSAGES))


class ScriptedBackend:
    """A ChatBackend that delegates to a real HTTP client but replaces ``chat``."""

    def __init__(self, inner: OpenAICompatibleClient, chat: Callable[[], ChatResult]) -> None:
        self._inner = inner
        self._chat = chat
        self.calls = 0

    @property
    def base_url(self) -> str:
        return self._inner.base_url

    @property
    def host(self) -> str:
        return self._inner.host

    @property
    def model(self) -> str:
        return self._inner.model

    def build_body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.1,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._inner.build_body(messages, max_tokens=max_tokens, temperature=temperature, extra=extra)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.1,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        return self._chat()


def scripted(
    server: FakeOpenAIServer, log_path: Path, chat: Callable[[], ChatResult]
) -> tuple[GuardedChat, ScriptedBackend]:
    backend = ScriptedBackend(OpenAICompatibleClient(server.base_url, "fake-model", timeout=5), chat)
    ledger = CostLedger(EgressLog(log_path), 10.0, 1.0, 2.0, remote=False)
    return GuardedChat(backend, ledger, allow_remote=False), backend


def guarded(
    server_url: str,
    log_path: Path,
    *,
    budget: float = 10.0,
    price_in: float | None = None,
    price_out: float | None = None,
    remote: bool = False,
    allow_remote: bool = False,
) -> GuardedChat:
    http = OpenAICompatibleClient(server_url, "fake-model", timeout=5)
    ledger = CostLedger(EgressLog(log_path), budget, price_in, price_out, remote=remote)
    return GuardedChat(http, ledger, allow_remote=allow_remote)


def seed_spend(path: Path, cost: float) -> None:
    EgressLog(path).append(
        EgressRecord(
            ts=utc_timestamp(),
            host="127.0.0.1",
            bytes_sent=1,
            doc_ids=(),
            model="earlier",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=cost,
            cost_source="provider",
            latency_ms=1.0,
            status="ok",
        )
    )


def test_successful_call_is_logged_with_every_field(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    result = chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=300)
    assert result.text.startswith("Stuck pipe")
    [entry] = lines(chat.ledger.log)
    assert entry["host"] == "127.0.0.1"
    assert entry["bytes_sent"] == len(fake_server.requests[0].body) == result.bytes_sent
    assert entry["doc_ids"] == ["DDR-ORD-101-009", "EOWR-ORD-104"]
    assert entry["model"] == "fake-model"
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == (120, 40)
    assert entry["cost_usd"] == pytest.approx((120 * 1.0 + 40 * 2.0) / 1e6)
    assert entry["cost_source"] == "price-table"
    assert entry["status"] == "ok"
    assert entry["error"] is None
    assert entry["tokens_estimated"] is False
    assert entry["latency_ms"] == result.latency_ms
    assert entry["estimate_usd"] == pytest.approx((prompt_estimate(300) * 1.0 + 300 * 2.0) / 1e6)
    assert entry["ts"].endswith("Z")
    assert chat.ledger.pending_usd() == 0.0


def test_provider_cost_is_used_when_present(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    fake_server.queue(FakeResponse(completion(cost=0.0042)), FakeResponse(completion()))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    chat.chat(MESSAGES, doc_ids=DOC_IDS)
    chat.chat(MESSAGES, doc_ids=DOC_IDS)
    provider, table = lines(chat.ledger.log)
    assert (provider["cost_usd"], provider["cost_source"]) == (0.0042, "provider")
    assert table["cost_source"] == "price-table"
    assert table["cost_usd"] == pytest.approx((120 * 1.0 + 40 * 2.0) / 1e6)
    assert chat.ledger.spent_usd() == pytest.approx(0.0042 + (120 * 1.0 + 40 * 2.0) / 1e6)


def test_loopback_without_prices_is_local_zero(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", budget=0.0)
    chat.chat(MESSAGES, doc_ids=DOC_IDS)
    [entry] = lines(chat.ledger.log)
    assert (entry["cost_usd"], entry["cost_source"], entry["estimate_usd"]) == (0.0, "local-zero", 0.0)


def test_missing_usage_is_priced_at_the_worst_case(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    fake_server.queue(FakeResponse(completion(prompt_tokens=None, completion_tokens=None)))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    [entry] = lines(chat.ledger.log)
    assert entry["tokens_estimated"] is True
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == (prompt_estimate(100), 100)
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])


def test_thinking_is_stripped_through_the_guard(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    fake_server.queue(FakeResponse(completion("<think>\nsum the ledger\n</think>\n\n744.3 h in total.")))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl")
    assert chat.chat(MESSAGES, doc_ids=DOC_IDS).text == "744.3 h in total."


def test_budget_stop_refuses_before_any_request(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    path = tmp_path / "egress.jsonl"
    seed_spend(path, 9.99)
    chat = guarded(fake_server.base_url, path, budget=10.0, price_in=10.0, price_out=30.0)
    with pytest.raises(BudgetExceeded, match="would exceed the budget"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=800)
    assert fake_server.requests == []
    refused = lines(chat.ledger.log)[-1]
    assert refused["status"] == "refused-budget"
    assert (refused["bytes_sent"], refused["cost_usd"], refused["prompt_tokens"]) == (0, 0.0, 0)
    assert refused["doc_ids"] == ["DDR-ORD-101-009", "EOWR-ORD-104"]
    assert "WELLBRIEF_BUDGET_USD" in refused["error"]
    assert refused["estimate_usd"] > 0.01
    assert chat.ledger.spent_usd() == pytest.approx(9.99)
    assert chat.ledger.pending_usd() == 0.0


def test_budget_is_cumulative_across_calls_and_log_instances(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    path = tmp_path / "egress.jsonl"
    fake_server.default = FakeResponse(completion(cost=0.4))
    first = guarded(fake_server.base_url, path, budget=1.0, price_in=0.0, price_out=0.0)
    first.chat(MESSAGES, doc_ids=DOC_IDS)
    second = guarded(fake_server.base_url, path, budget=1.0, price_in=0.0, price_out=0.0)
    second.chat(MESSAGES, doc_ids=DOC_IDS)
    assert EgressLog(path).spent_usd() == pytest.approx(0.8)
    strict = guarded(fake_server.base_url, path, budget=1.0, price_in=1.0, price_out=300.0)
    with pytest.raises(BudgetExceeded):
        strict.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=800)  # 0.8 spent + 0.24 worst case > 1.0
    assert len(fake_server.requests) == 2
    assert [entry["status"] for entry in lines(EgressLog(path))] == ["ok", "ok", "refused-budget"]


def test_remote_host_is_refused_without_allow_remote(tmp_path: Path, no_network: list[str]) -> None:
    chat = guarded(REMOTE_URL, tmp_path / "egress.jsonl", price_in=1.0, price_out=1.0, remote=True)
    with pytest.raises(RemoteNotAllowed, match="WELLBRIEF_ALLOW_REMOTE=1"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS)
    assert no_network == []
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["host"], entry["bytes_sent"]) == (
        "refused-remote",
        "llm.example.invalid",
        0,
    )
    assert entry["cost_usd"] == 0.0


def test_remote_host_without_prices_is_refused_even_when_allowed(
    tmp_path: Path, no_network: list[str]
) -> None:
    chat = guarded(REMOTE_URL, tmp_path / "egress.jsonl", remote=True, allow_remote=True)
    with pytest.raises(PricingUnknown, match="WELLBRIEF_LLM_PRICE_IN_PER_MTOK"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS)
    assert no_network == []
    [entry] = lines(chat.ledger.log)
    assert entry["status"] == "refused-budget"
    assert entry["estimate_usd"] is None


def test_allowed_remote_host_still_meets_the_budget_stop(tmp_path: Path, no_network: list[str]) -> None:
    path = tmp_path / "egress.jsonl"
    seed_spend(path, 10.0)
    chat = guarded(REMOTE_URL, path, price_in=1.0, price_out=1.0, remote=True, allow_remote=True)
    with pytest.raises(BudgetExceeded):
        chat.chat(MESSAGES, doc_ids=DOC_IDS)
    assert no_network == []
    assert lines(chat.ledger.log)[-1]["status"] == "refused-budget"


def test_ledger_must_match_the_host(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    log = EgressLog(tmp_path / "egress.jsonl")
    with pytest.raises(ValueError, match="remote=True"):
        GuardedChat(OpenAICompatibleClient(fake_server.base_url, "m"), CostLedger(log, 1, 1, 1, remote=True),
                    allow_remote=False)  # fmt: skip
    with pytest.raises(ValueError, match="remote=False"):
        GuardedChat(OpenAICompatibleClient(REMOTE_URL, "m"), CostLedger(log, 1, None, None, remote=False),
                    allow_remote=True)  # fmt: skip


def test_http_500_is_logged_as_an_error_charged_at_the_worst_case(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    fake_server.queue(FakeResponse({"error": {"message": "out of memory"}}, status=500))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    with pytest.raises(LLMHTTPError, match="HTTP 500"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=200)
    [entry] = lines(chat.ledger.log)
    assert entry["status"] == "error"
    assert "HTTP 500: out of memory" in entry["error"]
    assert entry["bytes_sent"] == len(fake_server.requests[0].body)
    assert entry["tokens_estimated"] is True
    assert entry["completion_tokens"] == 200
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])
    assert chat.ledger.pending_usd() == 0.0


def test_malformed_json_is_logged_as_an_error(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    fake_server.queue(FakeResponse(b"{not json"))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl")
    with pytest.raises(LLMResponseError, match="not JSON"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS)
    [entry] = lines(chat.ledger.log)
    assert entry["status"] == "error"
    assert "not JSON" in entry["error"]
    assert (entry["cost_usd"], entry["cost_source"]) == (0.0, "local-zero")


def test_rejected_request_is_logged_at_zero_cost(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    fake_server.queue(FakeResponse({"error": {"message": "bad request"}}, status=400))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    with pytest.raises(LLMHTTPError):
        chat.chat(MESSAGES, doc_ids=DOC_IDS)
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["cost_usd"], entry["tokens_estimated"]) == ("error", 0.0, False)


def test_unreachable_server_is_logged_at_zero_cost(closed_port: int, tmp_path: Path) -> None:
    chat = guarded(
        f"http://127.0.0.1:{closed_port}/v1", tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0
    )
    with pytest.raises(LLMConnectionError):
        chat.chat(MESSAGES, doc_ids=DOC_IDS)
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["bytes_sent"], entry["cost_usd"]) == ("error", 0, 0.0)


def test_invalid_request_is_neither_sent_nor_logged(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl")
    with pytest.raises(LLMConfigError):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, extra={"max_completion_tokens": 100_000})
    assert fake_server.requests == []
    assert not (tmp_path / "egress.jsonl").exists()


def test_extra_fields_count_in_the_prompt_estimate(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    extra = {"chat_template_kwargs": {"enable_thinking": False, "note": "x" * 5_000}}
    chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100, extra=extra)
    plain, padded = lines(chat.ledger.log)
    assert padded["estimate_usd"] - plain["estimate_usd"] >= 5_000 * 1.0 / 1e6
    assert padded["estimate_usd"] == pytest.approx((prompt_estimate(100, extra) * 1.0 + 100 * 2.0) / 1e6)
    assert padded["bytes_sent"] == len(fake_server.requests[1].body)


def test_a_single_string_as_doc_ids_is_refused(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl")
    with pytest.raises(LLMConfigError, match="not a single string"):
        chat.chat(MESSAGES, doc_ids="DDR-ORD-101-009")
    assert fake_server.requests == []
    assert not (tmp_path / "egress.jsonl").exists()


def test_surrogate_in_the_served_model_is_logged(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    body = json.dumps(completion(cost=0.5)).replace('"fake-model"', '"\\ud800"')
    fake_server.queue(FakeResponse(body))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    assert chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100).model == "\ufffd"
    assert len(fake_server.requests) == 1
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["model"], entry["cost_usd"]) == ("ok", "\ufffd", 0.5)
    assert chat.ledger.log.path.read_bytes().isascii()
    assert chat.ledger.spent_usd() == pytest.approx(0.5)


def test_surrogate_in_an_error_body_is_logged_at_the_worst_case(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    fake_server.queue(FakeResponse(b'{"error": {"message": "boom \\udc80"}}', status=500))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    with pytest.raises(LLMHTTPError, match="boom"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["tokens_estimated"]) == ("error", True)
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])
    assert entry["cost_usd"] > 0


def test_oversized_usage_values_are_logged(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    huge = "9" * 400
    ok = '{"choices": [{"message": {"content": "ok"}}], "usage": %s}'
    fake_server.queue(
        FakeResponse(ok % f'{{"prompt_tokens": 10, "completion_tokens": 5, "cost": {huge}}}'),
        FakeResponse(ok % f'{{"prompt_tokens": {huge}, "completion_tokens": 5}}'),
    )
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    with pytest.raises(LLMResponseError, match=re.escape("usage.prompt_tokens")):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    assert len(fake_server.requests) == 2
    priced, failed = lines(chat.ledger.log)
    assert (priced["status"], priced["cost_source"]) == ("ok", "price-table")
    assert priced["cost_usd"] == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6)
    assert (failed["status"], failed["tokens_estimated"]) == ("error", True)
    assert failed["cost_usd"] == pytest.approx(failed["estimate_usd"])
    assert chat.ledger.pending_usd() == 0.0


def test_unexpected_exception_is_logged_at_the_worst_case_and_raised_as_llm_error(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    def explode() -> ChatResult:
        raise RuntimeError("parser bug \ud800")

    chat, backend = scripted(fake_server, tmp_path / "egress.jsonl", explode)
    with pytest.raises(LLMError, match="failed unexpectedly: RuntimeError: parser bug") as info:
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    assert isinstance(info.value.__cause__, RuntimeError)
    str(info.value).encode("utf-8")
    assert backend.calls == 1
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["tokens_estimated"]) == ("error", True)
    assert entry["bytes_sent"] == len(encode_body(backend.build_body(MESSAGES, max_tokens=100)))
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])
    assert chat.ledger.pending_usd() == 0.0


def test_interruption_is_logged_and_re_raised(fake_server: FakeOpenAIServer, tmp_path: Path) -> None:
    def interrupt() -> ChatResult:
        raise KeyboardInterrupt

    chat, _ = scripted(fake_server, tmp_path / "egress.jsonl", interrupt)
    with pytest.raises(KeyboardInterrupt):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    [entry] = lines(chat.ledger.log)
    assert "interrupted (KeyboardInterrupt)" in entry["error"]
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])
    assert chat.ledger.pending_usd() == 0.0


def test_a_reply_that_cannot_be_priced_is_logged_at_the_worst_case(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    reply = ChatResult("ok", 10, 5, -1.0, latency_ms=1.0, bytes_sent=1, model="fake-model")
    chat, _ = scripted(fake_server, tmp_path / "egress.jsonl", lambda: reply)
    with pytest.raises(LLMError, match="provider_cost_usd"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["tokens_estimated"]) == ("error", True)
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])


def test_a_reply_that_cannot_be_logged_as_ok_is_logged_as_an_error(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    reply = ChatResult("ok", 10, 5, None, latency_ms=float("nan"), bytes_sent=1, model="fake-model")
    chat, _ = scripted(fake_server, tmp_path / "egress.jsonl", lambda: reply)
    with pytest.raises(LLMError, match="failed unexpectedly: ValueError"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    [entry] = lines(chat.ledger.log)
    assert (entry["status"], entry["tokens_estimated"]) == ("error", True)
    assert entry["cost_usd"] == pytest.approx(entry["estimate_usd"])
    assert chat.ledger.pending_usd() == 0.0


def test_a_record_that_cannot_be_written_keeps_its_reservation(
    fake_server: FakeOpenAIServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)

    def disk_full(self: EgressLog, record: EgressRecord) -> None:
        raise EgressLogError(f"{self.path}: the record cannot be appended: disk full")

    monkeypatch.setattr(EgressLog, "append", disk_full)
    with pytest.raises(EgressLogError, match="disk full"):
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    assert len(fake_server.requests) == 1
    expected = (prompt_estimate(100) * 1.0 + 100 * 2.0) / 1e6
    assert chat.ledger.pending_usd() == pytest.approx(expected)


def test_a_failed_error_record_also_keeps_its_reservation(
    fake_server: FakeOpenAIServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_server.queue(FakeResponse({"error": {"message": "overloaded"}}, status=503))
    chat = guarded(fake_server.base_url, tmp_path / "egress.jsonl", price_in=1.0, price_out=2.0)
    real_append = EgressLog.append

    def fail_on_error(self: EgressLog, record: EgressRecord) -> None:
        if record.status == "error":
            raise EgressLogError("cannot be appended")
        real_append(self, record)

    monkeypatch.setattr(EgressLog, "append", fail_on_error)
    with pytest.raises(EgressLogError) as info:
        chat.chat(MESSAGES, doc_ids=DOC_IDS, max_tokens=100)
    assert isinstance(info.value.__context__, LLMHTTPError)
    assert chat.ledger.pending_usd() > 0
