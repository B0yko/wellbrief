"""Tests for the egress guard, the egress log and the cost ledger (no HTTP involved)."""

from __future__ import annotations

import ipaddress
import json
import math
import threading
from pathlib import Path

import pytest

from wellbrief.egress import (
    BYTES_PER_TOKEN,
    MAX_TOKEN_COUNT,
    MESSAGE_OVERHEAD_TOKENS,
    REQUEST_OVERHEAD_TOKENS,
    BudgetExceeded,
    CostLedger,
    EgressLog,
    EgressLogError,
    EgressRecord,
    EgressStatus,
    InvalidBaseURL,
    PricingUnknown,
    RemoteNotAllowed,
    check_egress,
    estimate_prompt_tokens,
    estimate_request_tokens,
    is_loopback_host,
    network_mode,
    split_base_url,
    unique_ids,
    utc_timestamp,
)


def record(
    cost: float,
    *,
    status: EgressStatus = "ok",
    doc_ids: tuple[str, ...] = ("DDR-ORD-101-009",),
    model: str = "fake-model",
    error: str | None = None,
) -> EgressRecord:
    return EgressRecord(
        ts=utc_timestamp(),
        host="127.0.0.1",
        bytes_sent=100,
        doc_ids=doc_ids,
        model=model,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=cost,
        cost_source="price-table",
        latency_ms=12.5,
        status=status,
        error=error,
    )


# --- is_loopback_host ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("LocalHost", True),
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        ("127.255.255.255", True),
        ("::1", True),
        ("[::1]", True),
        ("0:0:0:0:0:0:0:1", False),
        ("::0:1", False),
        ("[0:0:0:0:0:0:0:1]", False),
        ("localhost.example.com", False),
        ("localhost.", False),
        ("127.0.0.1.nip.io", False),
        ("0.0.0.0", False),
        ("192.0.2.1", False),
        ("::", False),
        ("::ffff:127.0.0.1", False),
        ("::1%lo0", False),
        ("[127.0.0.1]", False),
        ("127.1", False),
        ("0177.0.0.1", False),
        ("2130706433", False),
        ("openrouter.ai", False),
        (" localhost", False),
        ("", False),
    ],
)
def test_is_loopback_host_truth_table(host: str, expected: bool) -> None:
    assert is_loopback_host(host) is expected


def test_is_loopback_host_stops_at_the_edges_of_127_slash_8() -> None:
    network = ipaddress.IPv4Network("127.0.0.0/8")
    assert is_loopback_host(str(network.network_address))
    assert is_loopback_host(str(network.broadcast_address))
    assert not is_loopback_host(str(network.network_address - 1))
    assert not is_loopback_host(str(network.broadcast_address + 1))


def test_is_loopback_host_never_resolves_names(no_network: list[str]) -> None:
    for host in ("localhost", "localhost.example.com", "127.0.0.1.nip.io", "openrouter.ai", "::1"):
        is_loopback_host(host)
        check_egress(f"http://{host if ':' not in host else '[' + host + ']'}:8080/v1", allow_remote=True)
    assert no_network == []


# --- base URL parsing ---------------------------------------------------------------------------


def test_split_base_url_parts() -> None:
    url = split_base_url("http://127.0.0.1:8080/v1/")
    assert (url.scheme, url.host, url.port, url.netloc, url.path) == (
        "http",
        "127.0.0.1",
        8080,
        "127.0.0.1:8080",
        "/v1",
    )
    assert url.endpoint("chat/completions") == "http://127.0.0.1:8080/v1/chat/completions"
    ipv6 = split_base_url("http://[::1]:8080/v1")
    assert ipv6.host == "::1"
    assert ipv6.endpoint("/chat/completions") == "http://[::1]:8080/v1/chat/completions"
    assert split_base_url("HTTPS://OpenRouter.ai/api/v1").host == "openrouter.ai"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "127.0.0.1:8080/v1",
        "ftp://127.0.0.1/v1",
        "file:///etc/hosts",
        "http:///v1",
        "http://user:secret@127.0.0.1:8080/v1",
        "http://127.0.0.1@example.invalid/v1",
        "http://127.0.0.1:8080/v1?x=1",
        "http://127.0.0.1:8080/v1#frag",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1:8080 /v1",
        "http://127.0.0.1\\@example.invalid/v1",
        "http://127.0.0.1:8080/v1\n",
        "http://xn--caf-dma.example/v1\u00e9",
        "http://[::1/v1",
    ],
)
def test_split_base_url_rejects_unusable_urls(bad: str) -> None:
    with pytest.raises(InvalidBaseURL):
        split_base_url(bad)


# --- network mode and the egress decision --------------------------------------------------------


def test_network_mode() -> None:
    assert network_mode("offline", None) == "offline"
    assert network_mode("offline", "https://openrouter.ai/api/v1") == "offline"
    assert network_mode("llm", None) == "offline"
    assert network_mode("llm", "") == "offline"
    assert network_mode("llm", "http://127.0.0.1:8080/v1") == "local-llm"
    assert network_mode("llm", "http://localhost:8080/v1") == "local-llm"
    assert network_mode("llm", "http://[::1]:8080/v1") == "local-llm"
    assert network_mode("llm", "https://openrouter.ai/api/v1") == "remote-llm"
    assert network_mode("llm", "http://127.0.0.1.nip.io:8080/v1") == "remote-llm"
    with pytest.raises(ValueError, match="Unknown narrator"):
        network_mode("ollama", None)
    with pytest.raises(InvalidBaseURL):
        network_mode("llm", "not a url")


def test_check_egress_allows_loopback() -> None:
    assert check_egress("http://127.0.0.1:8080/v1", allow_remote=False).host == "127.0.0.1"
    assert check_egress("http://LOCALHOST:8080/v1", allow_remote=False).host == "localhost"


def test_check_egress_refuses_remote_without_allow(no_network: list[str]) -> None:
    with pytest.raises(RemoteNotAllowed, match="WELLBRIEF_ALLOW_REMOTE=1") as info:
        check_egress("https://openrouter.ai/api/v1", allow_remote=False)
    assert info.value.host == "openrouter.ai"
    with pytest.raises(RemoteNotAllowed):
        check_egress("http://127.0.0.1.nip.io:8080/v1", allow_remote=False)
    assert no_network == []


def test_check_egress_allows_remote_when_allowed(no_network: list[str]) -> None:
    assert check_egress("https://openrouter.ai/api/v1", allow_remote=True).host == "openrouter.ai"
    assert no_network == []


# --- token estimate -----------------------------------------------------------------------------


def test_estimate_prompt_tokens_is_one_per_utf8_byte() -> None:
    assert BYTES_PER_TOKEN == 1
    assert estimate_prompt_tokens("") == 0
    assert estimate_prompt_tokens("Stuck pipe at 1,402 m") == 21
    assert estimate_prompt_tokens('12\u00bc"') == 5  # the vulgar fraction is two UTF-8 bytes


def test_estimate_prompt_tokens_covers_one_token_per_character_on_ascii() -> None:
    # Qwen-style tokenizers split every digit into its own token, so a date- and id-heavy line can need
    # close to one token per character; the estimate never falls below one per character on ASCII.
    text = "DDR-ORD-101-009 2026-07-31 1402 m 23.5 h"
    assert estimate_prompt_tokens(text) == len(text)


def test_estimate_prompt_tokens_counts_a_lone_surrogate_instead_of_failing() -> None:
    assert estimate_prompt_tokens("a\ud800") == 4


def test_estimate_request_tokens_counts_the_whole_body_plus_overheads() -> None:
    messages = [{"role": "system", "content": "abc"}, {"role": "user", "content": "defg"}]
    body = json.dumps({"model": "m", "messages": messages}).encode("utf-8")
    expected = REQUEST_OVERHEAD_TOKENS + 2 * MESSAGE_OVERHEAD_TOKENS + len(body)
    assert estimate_request_tokens(body, 2) == expected
    # The body holds more than the message text, so the estimate covers every field that is sent.
    text_only = sum(len(m["role"]) + len(m["content"]) for m in messages)
    assert (
        estimate_request_tokens(body, 2) > REQUEST_OVERHEAD_TOKENS + 2 * MESSAGE_OVERHEAD_TOKENS + text_only
    )
    with pytest.raises(ValueError):
        estimate_request_tokens(body, -1)


# --- egress log ---------------------------------------------------------------------------------


def test_egress_log_writes_one_json_line_per_record(tmp_path: Path) -> None:
    log = EgressLog(tmp_path / "ws" / "egress.jsonl")
    assert log.records() == []
    assert log.spent_usd() == 0.0
    log.append(record(0.25, doc_ids=("A", "B")))
    log.append(record(0.0, status="refused-budget"))
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert list(first) == [
        "ts",
        "host",
        "bytes_sent",
        "doc_ids",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "cost_source",
        "latency_ms",
        "status",
        "error",
        "estimate_usd",
        "tokens_estimated",
    ]
    assert first["doc_ids"] == ["A", "B"]
    assert first["ts"].endswith("Z")
    assert log.spent_usd() == pytest.approx(0.25)


def test_spend_is_cumulative_across_log_instances_on_one_file(tmp_path: Path) -> None:
    path = tmp_path / "egress.jsonl"
    first = EgressLog(path)
    first.append(record(0.25))
    second = EgressLog(path)
    second.append(record(0.5))
    assert first.spent_usd() == pytest.approx(0.75)
    assert second.spent_usd() == pytest.approx(0.75)
    ledger = CostLedger(second, 1.0, 1.0, 1.0, remote=True)
    assert ledger.spent_usd() == pytest.approx(0.75)
    with pytest.raises(BudgetExceeded):
        ledger.check_before_call(0, 300_000)  # 0.3 USD of output on top of 0.75 spent


def test_malformed_log_line_blocks_the_ledger(tmp_path: Path) -> None:
    path = tmp_path / "egress.jsonl"
    log = EgressLog(path)
    log.append(record(0.1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"cost_usd": 0.2, "status": "ok"')  # a writer stopped mid-line
    with pytest.raises(EgressLogError, match="line 2"):
        log.spent_usd()
    log.append(record(0.3))
    assert path.read_text(encoding="utf-8").splitlines()[-1].startswith('{"ts"')
    with pytest.raises(EgressLogError):
        CostLedger(log, 10.0, 1.0, 1.0, remote=True).check_before_call(10, 10)


@pytest.mark.parametrize(
    "bad", ['"text"', '{"cost_usd": -1}', '{"cost_usd": "0.1"}', '{"status": "ok"}', "[1]"]
)
def test_log_rejects_lines_that_cannot_be_totalled(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "egress.jsonl"
    path.write_text(bad + "\n", encoding="utf-8")
    with pytest.raises(EgressLogError):
        EgressLog(path).spent_usd()


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "egress.jsonl"
    path.write_text('\n{"cost_usd": 0.5}\n\n', encoding="utf-8")
    assert EgressLog(path).spent_usd() == pytest.approx(0.5)


def test_log_is_written_as_ascii_and_keeps_any_server_string(tmp_path: Path) -> None:
    log = EgressLog(tmp_path / "egress.jsonl")
    log.append(record(0.5, model="\ud800", error="boom \udc80 caf\u00e9"))
    raw = log.path.read_bytes()
    assert raw.isascii()
    [entry] = log.records()
    assert (entry["model"], entry["error"]) == ("\ud800", "boom \udc80 caf\u00e9")
    assert log.spent_usd() == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b'{"cost_usd": 0.1}\n\xff\xfe\n', "line 2 is not valid UTF-8"),
        ('{"cost_usd": 0.1, "error": "\u00e9'.encode()[:-1], "line 1 is not valid UTF-8"),
        (b'{"cost_usd": ' + b"9" * 400 + b"}\n", "very long integer"),
        (b'{"cost_usd": ' + b"9" * 5_000 + b"}\n", "line 1 is not valid JSON"),
        (b"[" * 100_000 + b"]" * 100_000 + b"\n", "line 1 is not valid JSON"),
        (b'{"cost_usd": 1e999}\n', "cost_usd=inf"),
    ],
)
def test_corrupt_log_raises_egress_log_error(tmp_path: Path, data: bytes, match: str) -> None:
    path = tmp_path / "egress.jsonl"
    path.write_bytes(data)
    with pytest.raises(EgressLogError, match=match):
        EgressLog(path).spent_usd()


def test_unreadable_or_unwritable_log_raises_egress_log_error(tmp_path: Path) -> None:
    directory = tmp_path / "egress.jsonl"
    directory.mkdir()
    with pytest.raises(EgressLogError, match="cannot be read"):
        EgressLog(directory).records()
    with pytest.raises(EgressLogError, match="cannot be appended"):
        EgressLog(directory).append(record(0.1))
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(EgressLogError, match="cannot be appended"):
        EgressLog(blocker / "egress.jsonl").append(record(0.1))


def test_concurrent_appends_keep_every_line(tmp_path: Path) -> None:
    log = EgressLog(tmp_path / "egress.jsonl")

    def worker() -> None:
        for _ in range(25):
            EgressLog(log.path).append(record(0.01))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(log.records()) == 100
    assert log.spent_usd() == pytest.approx(1.0)


# --- cost ledger --------------------------------------------------------------------------------


def test_estimate_worst_case_uses_prices_per_million_tokens(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 10.0, 0.5, 2.0, remote=True)
    assert ledger.estimate_worst_case(1_000, 800) == pytest.approx((1_000 * 0.5 + 800 * 2.0) / 1e6)
    assert ledger.estimate_worst_case(0, 1) == pytest.approx(2.0 / 1e6)
    assert ledger.cost_source == "price-table"
    assert ledger.cost_of(100, 50) == pytest.approx((100 * 0.5 + 50 * 2.0) / 1e6)


def test_check_before_call_refuses_when_spent_plus_estimate_exceeds_budget(tmp_path: Path) -> None:
    log = EgressLog(tmp_path / "e.jsonl")
    log.append(record(9.99))
    ledger = CostLedger(log, 10.0, 10.0, 30.0, remote=True)
    assert ledger.check_before_call(100, 100) == pytest.approx((100 * 10 + 100 * 30) / 1e6)
    with pytest.raises(BudgetExceeded, match="WELLBRIEF_BUDGET_USD") as info:
        ledger.check_before_call(1_000, 800)
    assert info.value.spent_usd == pytest.approx(9.99)
    assert info.value.estimate_usd == pytest.approx((1_000 * 10 + 800 * 30) / 1e6)
    assert info.value.budget_usd == 10.0


def test_budget_boundary_is_inclusive(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 2.0, 1.0, 1.0, remote=True)
    assert ledger.check_before_call(1_000_000, 1_000_000) == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        ledger.check_before_call(1_000_000, 1_000_001)


def test_remote_host_without_prices_is_refused(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 10.0, None, None, remote=True)
    with pytest.raises(PricingUnknown) as info:
        ledger.check_before_call(10, 800)
    message = str(info.value)
    assert "WELLBRIEF_LLM_PRICE_IN_PER_MTOK" in message
    assert "WELLBRIEF_LLM_PRICE_OUT_PER_MTOK" in message
    assert isinstance(info.value, BudgetExceeded)
    with pytest.raises(PricingUnknown):
        ledger.cost_of(1, 1)


def test_loopback_host_without_prices_costs_zero(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 0.0, None, None, remote=False)
    assert ledger.cost_source == "local-zero"
    assert ledger.estimate_worst_case(50_000, 800) == 0.0
    assert ledger.check_before_call(50_000, 800) == 0.0
    settled = ledger.settle(10, 5, None, prompt_estimate=99, max_tokens=800)
    assert (settled.cost_usd, settled.cost_source) == (0.0, "local-zero")


def test_loopback_host_with_prices_is_priced(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 10.0, 1.0, 1.0, remote=False)
    assert ledger.cost_source == "price-table"
    assert ledger.estimate_worst_case(1_000, 1_000) == pytest.approx(0.002)


def test_settle_prefers_provider_cost_and_falls_back_to_prices(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 10.0, 1.0, 2.0, remote=True)
    provider = ledger.settle(100, 50, 0.0042, prompt_estimate=999, max_tokens=800)
    assert (provider.cost_usd, provider.cost_source, provider.tokens_estimated) == (0.0042, "provider", False)
    table = ledger.settle(100, 50, None, prompt_estimate=999, max_tokens=800)
    assert table.cost_source == "price-table"
    assert table.cost_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1e6)
    missing = ledger.settle(None, None, None, prompt_estimate=999, max_tokens=800)
    assert (missing.prompt_tokens, missing.completion_tokens, missing.tokens_estimated) == (999, 800, True)
    assert missing.cost_usd == pytest.approx((999 * 1.0 + 800 * 2.0) / 1e6)
    with pytest.raises(ValueError):
        ledger.settle(1, 1, -0.5, prompt_estimate=1, max_tokens=1)


@pytest.mark.parametrize(
    ("budget", "price_in", "price_out"),
    [
        (-1.0, None, None),
        (math.nan, None, None),
        (10**400, None, None),
        (10.0, 1.0, None),
        (10.0, None, 1.0),
        (10.0, -1.0, 1.0),
        (10.0, 10**400, 1.0),
    ],
)
def test_ledger_rejects_bad_configuration(
    tmp_path: Path, budget: float, price_in: float | None, price_out: float | None
) -> None:
    with pytest.raises(ValueError):
        CostLedger(EgressLog(tmp_path / "e.jsonl"), budget, price_in, price_out, remote=True)


@pytest.mark.parametrize(
    ("prompt_tokens", "max_tokens"),
    [(-1, 10), (10, 0), (10, True), (MAX_TOKEN_COUNT + 1, 10), (10, 10**400)],
)
def test_estimate_rejects_bad_token_counts(tmp_path: Path, prompt_tokens: int, max_tokens: int) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 10.0, 1.0, 1.0, remote=True)
    with pytest.raises(ValueError):
        ledger.estimate_worst_case(prompt_tokens, max_tokens)


def test_reservations_count_against_the_budget_until_released(tmp_path: Path) -> None:
    log = EgressLog(tmp_path / "e.jsonl")
    ledger = CostLedger(log, 1.0, 1.0, 1.0, remote=True)
    other = CostLedger(EgressLog(log.path), 1.0, 1.0, 1.0, remote=True)
    first = ledger.reserve(0, 600_000)
    assert ledger.pending_usd() == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded):
        other.reserve(0, 600_000)  # a second call in flight on the same workspace log
    with first:
        pass
    first.release()  # releasing twice is harmless
    assert ledger.pending_usd() == 0.0
    with other.reserve(0, 600_000) as second:
        assert second.estimate_usd == pytest.approx(0.6)
    assert other.pending_usd() == 0.0


def test_unique_ids_keeps_first_seen_order() -> None:
    assert unique_ids(["B", "A", "B", "C", "A"]) == ("B", "A", "C")
    assert unique_ids(iter(("A", "A"))) == ("A",)


@pytest.mark.parametrize("bad", ["DDR-ORD-101-009", b"DDR", ["A", 1], [["A"]]])
def test_unique_ids_refuses_a_single_string_and_non_strings(bad: object) -> None:
    with pytest.raises(TypeError):
        unique_ids(bad)  # type: ignore[arg-type]


def test_ledger_math_stays_finite_at_the_token_limit(tmp_path: Path) -> None:
    ledger = CostLedger(EgressLog(tmp_path / "e.jsonl"), 10.0, 1.0, 1.0, remote=True)
    assert math.isfinite(ledger.estimate_worst_case(MAX_TOKEN_COUNT, MAX_TOKEN_COUNT))
    with pytest.raises(BudgetExceeded):
        ledger.check_before_call(MAX_TOKEN_COUNT, 1)
