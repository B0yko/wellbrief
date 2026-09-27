"""Tests for the OpenAI-compatible HTTP client against fake servers on ephemeral loopback ports."""

from __future__ import annotations

import json
import re
import shutil
import ssl
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from fake_openai import FakeOpenAIServer, FakeResponse, completion
from wellbrief.egress import MAX_TOKEN_COUNT, InvalidBaseURL
from wellbrief.narrate.llm import (
    ALLOWED_EXTRA_KEYS,
    LLMConfigError,
    LLMConnectionError,
    LLMHTTPError,
    LLMResponseError,
    LLMTimeoutError,
    OpenAICompatibleClient,
    encode_body,
    strip_reasoning,
)

MESSAGES = [
    {"role": "system", "content": "Use only the evidence."},
    {"role": "user", "content": "What happened in the 17 1/2in section?"},
]


def test_successful_call_parses_text_and_usage(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(
        FakeResponse(completion("Pack-off at 1,402 m [DDR-ORD-101-009].", model="served-model"))
    )
    http = OpenAICompatibleClient(fake_server.base_url, "qwen3-test")
    result = http.chat(MESSAGES)
    assert result.text == "Pack-off at 1,402 m [DDR-ORD-101-009]."
    assert (result.prompt_tokens, result.completion_tokens) == (120, 40)
    assert result.provider_cost_usd is None
    assert result.model == "served-model"
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0
    request = fake_server.requests[0]
    assert (request.method, request.path) == ("POST", "/v1/chat/completions")
    assert result.bytes_sent == len(request.body)
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    body = request.json()
    assert body == {
        "model": "qwen3-test",
        "messages": MESSAGES,
        "max_tokens": 800,
        "temperature": 0.1,
        "stream": False,
    }


def test_bearer_token_is_sent_only_when_a_key_is_set(fake_server: FakeOpenAIServer) -> None:
    OpenAICompatibleClient(fake_server.base_url, "m", api_key="sk-test-123").chat(MESSAGES)
    OpenAICompatibleClient(fake_server.base_url, "m", api_key="").chat(MESSAGES)
    assert fake_server.requests[0].headers["authorization"] == "Bearer sk-test-123"
    assert "authorization" not in fake_server.requests[1].headers


def test_request_options_and_extra_fields(fake_server: FakeOpenAIServer) -> None:
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    http.chat(MESSAGES, max_tokens=64, temperature=0, extra={"top_p": 0.9, "seed": 7})
    body = fake_server.requests[0].json()
    assert (body["max_tokens"], body["temperature"], body["top_p"], body["seed"]) == (64, 0.0, 0.9, 7)
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "n_predict",
        "n",
        "n_cmpl",
        "stream",
        "stream_options",
        "temperature",
        "tools",
        "tool_choice",
        "response_format",
        "plugins",
        "web_search_options",
        "logprobs",
    ],
)
def test_extra_refuses_fields_outside_the_allowlist(fake_server: FakeOpenAIServer, key: str) -> None:
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    with pytest.raises(LLMConfigError, match=key):
        http.chat(MESSAGES, extra={key: 1})
    assert fake_server.requests == []


def test_extra_accepts_every_allowed_sampling_field(fake_server: FakeOpenAIServer) -> None:
    extra = {
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "seed": 7,
        "stop": ["</answer>"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.05,
        "repeat_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"effort": "none", "exclude": True},
        "reasoning_effort": "none",
    }
    assert set(extra) == ALLOWED_EXTRA_KEYS
    OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES, extra=extra)
    body = fake_server.requests[0].json()
    assert {key: body[key] for key in extra} == extra


@pytest.mark.parametrize(
    ("key", "value", "allowed"),
    [
        ("reasoning_effort", "none", True),
        ("reasoning", {"effort": "none"}, True),
        ("reasoning", {"enabled": False}, True),
        ("reasoning", {"enabled": False, "exclude": False}, True),
        ("reasoning_effort", "high", False),
        ("reasoning", {"effort": "high"}, False),
        ("reasoning", {"enabled": True}, False),
        ("reasoning", {"exclude": True}, False),
        ("reasoning", {"effort": "none", "max_tokens": 4000}, False),
        ("reasoning", "none", False),
    ],
)
def test_extra_may_only_switch_reasoning_off(
    fake_server: FakeOpenAIServer, key: str, value: object, allowed: bool
) -> None:
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    if allowed:
        http.chat(MESSAGES, extra={key: value})
        assert fake_server.requests[0].json()[key] == value
    else:
        with pytest.raises(LLMConfigError, match="switch reasoning off"):
            http.chat(MESSAGES, extra={key: value})
        assert fake_server.requests == []


@pytest.mark.parametrize(
    ("messages", "extra", "match"),
    [
        (MESSAGES, {"top_p": float("nan")}, "cannot be sent as JSON"),
        (MESSAGES, {"seed": object()}, "cannot be sent as JSON"),
        ([{"role": "user", "content": "bad \ud800 text"}], None, "cannot be sent as JSON"),
        (MESSAGES, [("top_p", 0.9)], "mapping"),
    ],
)
def test_bodies_that_cannot_be_serialised_are_refused_before_sending(
    fake_server: FakeOpenAIServer, messages: list[dict[str, str]], extra: object, match: str
) -> None:
    with pytest.raises(LLMConfigError, match=match):
        OpenAICompatibleClient(fake_server.base_url, "m").chat(messages, extra=extra)  # type: ignore[arg-type]
    assert fake_server.requests == []


def test_encode_body_matches_what_is_sent(fake_server: FakeOpenAIServer) -> None:
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    result = http.chat(MESSAGES, extra={"seed": 1})
    assert fake_server.requests[0].body == encode_body(http.build_body(MESSAGES, extra={"seed": 1}))
    assert result.bytes_sent == len(fake_server.requests[0].body)


def test_disable_thinking_sends_chat_template_kwargs(fake_server: FakeOpenAIServer) -> None:
    http = OpenAICompatibleClient(fake_server.base_url, "m", disable_thinking=True)
    http.chat(MESSAGES)
    http.chat(MESSAGES, extra={"chat_template_kwargs": {"custom": "x", "enable_thinking": True}})
    assert fake_server.requests[0].json()["chat_template_kwargs"] == {"enable_thinking": False}
    assert fake_server.requests[1].json()["chat_template_kwargs"] == {"custom": "x", "enable_thinking": False}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "<think>\nThe user asks about salt.\n</think>\n\nPack-off in Keldra Salt.",
            "Pack-off in Keldra Salt.",
        ),
        ("<think></think>Answer.", "Answer."),
        ("<THINK>upper case</THINK> Answer.", "Answer."),
        ("A <think>one</think>B<think>two</think> C", "A B C"),
        ("<think>reasoning that never finished because max_tokens ran out", ""),
        ("Answer first. <think>then a dangling block", "Answer first."),
        ("reasoning opened by the template\n</think>\n\nAnswer.", "Answer."),
        ("<think> a <think> nested </think> b </think> Answer.", "Answer."),
        ("<think> a <think> b </think> answer", "answer"),
        ("X<think>y</think>Z</think>W", "W"),
        ("No reasoning at all.", "No reasoning at all."),
        ("  padded  ", "padded"),
    ],
)
def test_strip_reasoning(raw: str, expected: str) -> None:
    assert strip_reasoning(raw) == expected


@pytest.mark.parametrize("unit", ["<think>", "</think>", "<think></think>", "</think><think>", "<think>x"])
def test_strip_reasoning_is_linear_on_adversarial_tags(unit: str) -> None:
    text = unit * (1_048_576 // len(unit))  # about 1 MiB of tags
    start = time.perf_counter()
    strip_reasoning(text)
    assert time.perf_counter() - start < 1.0


def test_reasoning_is_stripped_from_returned_text(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(
        FakeResponse(completion("<think>\nLet me check the ledger.\n</think>\n\n29.5 h of NPT."))
    )
    result = OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)
    assert result.text == "29.5 h of NPT."


def test_separate_reasoning_fields_are_ignored(fake_server: FakeOpenAIServer) -> None:
    body = completion("Answer.")
    body["choices"][0]["message"]["reasoning"] = "mlx-lm style reasoning"
    body["choices"][0]["message"]["reasoning_content"] = "llama.cpp style reasoning"
    fake_server.queue(FakeResponse(body))
    assert OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES).text == "Answer."


def test_null_content_and_text_parts(fake_server: FakeOpenAIServer) -> None:
    empty = completion("x", finish_reason="length")
    empty["choices"][0]["message"]["content"] = None
    parts = completion("x")
    parts["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "One "},
        {"type": "text", "text": "two."},
    ]
    fake_server.queue(FakeResponse(empty), FakeResponse(parts))
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    first = http.chat(MESSAGES)
    assert (first.text, first.finish_reason) == ("", "length")
    assert http.chat(MESSAGES).text == "One two."


def test_provider_cost_is_read_from_usage(fake_server: FakeOpenAIServer) -> None:
    byok = completion(
        cost=0.0001, extra_usage={"is_byok": True, "cost_details": {"upstream_inference_cost": 0.002}}
    )
    fake_server.queue(
        FakeResponse(completion(cost=0.0042)),
        FakeResponse(completion(cost=0)),
        FakeResponse(byok),
        FakeResponse(completion(extra_usage={"cost": "0.1"})),
    )
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    assert http.chat(MESSAGES).provider_cost_usd == pytest.approx(0.0042)
    assert http.chat(MESSAGES).provider_cost_usd == 0.0
    assert http.chat(MESSAGES).provider_cost_usd == pytest.approx(0.0021)
    assert http.chat(MESSAGES).provider_cost_usd is None


def test_out_of_range_usage_values_do_not_escape_as_overflow(fake_server: FakeOpenAIServer) -> None:
    huge = "9" * 400  # below Python's 4300-digit parsing limit, far beyond a float
    ok = '{"choices": [{"message": {"content": "ok"}}], "usage": %s}'
    fake_server.queue(
        FakeResponse(ok % f'{{"prompt_tokens": 1, "completion_tokens": 1, "cost": {huge}}}'),
        FakeResponse(ok % f'{{"prompt_tokens": {huge}, "completion_tokens": 1}}'),
        FakeResponse(ok % f'{{"prompt_tokens": 1, "completion_tokens": {MAX_TOKEN_COUNT + 1}}}'),
        FakeResponse(ok % f'{{"prompt_tokens": 1, "completion_tokens": {MAX_TOKEN_COUNT}}}'),
    )
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    assert http.chat(MESSAGES).provider_cost_usd is None  # unusable cost: priced from the tokens
    with pytest.raises(LLMResponseError, match=re.escape("usage.prompt_tokens")) as info:
        http.chat(MESSAGES)
    assert len(str(info.value)) < 500
    with pytest.raises(LLMResponseError, match=re.escape("usage.completion_tokens")):
        http.chat(MESSAGES)
    assert http.chat(MESSAGES).completion_tokens == MAX_TOKEN_COUNT


def test_lone_surrogates_from_the_server_are_replaced(fake_server: FakeOpenAIServer) -> None:
    body = json.dumps(completion("Answer \ud800 [DOC-1].", model="served-\udc80", finish_reason="stop\ud800"))
    assert "\\ud800" in body  # json.dumps escapes the lone surrogate, as a server would
    fake_server.queue(
        FakeResponse(body),
        FakeResponse(b'{"error": {"message": "boom \\udc80"}}', status=500),
        FakeResponse(b'{"error": {"message": "late \\ud800 failure"}}'),
    )
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    result = http.chat(MESSAGES)
    assert (result.text, result.model, result.finish_reason) == (
        "Answer \ufffd [DOC-1].",
        "served-\ufffd",
        "stop\ufffd",
    )
    for failing in (LLMHTTPError, LLMResponseError):
        with pytest.raises(failing) as info:
            http.chat(MESSAGES)
        str(info.value).encode("utf-8")  # printable and writable
        assert "\ufffd" in str(info.value)


def test_missing_usage_gives_unknown_token_counts(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(FakeResponse(completion(prompt_tokens=None, completion_tokens=None)))
    result = OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)
    assert (result.prompt_tokens, result.completion_tokens, result.provider_cost_usd) == (None, None, None)


def test_http_500_is_a_clear_error(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(FakeResponse({"error": {"code": 500, "message": "model crashed"}}, status=500))
    with pytest.raises(LLMHTTPError, match="HTTP 500: model crashed") as info:
        OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)
    assert info.value.status == 500
    assert info.value.may_have_billed
    assert info.value.bytes_sent > 0


def test_http_4xx_errors_are_not_billable_and_carry_hints(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(
        FakeResponse({"error": {"message": "No auth credentials found"}}, status=401),
        FakeResponse("not found", status=404),
        FakeResponse({"error": {"message": "timeout"}}, status=408),
    )
    http = OpenAICompatibleClient(fake_server.base_url, "m", api_key="sk-secret-value")
    with pytest.raises(LLMHTTPError, match="WELLBRIEF_LLM_API_KEY") as unauthorized:
        http.chat(MESSAGES)
    assert not unauthorized.value.may_have_billed
    with pytest.raises(LLMHTTPError, match="WELLBRIEF_LLM_BASE_URL"):
        http.chat(MESSAGES)
    with pytest.raises(LLMHTTPError) as timeout:
        http.chat(MESSAGES)
    assert timeout.value.may_have_billed


def test_api_key_is_masked_in_error_text(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(FakeResponse({"error": {"message": "bad key sk-secret-value"}}, status=400))
    with pytest.raises(LLMHTTPError) as info:
        OpenAICompatibleClient(fake_server.base_url, "m", api_key="sk-secret-value").chat(MESSAGES)
    assert "sk-secret-value" not in str(info.value)
    assert "***" in str(info.value)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"<html>502 Bad Gateway</html>", "not JSON"),
        (b"", "not JSON"),
        (b"\xff\xfe", "not JSON"),
        (b"[1, 2]", "not a JSON object"),
        (b'{"id": "x"}', "choices"),
        (b'{"choices": []}', "choices"),
        (b'{"choices": [{"index": 0}]}', "message"),
        (b'{"choices": [{"message": {"content": 42}}]}', "content"),
        (b'{"choices": [{"message": {"content": "ok"}}], "usage": "lots"}', "usage"),
        (b'{"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": -3}}', "prompt_tokens"),
        (b'{"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1.5}}', "completion"),
    ],
)
def test_malformed_responses_are_clear_errors(fake_server: FakeOpenAIServer, body: bytes, match: str) -> None:
    fake_server.queue(FakeResponse(body))
    with pytest.raises(LLMResponseError, match=match) as info:
        OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)
    assert info.value.may_have_billed


def test_error_inside_a_200_response_is_raised(fake_server: FakeOpenAIServer) -> None:
    errored = completion("partial", finish_reason="error")
    fake_server.queue(
        FakeResponse({"error": {"code": 502, "message": "provider disconnected"}}),
        FakeResponse(errored),
    )
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    with pytest.raises(LLMResponseError, match="provider disconnected"):
        http.chat(MESSAGES)
    with pytest.raises(LLMResponseError, match="finish_reason 'error'"):
        http.chat(MESSAGES)


def test_redirects_are_refused(fake_server: FakeOpenAIServer, second_server: FakeOpenAIServer) -> None:
    target = f"{second_server.base_url}/chat/completions"
    fake_server.queue(FakeResponse("", status=307, headers={"Location": target}))
    with pytest.raises(LLMHTTPError, match="redirect") as info:
        OpenAICompatibleClient(fake_server.base_url, "m", api_key="sk-test").chat(MESSAGES)
    assert info.value.status == 307
    assert len(fake_server.requests) == 1
    assert second_server.requests == []


def test_read_timeout_is_a_clear_error(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(FakeResponse(completion(), delay_s=1.0))
    with pytest.raises(LLMTimeoutError, match=re.escape("did not answer within 0.2 s")) as info:
        OpenAICompatibleClient(fake_server.base_url, "m", timeout=0.2).chat(MESSAGES)
    assert info.value.request_sent
    assert info.value.may_have_billed


def test_connection_refused_is_a_clear_error(closed_port: int) -> None:
    with pytest.raises(LLMConnectionError, match=re.escape("Could not reach 127.0.0.1")) as info:
        OpenAICompatibleClient(f"http://127.0.0.1:{closed_port}/v1", "m", timeout=2).chat(MESSAGES)
    assert not info.value.request_sent
    assert not info.value.may_have_billed
    assert info.value.bytes_sent == 0


def test_proxy_settings_are_ignored(
    fake_server: FakeOpenAIServer, closed_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead_proxy = f"http://127.0.0.1:{closed_port}"
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, dead_proxy)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    assert OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES).text
    assert len(fake_server.requests) == 1


def test_oversized_response_is_refused(
    fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wellbrief.narrate import llm

    monkeypatch.setattr(llm, "MAX_RESPONSE_BYTES", 64)
    fake_server.queue(FakeResponse(completion("x" * 200)))
    with pytest.raises(LLMResponseError, match="larger than 64 bytes"):
        OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"base_url": "ftp://127.0.0.1/v1"}, InvalidBaseURL),
        ({"base_url": "http://user:pw@127.0.0.1/v1"}, InvalidBaseURL),
        ({"model": " "}, LLMConfigError),
        ({"timeout": 0}, LLMConfigError),
        ({"timeout": float("inf")}, LLMConfigError),
    ],
)
def test_constructor_rejects_bad_arguments(kwargs: dict[str, object], error: type[Exception]) -> None:
    args: dict[str, object] = {"base_url": "http://127.0.0.1:8080/v1", "model": "m", **kwargs}
    with pytest.raises(error):
        OpenAICompatibleClient(**args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("messages", "options"),
    [
        ([], {}),
        ("hello", {}),
        ([{"content": "no role"}], {}),
        ([{"role": "user", "content": 3}], {}),
        (MESSAGES, {"max_tokens": 0}),
        (MESSAGES, {"max_tokens": True}),
        (MESSAGES, {"max_tokens": MAX_TOKEN_COUNT + 1}),
        (MESSAGES, {"temperature": -0.1}),
    ],
)
def test_bad_requests_are_refused_before_sending(
    fake_server: FakeOpenAIServer, messages: object, options: dict[str, object]
) -> None:
    with pytest.raises(LLMConfigError):
        OpenAICompatibleClient(fake_server.base_url, "m").chat(messages, **options)  # type: ignore[arg-type]
    assert fake_server.requests == []


def test_missing_ca_bundle_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(LLMConfigError, match="WELLBRIEF_CA_BUNDLE"):
        OpenAICompatibleClient("https://127.0.0.1:8443/v1", "m", ca_bundle=tmp_path / "missing.pem")
    garbage = tmp_path / "garbage.pem"
    garbage.write_text("not a certificate", encoding="utf-8")
    with pytest.raises(LLMConfigError, match="cannot be loaded"):
        OpenAICompatibleClient("https://127.0.0.1:8443/v1", "m", ca_bundle=garbage)


@pytest.fixture
def self_signed_cert(tmp_path: Path) -> tuple[Path, Path]:
    """A throwaway self-signed certificate for 127.0.0.1, generated with the openssl CLI."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl CLI not available")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(key), "-out", str(cert),
            "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return cert, key


def test_https_verifies_certificates_and_accepts_a_ca_bundle(self_signed_cert: tuple[Path, Path]) -> None:
    cert, key = self_signed_cert
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    server = FakeOpenAIServer(tls=tls).start()
    try:
        with pytest.raises(LLMConnectionError, match="certificate verification failed") as info:
            OpenAICompatibleClient(server.base_url, "m", timeout=5).chat(MESSAGES)
        assert "WELLBRIEF_CA_BUNDLE" in str(info.value)
        assert not info.value.may_have_billed
        assert server.requests == []
        result = OpenAICompatibleClient(server.base_url, "m", timeout=5, ca_bundle=cert).chat(MESSAGES)
        assert result.text
        assert len(server.requests) == 1
    finally:
        server.stop()


def test_dropped_connection_is_a_clear_error(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(FakeResponse(drop=True))
    with pytest.raises(LLMConnectionError, match="failed while reading the response") as info:
        OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)
    assert info.value.request_sent
    assert info.value.may_have_billed


def test_connect_timeout_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    http = OpenAICompatibleClient("http://127.0.0.1:9/v1", "m", timeout=3)

    def slow_connect(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(http._opener, "open", slow_connect)
    pattern = re.escape("Connecting to 127.0.0.1 timed out after 3 s")
    with pytest.raises(LLMTimeoutError, match=pattern) as info:
        http.chat(MESSAGES)
    assert not info.value.may_have_billed


def test_redirects_and_long_error_bodies(fake_server: FakeOpenAIServer) -> None:
    fake_server.queue(
        FakeResponse("", status=302, headers={"Location": "https://elsewhere.example/v1"}),
        FakeResponse("x" * 5_000, status=503),
        FakeResponse({"error": "plain string error"}, status=500),
    )
    http = OpenAICompatibleClient(fake_server.base_url, "m")
    with pytest.raises(LLMHTTPError) as redirect:
        http.chat(MESSAGES)
    assert not redirect.value.may_have_billed
    with pytest.raises(LLMHTTPError) as long_body:
        http.chat(MESSAGES)
    assert long_body.value.may_have_billed
    assert len(str(long_body.value)) < 400
    assert "..." in str(long_body.value)
    with pytest.raises(LLMHTTPError, match="plain string error"):
        http.chat(MESSAGES)


@pytest.mark.parametrize(
    "body",
    [
        b'{"choices": [{"message": {"content": [{"type": "image_url", "image_url": {}}]}}]}',
        b"[" * 100_000 + b"]" * 100_000,
        # json.loads itself refuses integers longer than 4300 digits (a ValueError).
        b'{"choices": [{"message": {"content": "ok"}}], "n": ' + b"9" * 5_000 + b"}",
    ],
)
def test_unusual_bodies_are_malformed_not_crashes(fake_server: FakeOpenAIServer, body: bytes) -> None:
    fake_server.queue(FakeResponse(body))
    with pytest.raises(LLMResponseError):
        OpenAICompatibleClient(fake_server.base_url, "m").chat(MESSAGES)


def test_api_key_is_validated_and_trimmed(fake_server: FakeOpenAIServer) -> None:
    with pytest.raises(LLMConfigError, match="WELLBRIEF_LLM_API_KEY"):
        OpenAICompatibleClient(fake_server.base_url, "m", api_key="sk-abc def")
    OpenAICompatibleClient(fake_server.base_url, "m", api_key="  sk-abc\n").chat(MESSAGES)
    assert fake_server.requests[0].headers["authorization"] == "Bearer sk-abc"


def test_messages_must_be_mappings(fake_server: FakeOpenAIServer) -> None:
    with pytest.raises(LLMConfigError, match="not a mapping"):
        OpenAICompatibleClient(fake_server.base_url, "m").chat([("user", "hi")])  # type: ignore[list-item]
    assert fake_server.requests == []
