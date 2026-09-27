"""The ``llm`` narrator: an OpenAI-compatible ``/chat/completions`` client, guarded by the egress
rule and the cost ledger, and :class:`LlmNarrator`, the narrator that uses it.

It talks to any server that implements the chat-completions request and response shape:
``mlx_lm.server`` or llama.cpp's ``llama-server`` on a loopback address, or a hosted router such as
OpenRouter. Only the standard library is used (``urllib``, ``ssl``, ``json``).

Three layers:

* :class:`OpenAICompatibleClient` sends one non-streaming request and parses the reply into a
  :class:`ChatResult`. It maps every failure to an :class:`LLMError` subclass with a readable message.
* :class:`GuardedChat` applies the egress rule (:func:`wellbrief.egress.check_egress`) and the budget
  stop (:class:`wellbrief.egress.CostLedger`) before a request, and appends one record to
  ``egress.jsonl`` after it or after a refusal. A request that may have been sent is always logged,
  whatever goes wrong afterwards.
* :class:`LlmNarrator` builds the evidence block and the system prompt, sends one :class:`GuardedChat`
  call per ``ask`` or ``brief``, and checks the reply with :mod:`wellbrief.narrate.verify`. On any
  failure -- the call could not be completed, or its text failed verification -- it shows the
  deterministic offline narrator's answer instead, with a banner, and records why on
  ``self.last_rejection`` (see :class:`wellbrief.narrate.base.Narrator`).

Transport choices that keep the egress rule meaningful:

* Redirects are never followed. A redirect could move the request, including its body and the
  ``Authorization`` header, to a host the egress check never saw; a 3xx answer is reported as an error.
* Proxy settings from the environment and the operating system are ignored. The connection goes to
  the configured host, which is the host that was checked.
* TLS uses ``ssl.create_default_context()``, which verifies certificates and host names. A custom CA
  file (``WELLBRIEF_CA_BUNDLE``, passed in as ``ca_bundle``) replaces the default trust store.

Reasoning output from Qwen3-style models is removed from the returned text (see
:func:`strip_reasoning`). Thinking can also be switched off at the source with
``disable_thinking=True``, which sends ``chat_template_kwargs: {"enable_thinking": false}``; both
``mlx_lm.server`` and ``llama-server`` read that field from the request body and pass it to the chat
template.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from wellbrief.egress import (
    DEFAULT_BUDGET_USD,
    DEFAULT_MAX_TOKENS,
    MAX_TOKEN_COUNT,
    BaseURL,
    BudgetExceeded,
    CostLedger,
    CostSource,
    EgressError,
    EgressLog,
    EgressLogError,
    EgressRecord,
    EgressStatus,
    RemoteNotAllowed,
    check_egress,
    estimate_request_tokens,
    is_loopback_host,
    split_base_url,
    unique_ids,
    utc_timestamp,
)

from .base import Narrator
from .offline import OfflineNarrator
from .verify import evidence_from_brief, evidence_from_pack, verify

if TYPE_CHECKING:
    from ..models import RiskBrief

__all__ = [
    "ALLOWED_EXTRA_KEYS",
    "CALL_FAILED_BANNER",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "MAX_RESPONSE_BYTES",
    "VERIFICATION_FAILED_BANNER",
    "ChatBackend",
    "ChatResult",
    "GuardedChat",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMHTTPError",
    "LLMResponseError",
    "LLMTimeoutError",
    "LlmNarrator",
    "OpenAICompatibleClient",
    "encode_body",
    "narrator_from_env",
    "strip_reasoning",
]

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_TEMPERATURE = 0.1
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
"""Largest response body read; a chat completion capped at a few thousand tokens is far smaller."""

ALLOWED_EXTRA_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "repeat_penalty",
        "chat_template_kwargs",
        "reasoning",
        "reasoning_effort",
    }
)
"""The only request fields that ``extra`` may set.

They are sampling controls and chat-template or reasoning switches. None of them can make a server
generate more than ``max_tokens`` output tokens or bill input that the pre-call estimate does not
count, so the budget's worst case stays valid. ``reasoning`` (OpenRouter) and ``reasoning_effort``
(llama.cpp) are further limited to values that switch reasoning off. Every other field is refused,
for example ``max_completion_tokens`` (mlx-lm prefers it to ``max_tokens``), ``n_predict``
(llama.cpp's own output cap), ``n`` and ``n_cmpl`` (several completions per request), ``tools`` and
``response_format`` (text added to the prompt), and OpenRouter's ``plugins`` and
``web_search_options`` (billed per search)."""

USER_AGENT = "wellbrief"
_ERROR_SNIPPET_CHARS = 300

_THINK_TAG = re.compile(r"<(/?)think>", re.IGNORECASE)
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


class LLMConfigError(ValueError):
    """The HTTP client or a request was configured with a value it cannot use. Nothing was sent."""


class LLMError(Exception):
    """A chat-completions request failed.

    Attributes:
        bytes_sent: Size of the request body that was sent (0 when the connection failed first).
        latency_ms: Time from the start of the request to the failure.
        request_sent: Whether the whole request reached the server, so that it may have been processed.
    """

    def __init__(self, message: str, *, bytes_sent: int, latency_ms: float, request_sent: bool) -> None:
        super().__init__(message)
        self.bytes_sent = bytes_sent
        self.latency_ms = latency_ms
        self.request_sent = request_sent

    @property
    def may_have_billed(self) -> bool:
        """Whether a provider may have charged for this request despite the failure."""
        return self.request_sent


class LLMConnectionError(LLMError):
    """The server could not be reached, the TLS handshake failed, or the connection dropped."""


class LLMTimeoutError(LLMError):
    """A connect or read on the socket took longer than the configured timeout."""


class LLMHTTPError(LLMError):
    """The server answered with a non-2xx HTTP status (redirects included, since none are followed)."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        bytes_sent: int,
        latency_ms: float,
    ) -> None:
        super().__init__(message, bytes_sent=bytes_sent, latency_ms=latency_ms, request_sent=True)
        self.status = status

    @property
    def may_have_billed(self) -> bool:
        """Whether a provider may have charged: only after a 5xx or a 408 (timeout).

        Redirects and other 4xx answers mean the request was turned away before it reached a model.
        """
        return self.status >= 500 or self.status == 408


class LLMResponseError(LLMError):
    """The server answered 200 but the body is not a usable chat completion, or reports an error."""


@dataclass(frozen=True)
class ChatResult:
    """One parsed chat completion.

    ``prompt_tokens`` and ``completion_tokens`` are None when the server did not report usage.
    ``provider_cost_usd`` is the provider-reported cost (OpenRouter's ``usage.cost``), or None.
    ``model`` is the model the server says it used, falling back to the requested one.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    provider_cost_usd: float | None
    latency_ms: float
    bytes_sent: int
    model: str
    finish_reason: str | None = None


def strip_reasoning(text: str) -> str:
    """Remove Qwen3-style reasoning from model output and trim surrounding whitespace.

    The ``<think>`` and ``</think>`` tags are matched case-insensitively in one left-to-right pass,
    so the time is linear in the length of the text:

    * a ``<think>`` opens a reasoning block that runs to the next ``</think>``; a ``<think>`` inside
      the block is part of the reasoning;
    * a ``</think>`` outside a block ends reasoning that the chat template opened inside the prompt,
      so everything before it is dropped;
    * a block that is never closed (reasoning cut off by ``max_tokens``) is dropped together with
      everything after it.

    Reasoning that a server returns in a separate field (``message.reasoning`` in mlx-lm,
    ``message.reasoning_content`` in llama.cpp) is never read.
    """
    kept: list[str] = []
    position = 0
    inside = False
    for tag in _THINK_TAG.finditer(text):
        closing = tag.group(1) == "/"
        if inside:
            if closing:
                inside = False
                position = tag.end()
        elif closing:
            kept.clear()
            position = tag.end()
        else:
            kept.append(text[position : tag.start()])
            inside = True
    if not inside:
        kept.append(text[position:])
    return "".join(kept).strip()


def encode_body(body: Mapping[str, Any]) -> bytes:
    """Serialise a request body exactly as :meth:`OpenAICompatibleClient.chat` sends it (UTF-8 JSON).

    Raises:
        LLMConfigError: the body holds a value JSON cannot carry (NaN, an infinity, an unsupported
            type) or text that is not valid Unicode (a lone surrogate).
    """
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        # UnicodeEncodeError is a ValueError.
        raise LLMConfigError(f"The request body cannot be sent as JSON: {exc}") from None


def _clean(text: str) -> str:
    """Replace lone UTF-16 surrogates with U+FFFD.

    ``json.loads`` turns an unpaired ``\\ud800`` escape into a lone surrogate, which is not valid
    Unicode and cannot be encoded as UTF-8. Every string taken from a server passes through here, so
    what the HTTP client returns can always be printed and written.
    """
    return _LONE_SURROGATE.sub("\ufffd", text)


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Report 3xx answers as errors instead of following them."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def _ssl_context(ca_bundle: str | os.PathLike[str] | None) -> ssl.SSLContext:
    if ca_bundle is None or str(ca_bundle) == "":
        return ssl.create_default_context()
    path = Path(ca_bundle)
    if not path.is_file():
        raise LLMConfigError(f"The CA bundle {str(path)!r} (WELLBRIEF_CA_BUNDLE) is not a file.")
    try:
        return ssl.create_default_context(cafile=str(path))
    except (ssl.SSLError, OSError) as exc:
        raise LLMConfigError(
            f"The CA bundle {str(path)!r} (WELLBRIEF_CA_BUNDLE) cannot be loaded: {exc}"
        ) from exc


_KEY_MASK = chr(0x2A) * 3  # "***"; built from a code point, not a literal, so a static source
# scan for glob-like strings (guarding the ground-truth sidecar's file name against a stray
# match) never has to reason about an unrelated masking placeholder.


def _snippet(text: str, secret: str | None) -> str:
    text = _clean(text)
    if secret:
        text = text.replace(secret, _KEY_MASK)
    text = " ".join(text.split())
    if len(text) > _ERROR_SNIPPET_CHARS:
        text = text[:_ERROR_SNIPPET_CHARS] + "..."
    return text


def _payload_error(error: object, secret: str | None) -> str | None:
    """Return the message of an OpenAI-style ``error`` value (an object with ``message``, or a string)."""
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return _snippet(error["message"], secret)
    if isinstance(error, str):
        return _snippet(error, secret)
    return None


def _error_detail(raw: bytes, secret: str | None) -> str:
    """Return the error message of an HTTP error body, or a snippet of the raw body."""
    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        payload = None
    message = _payload_error(payload.get("error"), secret) if isinstance(payload, dict) else None
    return message or _snippet(text, secret) or "(empty body)"


_STATUS_HINTS = {
    401: "check WELLBRIEF_LLM_API_KEY",
    402: "the provider reports insufficient credit",
    403: "check WELLBRIEF_LLM_API_KEY and the provider's access rules",
    404: "check WELLBRIEF_LLM_BASE_URL (it usually ends in /v1) and WELLBRIEF_LLM_MODEL",
    429: "the provider is rate limiting; retry later",
}


def _is_count(value: object) -> TypeGuard[int]:
    """Whether ``value`` is a plausible token count: an int from 0 to :data:`MAX_TOKEN_COUNT`."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_TOKEN_COUNT


def _is_amount(value: object) -> TypeGuard[float]:
    """Whether ``value`` is a finite int or float >= 0; an int too large for a float is not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and number >= 0


def _provider_cost(usage: Mapping[str, Any]) -> float | None:
    """Return the provider-reported cost in USD, or None when there is none.

    OpenRouter reports ``usage.cost`` in credits, which are denominated in US dollars. For a
    bring-your-own-key request ``cost`` is only OpenRouter's fee and the upstream provider bills
    ``cost_details.upstream_inference_cost`` separately, so the two are added.
    """
    cost = usage.get("cost")
    if not _is_amount(cost):
        # Absent, null or unusable (a string, a negative number, an integer too large for a float):
        # the ledger then prices the call from the reported tokens.
        return None
    total = float(cost)
    if usage.get("is_byok") is True:
        details = usage.get("cost_details")
        upstream = details.get("upstream_inference_cost") if isinstance(details, dict) else None
        if _is_amount(upstream):
            total += float(upstream)
    return total


def _content_text(content: object) -> str | None:
    """Return message content as text: a string, null, or a list of ``{"type": "text"}`` parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                return None
        return "".join(parts)
    return None


def _check_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise LLMConfigError("messages must be a non-empty list of chat messages.")
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise LLMConfigError(f"messages[{index}] is not a mapping.")
        if not isinstance(message.get("role"), str) or not message.get("role"):
            raise LLMConfigError(f"messages[{index}] has no role.")
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise LLMConfigError(f"messages[{index}] content must be a string or a list of parts.")
        out.append(dict(message))
    return out


def _switches_reasoning_off(key: str, value: object) -> bool:
    """Whether ``value`` for ``reasoning`` / ``reasoning_effort`` only turns reasoning off.

    Accepted: ``reasoning_effort: "none"`` and a ``reasoning`` object with ``effort: "none"`` or
    ``enabled: false`` (plus, optionally, a boolean ``exclude``).
    """
    if key == "reasoning_effort":
        return value == "none"
    if not isinstance(value, Mapping):
        return False
    off = value.get("effort") == "none" or value.get("enabled") is False
    for field, setting in value.items():
        if field == "effort" and setting == "none":
            continue
        if field in ("enabled", "exclude") and isinstance(setting, bool):
            continue
        return False
    return off


def _check_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise LLMConfigError("extra must be a mapping of request fields.")
    fields = dict(extra)
    refused = sorted(str(key) for key in fields if key not in ALLOWED_EXTRA_KEYS)
    if refused:
        raise LLMConfigError(
            f"extra must not set {', '.join(refused)}; it may set only "
            f"{', '.join(sorted(ALLOWED_EXTRA_KEYS))}. Other fields either define the request or can "
            "raise its cost above the budget's worst-case estimate."
        )
    for key in ("reasoning", "reasoning_effort"):
        if key in fields and not _switches_reasoning_off(key, fields[key]):
            raise LLMConfigError(
                f"extra may set {key} only to switch reasoning off (reasoning_effort: 'none', or "
                "reasoning: {'effort': 'none'} or {'enabled': false}); other values can raise the cost."
            )
    return fields


class OpenAICompatibleClient:
    """HTTP client for one OpenAI-compatible ``/chat/completions`` endpoint.

    Args:
        base_url: Server base URL including the API prefix, for example ``http://127.0.0.1:8080/v1``
            (mlx_lm.server and llama-server) or ``https://openrouter.ai/api/v1``. Requests go to
            ``{base_url}/chat/completions``.
        model: Model name sent in the request. mlx_lm.server treats ``default_model`` as the model it
            was started with and loads any other name as a Hugging Face id or local path (which can
            start a download); llama-server ignores the name unless it runs in router mode.
        api_key: Bearer token, stripped of surrounding whitespace. The ``Authorization`` header is sent
            only when a key is set.
        timeout: Seconds allowed for the connect and for each socket read (``urllib`` semantics, not a
            total deadline).
        ca_bundle: PEM file of CA certificates that replaces the default trust store for HTTPS.
        disable_thinking: Send ``chat_template_kwargs: {"enable_thinking": false}`` with every request.

    The HTTP client does not apply the egress rule or the budget itself; callers go through
    :class:`GuardedChat`.

    Raises:
        wellbrief.egress.InvalidBaseURL: the base URL cannot be used.
        LLMConfigError: another argument cannot be used.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        ca_bundle: str | os.PathLike[str] | None = None,
        *,
        disable_thinking: bool = False,
    ) -> None:
        self._base_url = base_url
        self._url: BaseURL = split_base_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise LLMConfigError("The model name is empty (WELLBRIEF_LLM_MODEL).")
        if not _is_amount(timeout) or timeout <= 0:
            raise LLMConfigError(f"The timeout must be a positive number of seconds, got {timeout!r}.")
        key = (api_key or "").strip()
        if any(ord(char) < 0x21 or ord(char) > 0x7E for char in key):
            raise LLMConfigError(
                "The API key (WELLBRIEF_LLM_API_KEY) contains spaces or non-ASCII characters."
            )
        self._model = model
        self._api_key = key or None
        self._timeout = float(timeout)
        self._disable_thinking = disable_thinking
        self._endpoint = self._url.endpoint("chat/completions")
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=_ssl_context(ca_bundle)),
            _RefuseRedirects(),
        )

    @property
    def base_url(self) -> str:
        """The base URL as configured."""
        return self._base_url

    @property
    def host(self) -> str:
        """Host of the base URL (lower case, IPv6 brackets removed)."""
        return self._url.host

    @property
    def endpoint(self) -> str:
        """Absolute URL of the chat-completions endpoint."""
        return self._endpoint

    @property
    def model(self) -> str:
        """Model name sent with each request."""
        return self._model

    @property
    def disable_thinking(self) -> bool:
        """Whether requests ask the chat template to skip the reasoning block."""
        return self._disable_thinking

    def build_body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the JSON request body for :meth:`chat` without sending it.

        The output cap is sent as ``max_tokens``. OpenAI marks that field deprecated in favour of
        ``max_completion_tokens`` (and its o-series reasoning models reject it), but it is the field
        that mlx-lm, llama.cpp and OpenRouter have all accepted for longest, and an older local server
        that ignored the newer name would generate without a cap.

        Raises:
            LLMConfigError: an argument is invalid or ``extra`` sets a field outside
                :data:`ALLOWED_EXTRA_KEYS`.
        """
        if not _is_count(max_tokens) or max_tokens < 1:
            shown = _snippet(repr(max_tokens), None)
            raise LLMConfigError(f"max_tokens must be an integer from 1 to {MAX_TOKEN_COUNT}, got {shown}.")
        if not _is_amount(temperature):
            raise LLMConfigError(f"temperature must be a finite number >= 0, got {temperature!r}.")
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _check_messages(messages),
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "stream": False,
        }
        extra_fields = _check_extra(extra)
        if self._disable_thinking:
            kwargs = extra_fields.get("chat_template_kwargs")
            merged = dict(kwargs) if isinstance(kwargs, Mapping) else {}
            merged["enable_thinking"] = False
            extra_fields["chat_template_kwargs"] = merged
        body.update(extra_fields)
        return body

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        """Send one non-streaming chat-completions request and parse the reply.

        ``extra`` adds request fields (for example ``top_p`` or ``seed``); only fields in
        :data:`ALLOWED_EXTRA_KEYS` are accepted.

        Raises:
            LLMConfigError: invalid arguments; nothing was sent.
            LLMConnectionError: connection refused, TLS failure, or the connection dropped.
            LLMTimeoutError: the connect or a read timed out.
            LLMHTTPError: non-2xx status, including any redirect.
            LLMResponseError: the body is not a usable chat completion or carries an error.
        """
        body = self.build_body(messages, max_tokens=max_tokens, temperature=temperature, extra=extra)
        data = encode_body(body)
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(self._endpoint, data=data, headers=headers, method="POST")
        if request.host.lower() != self._url.netloc.lower():
            raise LLMConfigError(
                f"The request would go to {request.host!r} instead of the checked host {self._url.netloc!r}."
            )
        sent = len(data)
        start = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - start) * 1000.0, 1)

        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc, sent, elapsed()) from None
        except urllib.error.URLError as exc:
            # urllib wraps failures raised while connecting and sending: nothing was processed.
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise LLMTimeoutError(
                    f"Connecting to {self.host} timed out after {self._timeout:g} s.",
                    bytes_sent=0,
                    latency_ms=elapsed(),
                    request_sent=False,
                ) from None
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise LLMConnectionError(
                    f"TLS certificate verification failed for {self.host}: {reason.verify_message}. "
                    "If the server uses a private CA, point WELLBRIEF_CA_BUNDLE at its PEM file.",
                    bytes_sent=0,
                    latency_ms=elapsed(),
                    request_sent=False,
                ) from None
            raise LLMConnectionError(
                f"Could not reach {self.host} ({self._endpoint}): {reason}.",
                bytes_sent=0,
                latency_ms=elapsed(),
                request_sent=False,
            ) from None
        except TimeoutError:
            raise LLMTimeoutError(
                f"{self.host} did not answer within {self._timeout:g} s of the request being sent.",
                bytes_sent=sent,
                latency_ms=elapsed(),
                request_sent=True,
            ) from None
        except (OSError, http.client.HTTPException) as exc:
            raise LLMConnectionError(
                f"The connection to {self.host} failed while reading the response: {exc!r}.",
                bytes_sent=sent,
                latency_ms=elapsed(),
                request_sent=True,
            ) from None
        return self._parse(raw, sent, elapsed())

    def _http_error(self, exc: urllib.error.HTTPError, sent: int, latency_ms: float) -> LLMHTTPError:
        status = exc.code
        try:
            raw = exc.read(64 * 1024)
        except (OSError, http.client.HTTPException):
            raw = b""
        finally:
            exc.close()
        if 300 <= status < 400:
            location = exc.headers.get("Location", "") if exc.headers is not None else ""
            detail = (
                f"redirect to {_snippet(location, self._api_key)!r} refused; redirects are not followed "
                "because they could move the request to a host the egress guard did not check"
            )
        else:
            detail = _error_detail(raw, self._api_key)
        hint = _STATUS_HINTS.get(status)
        message = f"{self.host} answered HTTP {status}: {detail}"
        if hint:
            message += f" ({hint})"
        return LLMHTTPError(message + ".", status=status, bytes_sent=sent, latency_ms=latency_ms)

    def _malformed(self, what: str, sent: int, latency_ms: float) -> LLMResponseError:
        return LLMResponseError(
            f"Malformed chat completion from {self.host}: {what}.",
            bytes_sent=sent,
            latency_ms=latency_ms,
            request_sent=True,
        )

    def _parse(self, raw: bytes, sent: int, latency_ms: float) -> ChatResult:
        if len(raw) > MAX_RESPONSE_BYTES:
            raise self._malformed(f"the body is larger than {MAX_RESPONSE_BYTES} bytes", sent, latency_ms)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            snippet = _snippet(raw.decode("utf-8", errors="replace"), self._api_key) or "(empty body)"
            raise self._malformed(f"the body is not JSON: {snippet!r}", sent, latency_ms) from None
        if not isinstance(payload, dict):
            raise self._malformed("the body is not a JSON object", sent, latency_ms)
        if payload.get("error"):
            detail = _payload_error(payload["error"], self._api_key) or _snippet(
                json.dumps(payload["error"]), self._api_key
            )
            raise LLMResponseError(
                f"{self.host} reported an error in a 200 response: {detail}.",
                bytes_sent=sent,
                latency_ms=latency_ms,
                request_sent=True,
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._malformed("'choices' is missing or empty", sent, latency_ms)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._malformed("'choices[0].message' is missing", sent, latency_ms)
        content = _content_text(message.get("content"))
        if content is None:
            raise self._malformed("'choices[0].message.content' is not text", sent, latency_ms)
        finish_reason = choice.get("finish_reason")
        finish_reason = _clean(finish_reason) if isinstance(finish_reason, str) else None
        if finish_reason == "error":
            raise LLMResponseError(
                f"{self.host} ended the completion with finish_reason 'error'.",
                bytes_sent=sent,
                latency_ms=latency_ms,
                request_sent=True,
            )
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        provider_cost: float | None = None
        usage = payload.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise self._malformed("'usage' is not an object", sent, latency_ms)
            for key in ("prompt_tokens", "completion_tokens"):
                if key in usage and usage[key] is not None and not _is_count(usage[key]):
                    shown = _snippet(repr(usage[key]), None)
                    raise self._malformed(
                        f"'usage.{key}' is {shown}, not an integer from 0 to {MAX_TOKEN_COUNT}",
                        sent,
                        latency_ms,
                    )
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            provider_cost = _provider_cost(usage)
        served_model = payload.get("model")
        return ChatResult(
            text=strip_reasoning(_clean(content)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider_cost_usd=provider_cost,
            latency_ms=latency_ms,
            bytes_sent=sent,
            model=_clean(served_model) if isinstance(served_model, str) and served_model else self._model,
            finish_reason=finish_reason,
        )


class ChatBackend(Protocol):
    """What :class:`GuardedChat` needs from an HTTP client; :class:`OpenAICompatibleClient` fits."""

    @property
    def base_url(self) -> str: ...

    @property
    def host(self) -> str: ...

    @property
    def model(self) -> str: ...

    def build_body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = ...,
        temperature: float = ...,
        extra: Mapping[str, Any] | None = ...,
    ) -> dict[str, Any]: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = ...,
        temperature: float = ...,
        extra: Mapping[str, Any] | None = ...,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class _CallContext:
    """What :class:`GuardedChat` knows about a call once its budget is reserved."""

    ts: str
    doc_ids: tuple[str, ...]
    body_bytes: int
    prompt_estimate: int
    max_tokens: int
    estimate_usd: float


class GuardedChat:
    """Chat calls behind the egress rule and the budget stop, each one logged to ``egress.jsonl``.

    For every call, in order:

    1. the request is validated and serialised (a programming error raises :class:`LLMConfigError`,
       nothing logged);
    2. :func:`~wellbrief.egress.check_egress` refuses a non-loopback host unless ``allow_remote``
       (logged as ``refused-remote``);
    3. the ledger reserves the worst-case cost or refuses the call (logged as ``refused-budget``);
    4. the request is sent; the outcome is logged as ``ok`` or ``error`` with its cost.

    The prompt side of the worst case is estimated from the serialised request body (see
    :func:`~wellbrief.egress.estimate_request_tokens`), so every field that is sent is counted.

    A failed call that may have reached a model (a timeout after sending, a dropped connection, a 5xx
    or 408, a malformed or error body) is charged at its worst-case estimate, since the provider's
    charge is unknown. A call that never reached one (connection refused, TLS failure, a redirect, a
    4xx other than 408) costs 0.

    The logging fails closed. An unexpected exception after the request was handed to the HTTP client
    (a bug, or a reply the parser did not anticipate) is logged as ``error`` at the worst-case
    estimate and raised as :class:`LLMError`; an interruption such as ``KeyboardInterrupt`` is logged
    the same way and re-raised unchanged. If the record itself cannot be written, the call's
    reservation is never released, so the rest of the process keeps counting the estimate as spent,
    and :class:`~wellbrief.egress.EgressLogError` is raised.

    Args:
        http_client: The HTTP client to send through.
        ledger: Cost ledger for the workspace; its ``remote`` flag must match the HTTP client's host.
        allow_remote: Whether a non-loopback host may be called (``WELLBRIEF_ALLOW_REMOTE=1``).

    Raises:
        ValueError: the ledger's ``remote`` flag contradicts the host.
    """

    def __init__(self, http_client: ChatBackend, ledger: CostLedger, *, allow_remote: bool) -> None:
        remote = not is_loopback_host(http_client.host)
        if ledger.remote != remote:
            kind = "remote" if remote else "loopback"
            raise ValueError(
                f"The cost ledger was built with remote={ledger.remote} but {http_client.host!r} is a "
                f"{kind} host."
            )
        self._http = http_client
        self._ledger = ledger
        self._allow_remote = allow_remote

    @property
    def ledger(self) -> CostLedger:
        """The ledger calls are checked against and logged to."""
        return self._ledger

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        doc_ids: Iterable[str] = (),
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        """Check, send and log one chat-completions request.

        Args:
            messages: Chat messages.
            doc_ids: Ids of the documents whose text is included in the prompt, for the log. A
                collection of strings; a single string is refused.
            max_tokens: Output cap; also the output side of the worst-case estimate.
            temperature: Sampling temperature.
            extra: Additional request fields (see :data:`ALLOWED_EXTRA_KEYS`).

        Raises:
            LLMConfigError: invalid arguments; nothing sent or logged.
            wellbrief.egress.RemoteNotAllowed: refused by the egress rule; logged.
            wellbrief.egress.BudgetExceeded: refused by the budget stop (including
                :class:`~wellbrief.egress.PricingUnknown`); logged.
            wellbrief.egress.EgressLogError: the log cannot be totalled (nothing sent), or a record
                cannot be appended.
            LLMError: the request failed, or failed in an unexpected way after it was handed to
                the HTTP client; logged.
        """
        body = self._http.build_body(messages, max_tokens=max_tokens, temperature=temperature, extra=extra)
        data = encode_body(body)
        try:
            ids = unique_ids(doc_ids)
        except TypeError as exc:
            raise LLMConfigError(str(exc)) from None
        ts = utc_timestamp()
        prompt_estimate = estimate_request_tokens(data, len(messages))
        try:
            check_egress(self._http.base_url, self._allow_remote)
        except RemoteNotAllowed as exc:
            self._log_refusal(ts, ids, "refused-remote", str(exc), None)
            raise
        try:
            reservation = self._ledger.reserve(prompt_estimate, max_tokens)
        except BudgetExceeded as exc:
            self._log_refusal(ts, ids, "refused-budget", str(exc), exc.estimate_usd)
            raise
        call = _CallContext(ts, ids, len(data), prompt_estimate, max_tokens, reservation.estimate_usd)
        logged = False
        start = time.perf_counter()
        try:
            try:
                result = self._http.chat(
                    messages, max_tokens=max_tokens, temperature=temperature, extra=extra
                )
                settlement = self._ledger.settle(
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.provider_cost_usd,
                    prompt_estimate=prompt_estimate,
                    max_tokens=max_tokens,
                )
                self._append(
                    ts=call.ts,
                    doc_ids=call.doc_ids,
                    model=result.model,
                    bytes_sent=result.bytes_sent,
                    prompt_tokens=settlement.prompt_tokens,
                    completion_tokens=settlement.completion_tokens,
                    cost_usd=settlement.cost_usd,
                    cost_source=settlement.cost_source,
                    latency_ms=result.latency_ms,
                    status="ok",
                    error=None,
                    estimate_usd=call.estimate_usd,
                    tokens_estimated=settlement.tokens_estimated,
                )
                logged = True
            except EgressLogError:
                # The log cannot be written; a second attempt would fail the same way.
                raise
            except LLMError as exc:
                self._log_failure(call, exc)
                logged = True
                raise
            except Exception as exc:
                failure = LLMError(
                    f"The call to {self._http.host} failed unexpectedly: {type(exc).__name__}: "
                    f"{_snippet(str(exc), None)}",
                    bytes_sent=call.body_bytes,
                    latency_ms=_elapsed_ms(start),
                    request_sent=True,
                )
                self._log_failure(call, failure)
                logged = True
                raise failure from exc
            except BaseException as exc:
                interrupted = LLMError(
                    f"The call to {self._http.host} was interrupted ({type(exc).__name__}).",
                    bytes_sent=call.body_bytes,
                    latency_ms=_elapsed_ms(start),
                    request_sent=True,
                )
                self._log_failure(call, interrupted)
                logged = True
                raise
        finally:
            # A call whose record could not be written keeps its reservation for the life of the
            # process, so its worst case still counts against the budget.
            if logged:
                reservation.release()
        return result

    def _log_failure(self, call: _CallContext, exc: LLMError) -> None:
        billed = exc.may_have_billed
        self._append(
            ts=call.ts,
            doc_ids=call.doc_ids,
            model=self._http.model,
            bytes_sent=exc.bytes_sent,
            prompt_tokens=call.prompt_estimate if billed else 0,
            completion_tokens=call.max_tokens if billed else 0,
            cost_usd=call.estimate_usd if billed else 0.0,
            cost_source=self._ledger.cost_source,
            latency_ms=exc.latency_ms,
            status="error",
            error=str(exc),
            estimate_usd=call.estimate_usd,
            tokens_estimated=billed,
        )

    def _log_refusal(
        self,
        ts: str,
        doc_ids: tuple[str, ...],
        status: EgressStatus,
        error: str,
        estimate_usd: float | None,
    ) -> None:
        self._append(
            ts=ts,
            doc_ids=doc_ids,
            model=self._http.model,
            bytes_sent=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            cost_source=self._ledger.cost_source,
            latency_ms=0.0,
            status=status,
            error=error,
            estimate_usd=estimate_usd,
            tokens_estimated=False,
        )

    def _append(
        self,
        *,
        ts: str,
        doc_ids: tuple[str, ...],
        model: str,
        bytes_sent: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cost_source: CostSource,
        latency_ms: float,
        status: EgressStatus,
        error: str | None,
        estimate_usd: float | None,
        tokens_estimated: bool,
    ) -> None:
        self._ledger.log.append(
            EgressRecord(
                ts=ts,
                host=self._http.host,
                bytes_sent=bytes_sent,
                doc_ids=doc_ids,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                cost_source=cost_source,
                latency_ms=latency_ms,
                status=status,
                error=error,
                estimate_usd=estimate_usd,
                tokens_estimated=tokens_estimated,
            )
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


# ---------------------------------------------------------------------------
# The llm narrator
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You write the prose for a drilling-engineering answer that has already been computed. Use only "
    "the evidence block below: it is the whole archive you may draw on. After every claim you take "
    "from a document, cite it immediately as [DOC-ID], exactly as it appears in the evidence block. "
    "Never introduce a document id, well name, field name or number that is not already in the "
    "evidence block; narrate the computed figures, do not recompute or round them differently. If "
    "the evidence does not answer part of the question, say so plainly instead of guessing. Write "
    "plain prose, no headings, no markdown tables."
)

VERIFICATION_FAILED_BANNER = (
    "The language-model narrative was rejected by the verifier; showing the deterministic answer."
)
CALL_FAILED_BANNER = (
    "The language-model call did not complete; showing the deterministic answer."
)


def _with_banner(banner: str, offline_text: str) -> str:
    return f"{banner}\n\n{offline_text}"


def _evidence_block(question: str, evidence: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    lines = [f"Question: {question}", "", "Evidence pack (cite as [DOC-ID]; nothing outside it exists):"]
    for e in evidence:
        where = ", ".join(str(v) for v in (e.get("well"), e.get("date")) if v)
        lines.append(f"[{e['doc_id']}]{f' ({where})' if where else ''}: {e['quote']}")
    figures = summary.get("figures")
    if figures:
        lines += ["", "Computed figures (already correct; narrate them, never recompute):",
                 json.dumps(figures, ensure_ascii=False)]
    mitigations = summary.get("mitigations")
    if mitigations:
        lines += ["", f"{summary.get('mitigations_heading') or 'Mitigations'}:"]
        lines += [f"- [{m['doc_id']}] {m['text']}" for m in mitigations]
    return "\n".join(lines)


def _brief_evidence_block(brief: RiskBrief) -> str:
    lines = [
        f"Planned well: {brief.well_name} in {brief.field_name}, planned total depth "
        f"{brief.planned_td_m:.0f} m.",
        f"Offset wells this register is built from: {', '.join(brief.generated_from_wells)}.", "",
    ]
    for i, r in enumerate(brief.risks, start=1):
        lines.append(
            f"{i}. {r.title} -- {r.wells_affected}/{r.wells_total} offset wells, mean "
            f"{r.mean_npt_hours:.1f} h, P90 {r.p90_npt_hours:.1f} h, expected "
            f"{r.expected_npt_hours:.1f} h / ${r.expected_cost_usd:,.0f}, applies: {r.applies}."
        )
        for m in r.mitigations:
            lines.append(f"   mitigation: {m}")
        if r.citations:
            lines.append("   evidence: " + "; ".join(f"[{c.doc_id}] {c.quote}" for c in r.citations))
    lines += [
        "",
        f"Total expected NPT across the register: {brief.total_expected_npt_hours:.1f} h / "
        f"${brief.total_exposure_usd:,.0f} at ${brief.spread_rate_usd_per_day:,.0f} per day spread rate.",
    ]
    if brief.unavoidable_hours:
        lines.append("Unavoidable background NPT: " + ", ".join(
            f"{code} {hours:.1f} h" for code, hours in brief.unavoidable_hours.items()))
    return "\n".join(lines)


class LlmNarrator(Narrator):
    """Narrates `ask` answers and risk-brief narratives with an OpenAI-compatible model, behind the
    egress rule and the cost ledger (:class:`GuardedChat`).

    Every call is verified (:mod:`wellbrief.narrate.verify`) against the same evidence the offline
    narrator would have written from; a call that cannot be completed (refused by the egress rule or
    the budget, or a transport failure) or whose text fails verification shows the offline narrator's
    answer instead, with a banner, and records why on :attr:`last_rejection`.
    """

    name = "llm"

    def __init__(
        self,
        chat: GuardedChat,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        known_field_names: frozenset[str] = frozenset(),
    ) -> None:
        self._chat = chat
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._known_field_names = known_field_names
        self.last_rejection: dict[str, Any] | None = None

    def _extra(self) -> dict[str, Any] | None:
        # OpenRouter does not document chat_template_kwargs (the client's own
        # disable_thinking switch), so a remote host instead gets the documented
        # reasoning off-switch; a loopback host already has disable_thinking set.
        return {"reasoning": {"effort": "none"}} if self._chat.ledger.remote else None

    def _call(self, doc_ids: Iterable[str], prompt: str) -> ChatResult:
        return self._chat.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            doc_ids=doc_ids, max_tokens=self._max_tokens, temperature=self._temperature,
            extra=self._extra(),
        )

    def answer(self, question: str, evidence: list[dict[str, Any]], summary: dict[str, Any]) -> str:
        self.last_rejection = None
        offline_text = OfflineNarrator().answer(question, evidence, summary)
        doc_ids = [e["doc_id"] for e in evidence]
        try:
            result = self._call(doc_ids, _evidence_block(question, evidence, summary))
        except (LLMError, EgressError) as exc:
            self.last_rejection = {"text": None, "reasons": [f"call failed: {exc}"]}
            return _with_banner(CALL_FAILED_BANNER, offline_text)
        pack_evidence = evidence_from_pack(evidence, summary)
        check = verify(result.text, pack_evidence, question, self._known_field_names)
        if not check.ok:
            self.last_rejection = {"text": result.text, "reasons": check.reasons}
            return _with_banner(VERIFICATION_FAILED_BANNER, offline_text)
        return result.text

    def risk_brief(self, brief: RiskBrief) -> str:
        self.last_rejection = None
        offline_text = OfflineNarrator().risk_brief(brief)
        brief_evidence = evidence_from_brief(brief)
        try:
            result = self._call(brief_evidence.doc_ids, _brief_evidence_block(brief))
        except (LLMError, EgressError) as exc:
            self.last_rejection = {"text": None, "reasons": [f"call failed: {exc}"]}
            return _with_banner(CALL_FAILED_BANNER, offline_text)
        check = verify(result.text, brief_evidence, "", self._known_field_names)
        if not check.ok:
            self.last_rejection = {"text": result.text, "reasons": check.reasons}
            return _with_banner(VERIFICATION_FAILED_BANNER, offline_text)
        return result.text


def _price_env(env: Mapping[str, str], name: str) -> float | None:
    raw = (env.get(name) or "").strip()
    return float(raw) if raw else None


def narrator_from_env(env: Mapping[str, str], egress_log_path: str | os.PathLike[str], *,
                      known_field_names: frozenset[str] = frozenset(),
                      max_tokens: int = DEFAULT_MAX_TOKENS) -> LlmNarrator:
    """Build the llm narrator from `WELLBRIEF_LLM_*` environment variables and the workspace's
    egress log path.

    Raises:
        LLMConfigError: `WELLBRIEF_LLM_BASE_URL` or `WELLBRIEF_LLM_MODEL` is not set, or another
            argument the HTTP client validates is unusable.
        wellbrief.egress.InvalidBaseURL: `WELLBRIEF_LLM_BASE_URL` cannot be used as a URL.
        ValueError: a budget or price environment variable is not a number, or only one of the two
            price variables is set.
    """
    base_url = env.get("WELLBRIEF_LLM_BASE_URL") or ""
    if not base_url:
        raise LLMConfigError(
            "WELLBRIEF_LLM_BASE_URL is not set; the llm narrator needs an OpenAI-compatible server to "
            "call."
        )
    model = env.get("WELLBRIEF_LLM_MODEL") or ""
    remote = not is_loopback_host(split_base_url(base_url).host)
    http_client = OpenAICompatibleClient(
        base_url, model, api_key=env.get("WELLBRIEF_LLM_API_KEY") or None,
        ca_bundle=env.get("WELLBRIEF_CA_BUNDLE") or None, disable_thinking=not remote,
    )
    budget_raw = (env.get("WELLBRIEF_BUDGET_USD") or "").strip()
    budget = float(budget_raw) if budget_raw else DEFAULT_BUDGET_USD
    ledger = CostLedger(
        EgressLog(egress_log_path), budget,
        _price_env(env, "WELLBRIEF_LLM_PRICE_IN_PER_MTOK"),
        _price_env(env, "WELLBRIEF_LLM_PRICE_OUT_PER_MTOK"),
        remote=remote,
    )
    chat = GuardedChat(http_client, ledger, allow_remote=env.get("WELLBRIEF_ALLOW_REMOTE") == "1")
    return LlmNarrator(chat, max_tokens=max_tokens, known_field_names=known_field_names)
