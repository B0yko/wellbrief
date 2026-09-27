"""Egress guard, egress log and cost ledger for outbound LLM requests.

wellbrief is offline by default. The ``llm`` narrator is the only component that sends data out of
the process, and every request it makes passes through this module first:

* :func:`check_egress` refuses a base URL whose host is not a loopback literal unless the caller
  allows remote hosts explicitly (``WELLBRIEF_ALLOW_REMOTE=1``, read by the caller). The decision is
  made on the URL text alone, with no DNS lookup: a lookup is itself outbound traffic, and its answer
  can change between the check and the connection.
* :class:`EgressLog` appends one JSON line per request, sent or refused, to the workspace's
  ``egress.jsonl``. The file is the ledger: the amount spent is always re-read from it, so the total
  is cumulative across processes and runs that share a workspace.
* :class:`CostLedger` estimates the worst-case cost of a call before it is sent and refuses the call
  when the recorded spend plus that estimate would exceed the budget. After the call it prices the
  request from the provider-reported cost when there is one, and from the configured price table
  otherwise.
"""

from __future__ import annotations

import ipaddress
import itertools
import json
import math
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

__all__ = [
    "ALLOW_REMOTE_ENV",
    "BUDGET_ENV",
    "BYTES_PER_TOKEN",
    "DEFAULT_BUDGET_USD",
    "DEFAULT_MAX_TOKENS",
    "MAX_TOKEN_COUNT",
    "MESSAGE_OVERHEAD_TOKENS",
    "PRICE_IN_ENV",
    "PRICE_OUT_ENV",
    "REQUEST_OVERHEAD_TOKENS",
    "BaseURL",
    "BudgetExceeded",
    "CostLedger",
    "CostSource",
    "EgressError",
    "EgressLog",
    "EgressLogError",
    "EgressRecord",
    "EgressStatus",
    "InvalidBaseURL",
    "NetworkMode",
    "PricingUnknown",
    "RemoteNotAllowed",
    "Reservation",
    "Settlement",
    "check_egress",
    "estimate_prompt_tokens",
    "estimate_request_tokens",
    "is_loopback_host",
    "network_mode",
    "resolve_network_mode",
    "split_base_url",
    "unique_ids",
    "utc_timestamp",
]

NetworkMode = Literal["offline", "local-llm", "remote-llm"]
"""Where narration can send data: nowhere, a loopback LLM server, or a remote LLM host."""

CostSource = Literal["provider", "price-table", "local-zero"]
"""How a logged cost was obtained: reported by the provider, tokens x configured price, or a loopback
server with no price configured (cost 0)."""

EgressStatus = Literal["ok", "error", "refused-budget", "refused-remote"]
"""Outcome of one logged request."""

ALLOW_REMOTE_ENV = "WELLBRIEF_ALLOW_REMOTE"
BUDGET_ENV = "WELLBRIEF_BUDGET_USD"
PRICE_IN_ENV = "WELLBRIEF_LLM_PRICE_IN_PER_MTOK"
PRICE_OUT_ENV = "WELLBRIEF_LLM_PRICE_OUT_PER_MTOK"

DEFAULT_BUDGET_USD = 10.0
DEFAULT_MAX_TOKENS = 800

BYTES_PER_TOKEN = 1
"""Prompt-size ratio used by :func:`estimate_prompt_tokens`: one token per UTF-8 byte.

Byte-level BPE tokenizers (the GPT-4 class, Qwen, DeepSeek and Llama 3 families) have every single
byte in their base vocabulary, and merges only ever reduce the count, so a text never encodes to more
tokens than it has bytes. English prose averages 3.5 to 4.5 characters per token and number-dense
report text (dates, depths, document ids) roughly 2 to 3, so this over-estimates real prompts by a
factor of 2 to 4. A ratio such as ``len / 3`` is closer on prose but is not an upper bound on
digit-heavy text, because Qwen-family tokenizers split every digit into its own token."""

MESSAGE_OVERHEAD_TOKENS = 8
"""Allowance per chat message for the role marker and the template's start and end tokens."""

REQUEST_OVERHEAD_TOKENS = 64
"""Allowance per request for text a chat template adds on its own (a default system preamble and the
generation prompt)."""

MAX_TOKEN_COUNT = 2**53 - 1
"""Largest token count the ledger accepts (the largest integer JSON numbers carry exactly).

Larger values cannot come from a real request or reply, and multiplying them by a price would overflow
a float, so they are rejected as invalid input."""

_LOCALHOST = "localhost"
_IPV4_LOOPBACK = ipaddress.IPv4Network("127.0.0.0/8")
_IPV6_LOOPBACK_LITERALS = frozenset({"::1", "[::1]"})
_MTOK = 1_000_000.0


class EgressError(Exception):
    """Base class for every refusal or failure raised by this module."""


class InvalidBaseURL(EgressError, ValueError):
    """The LLM base URL cannot be used as given."""


class RemoteNotAllowed(EgressError):
    """The LLM host is not a loopback literal and remote hosts were not allowed."""

    def __init__(self, host: str) -> None:
        self.host = host
        super().__init__(
            f"Refusing to send data to {host!r}: it is not localhost, an address in 127.0.0.0/8 or ::1. "
            f"Set {ALLOW_REMOTE_ENV}=1 to allow a remote LLM host."
        )


class BudgetExceeded(EgressError):
    """The call was refused before it was sent because the budget does not cover its worst case."""

    def __init__(
        self,
        message: str,
        *,
        spent_usd: float,
        estimate_usd: float | None,
        budget_usd: float,
    ) -> None:
        self.spent_usd = spent_usd
        self.estimate_usd = estimate_usd
        self.budget_usd = budget_usd
        super().__init__(message)


class PricingUnknown(BudgetExceeded):
    """A remote host has no configured prices, so the budget cannot be enforced before the call.

    It subclasses :class:`BudgetExceeded` so that one ``except`` clause handles both refusals; both are
    logged with status ``refused-budget``.
    """

    def __init__(self, *, spent_usd: float, budget_usd: float) -> None:
        super().__init__(
            "Refusing to call a remote LLM host: no token prices are configured, so the worst-case "
            f"cost of the call cannot be checked against {BUDGET_ENV} before it is sent. "
            f"Set both {PRICE_IN_ENV} and {PRICE_OUT_ENV} (USD per million tokens, from the provider's "
            "price list).",
            spent_usd=spent_usd,
            estimate_usd=None,
            budget_usd=budget_usd,
        )


class EgressLogError(EgressError):
    """``egress.jsonl`` cannot be read as a ledger, or a record cannot be appended to it.

    The budget check refuses to guess: a line it cannot parse could hide spend, so every LLM call is
    refused until the file is repaired.
    """


@dataclass(frozen=True)
class BaseURL:
    """A validated LLM base URL, parsed once so that the egress check and the HTTP client agree."""

    scheme: str
    host: str
    """Host name in lower case, with the brackets of an IPv6 literal removed."""
    port: int | None
    netloc: str
    """Network location exactly as the HTTP client will send it (``host[:port]``)."""
    path: str
    """Path prefix without a trailing slash, for example ``/v1``."""

    def endpoint(self, suffix: str) -> str:
        """Return the absolute URL of ``suffix`` (for example ``/chat/completions``) under this base."""
        return f"{self.scheme}://{self.netloc}{self.path}/{suffix.lstrip('/')}"


def split_base_url(base_url: str) -> BaseURL:
    """Parse and validate an LLM base URL such as ``http://127.0.0.1:8080/v1``.

    The URL must be ``http`` or ``https``, name a host, and carry no user information, query or
    fragment. Whitespace, control characters, backslashes and non-ASCII characters are rejected, because
    those are where URL parsers disagree, and the host this module checks must be the host that the
    HTTP client connects to.

    Raises:
        InvalidBaseURL: the URL cannot be used.
    """
    if not isinstance(base_url, str) or not base_url:
        raise InvalidBaseURL("The LLM base URL is empty.")
    for char in base_url:
        if char.isspace() or ord(char) < 0x20 or ord(char) >= 0x7F or char == "\\":
            raise InvalidBaseURL(
                f"The LLM base URL {base_url!r} contains whitespace, a control character, a backslash "
                "or a non-ASCII character."
            )
    if "?" in base_url or "#" in base_url:
        raise InvalidBaseURL(f"The LLM base URL {base_url!r} must not contain a query or a fragment.")
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError as exc:
        raise InvalidBaseURL(f"The LLM base URL {base_url!r} cannot be parsed: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise InvalidBaseURL(f"The LLM base URL {base_url!r} must start with http:// or https://.")
    if "@" in parts.netloc:
        raise InvalidBaseURL(
            f"The LLM base URL for host {parts.hostname!r} carries credentials; pass the key through "
            "WELLBRIEF_LLM_API_KEY instead."
        )
    host = parts.hostname
    if not host:
        raise InvalidBaseURL(f"The LLM base URL {base_url!r} does not name a host.")
    return BaseURL(scheme=scheme, host=host, port=port, netloc=parts.netloc, path=parts.path.rstrip("/"))


def is_loopback_host(host: str) -> bool:
    """Return True only when ``host`` is a loopback literal, without any DNS lookup.

    Accepted: ``localhost`` in any letter case, an IPv4 address in ``127.0.0.0/8`` written as a
    dotted quad, and the IPv6 loopback address written exactly ``::1`` or ``[::1]``. Everything else
    is False, including names that merely contain a loopback address (``127.0.0.1.nip.io``),
    subdomains of localhost, ``0.0.0.0``, other spellings of the IPv6 loopback address
    (``0:0:0:0:0:0:0:1``), IPv4-mapped IPv6 addresses, zone ids, and IPv4 shorthand forms such as
    ``127.1`` that some resolvers expand. Refusing an unusual spelling of a loopback address is the
    safe direction for this check.
    """
    if not isinstance(host, str) or not host:
        return False
    if host.lower() == _LOCALHOST or host in _IPV6_LOOPBACK_LITERALS:
        return True
    try:
        address = ipaddress.IPv4Address(host)
    except ValueError:
        return False
    return address in _IPV4_LOOPBACK


def network_mode(narrator: str, base_url: str | None) -> NetworkMode:
    """Classify where narration can send data, for ``status`` and the UI badge.

    ``offline`` when the narrator is ``offline``, or when it is ``llm`` with no base URL configured
    (nothing can be sent; the narrator reports the missing setting itself). ``local-llm`` when the
    base URL's host is a loopback literal, ``remote-llm`` otherwise. The mode describes the
    configuration; whether a remote call is permitted is decided by :func:`check_egress`.

    Raises:
        ValueError: ``narrator`` is not ``offline`` or ``llm``.
        InvalidBaseURL: the ``llm`` narrator is configured with an unusable base URL.
    """
    if narrator == "offline":
        return "offline"
    if narrator != "llm":
        raise ValueError(f"Unknown narrator {narrator!r}; expected 'offline' or 'llm'.")
    if not base_url:
        return "offline"
    return "local-llm" if is_loopback_host(split_base_url(base_url).host) else "remote-llm"


def resolve_network_mode(narrator: str, env: dict[str, str] | None = None) -> NetworkMode:
    """:func:`network_mode` for a caller that only has the narrator name and the process
    environment, such as ``status`` and the HTTP API: reads ``WELLBRIEF_LLM_BASE_URL`` from
    ``env`` (``os.environ`` by default) and falls back to ``"offline"`` for a narrator name or a
    base URL that :func:`network_mode` itself would raise on, since a caller that reached this
    point already validated its own configuration (or is not using it at all)."""
    try:
        return network_mode(narrator, (env if env is not None else os.environ).get("WELLBRIEF_LLM_BASE_URL"))
    except (ValueError, InvalidBaseURL):
        return "offline"


def check_egress(base_url: str, allow_remote: bool) -> BaseURL:
    """Allow a request to ``base_url`` or refuse it, without resolving the host.

    Returns:
        The parsed URL, for the caller to connect to.

    Raises:
        InvalidBaseURL: the URL cannot be used.
        RemoteNotAllowed: the host is not a loopback literal and ``allow_remote`` is False.
    """
    parsed = split_base_url(base_url)
    if not allow_remote and not is_loopback_host(parsed.host):
        raise RemoteNotAllowed(parsed.host)
    return parsed


def estimate_prompt_tokens(text: str) -> int:
    """Return a conservative token count for ``text``: one token per UTF-8 byte.

    A lone surrogate (not valid Unicode, so it has no UTF-8 form) counts as its three-byte encoding.

    For ASCII text that is one token per character. See :data:`BYTES_PER_TOKEN` for why this is an
    upper bound. The estimate only feeds the pre-call budget check, where over-estimating is safe;
    recorded costs use the provider's reported cost or token counts. It is not meant for fitting a
    prompt into a model's context window.
    """
    return math.ceil(len(text.encode("utf-8", errors="surrogatepass")) / BYTES_PER_TOKEN)


def estimate_request_tokens(body: bytes, message_count: int) -> int:
    """Return a conservative prompt-token count for a serialised chat-completions request body.

    The estimate is one token per byte of the whole JSON body (see :data:`BYTES_PER_TOKEN`), plus
    :data:`MESSAGE_OVERHEAD_TOKENS` per message and :data:`REQUEST_OVERHEAD_TOKENS` once. Counting
    the body rather than only the message text covers every field the server may turn into prompt
    tokens (extra message keys, chat-template arguments), and JSON quoting and escaping only add
    bytes, so the count stays an upper bound for byte-level BPE tokenizers.

    Raises:
        ValueError: ``message_count`` is negative.
    """
    _check_tokens("message_count", message_count, 0)
    return (
        REQUEST_OVERHEAD_TOKENS
        + message_count * MESSAGE_OVERHEAD_TOKENS
        + math.ceil(len(body) / BYTES_PER_TOKEN)
    )


def utc_timestamp() -> str:
    """Return the current time as an ISO 8601 UTC string with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class EgressRecord:
    """One line of ``egress.jsonl``.

    ``prompt_tokens`` and ``completion_tokens`` are the counts the cost is based on. They are the
    provider's reported usage when it was reported; ``tokens_estimated`` is True when they are the
    pre-call worst case instead (no usage reported, or a failed call that may have been billed).
    ``estimate_usd`` is the worst-case estimate the budget check used, or None when no estimate was
    made. ``error`` is None unless ``status`` is not ``ok``.
    """

    ts: str
    host: str
    bytes_sent: int
    doc_ids: tuple[str, ...]
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_source: CostSource
    latency_ms: float
    status: EgressStatus
    error: str | None = None
    estimate_usd: float | None = None
    tokens_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict in a stable key order."""
        return {
            "ts": self.ts,
            "host": self.host,
            "bytes_sent": self.bytes_sent,
            "doc_ids": list(self.doc_ids),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "cost_source": self.cost_source,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
            "estimate_usd": self.estimate_usd,
            "tokens_estimated": self.tokens_estimated,
        }


_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _ledger_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _file_lock(key: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
        return lock


class EgressLog:
    """Append-only JSON Lines log of outbound LLM requests; the cost ledger's source of truth.

    Several instances may point at the same file (for example one per command in the same workspace);
    they all read the same total. Writes from threads of one process are serialised per file, and each
    record is written with a single ``write`` call on a file opened for appending, then flushed to
    disk.

    Records are written as pure ASCII JSON (``ensure_ascii``), so any string a server returns can be
    logged, including a lone UTF-16 surrogate from a JSON ``\\ud800`` escape, and a write cut short
    cannot leave half of a multi-byte character behind.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = _file_lock(_ledger_key(self._path))

    @property
    def path(self) -> Path:
        """Location of the log file."""
        return self._path

    def append(self, record: EgressRecord) -> None:
        """Append one record as a JSON line, creating the file and its directory when needed.

        Raises:
            EgressLogError: the file or its directory cannot be written.
        """
        line = json.dumps(record.to_dict(), ensure_ascii=True, allow_nan=False) + "\n"
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a+b") as handle:
                    handle.seek(0, os.SEEK_END)
                    if handle.tell() > 0:
                        handle.seek(-1, os.SEEK_END)
                        if handle.read(1) != b"\n":
                            # A previous writer stopped mid-line; start on a new line so this record
                            # stays parseable. The broken line is still reported by records().
                            line = "\n" + line
                    handle.write(line.encode("ascii"))
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise EgressLogError(f"{self._path}: the record cannot be appended: {exc}") from exc

    def records(self) -> list[dict[str, Any]]:
        """Return every record in file order; an absent file has no records.

        Raises:
            EgressLogError: the file cannot be read, or a non-empty line is not valid UTF-8 or not a
                JSON object.
        """
        out: list[dict[str, Any]] = []
        with self._lock:
            try:
                with self._path.open("rb") as handle:
                    for lineno, raw in enumerate(handle, start=1):
                        if raw.strip():
                            out.append(self._parse_line(lineno, raw))
            except FileNotFoundError:
                return []
            except OSError as exc:
                raise EgressLogError(f"{self._path}: the log cannot be read: {exc}") from exc
        return out

    def _parse_line(self, lineno: int, raw: bytes) -> dict[str, Any]:
        where = f"{self._path}: line {lineno}"
        try:
            value = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise EgressLogError(
                f"{where} is not valid UTF-8; the spend cannot be totalled until the line is repaired or "
                "removed."
            ) from exc
        except (ValueError, RecursionError) as exc:
            # ValueError covers JSONDecodeError and integers longer than Python's digit limit.
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else type(exc).__name__
            raise EgressLogError(
                f"{where} is not valid JSON ({detail}); the spend cannot be totalled until the line is "
                "repaired or removed."
            ) from exc
        if not isinstance(value, dict):
            raise EgressLogError(f"{where} is not a JSON object.")
        return value

    def spent_usd(self) -> float:
        """Return the sum of ``cost_usd`` over all records (0.0 for an absent file).

        Raises:
            EgressLogError: a line cannot be parsed or its ``cost_usd`` is not a finite number >= 0.
        """
        costs: list[float] = []
        for index, record in enumerate(self.records(), start=1):
            cost = _finite_amount(record.get("cost_usd"))
            if cost is None:
                shown = _short_repr(record.get("cost_usd"))
                raise EgressLogError(
                    f"{self._path}: record {index} has cost_usd={shown}; expected a finite number >= 0."
                )
            costs.append(cost)
        return math.fsum(costs)


def _short_repr(value: object, limit: int = 80) -> str:
    text = repr(value) if not isinstance(value, int) or value.bit_length() < 256 else "<a very long integer>"
    return text if len(text) <= limit else text[:limit] + "..."


def _finite_amount(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite int or float >= 0, else None.

    Integers too large for a float are None rather than an ``OverflowError``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _check_amount(name: str, value: float) -> float:
    number = _finite_amount(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number >= 0, got {_short_repr(value)}.")
    return number


def _check_price(name: str, value: float | None) -> float | None:
    return None if value is None else _check_amount(name, value)


def _check_tokens(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value > MAX_TOKEN_COUNT:
        raise ValueError(
            f"{name} must be an integer from {minimum} to {MAX_TOKEN_COUNT}, got {_short_repr(value)}."
        )


_PENDING: dict[str, dict[int, float]] = {}
_PENDING_LOCK = threading.Lock()
_RESERVATION_IDS = itertools.count(1)


class Reservation:
    """A worst-case estimate held against the budget while its call is in flight.

    Without it, two calls started together could each pass the budget check before either is logged.
    Release it after the call's record has been appended to the log; it is a context manager that
    does so on exit, and releasing twice is harmless.
    """

    def __init__(self, key: str, reservation_id: int, estimate_usd: float) -> None:
        self._key = key
        self._id = reservation_id
        self._estimate_usd = estimate_usd
        self._released = False

    @property
    def estimate_usd(self) -> float:
        """The worst-case cost this reservation holds."""
        return self._estimate_usd

    def release(self) -> None:
        """Stop holding the estimate against the budget."""
        with _PENDING_LOCK:
            if self._released:
                return
            self._released = True
            held = _PENDING.get(self._key)
            if held is not None:
                held.pop(self._id, None)
                if not held:
                    del _PENDING[self._key]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


@dataclass(frozen=True)
class Settlement:
    """The cost of a completed call and the token counts it is based on."""

    cost_usd: float
    cost_source: CostSource
    prompt_tokens: int
    completion_tokens: int
    tokens_estimated: bool


class CostLedger:
    """Budget stop and pricing for LLM calls, backed by an :class:`EgressLog`.

    Prices are USD per million tokens and must be set together or not at all. ``remote`` says whether
    the LLM host is remote (not a loopback literal); a remote host without prices cannot be called,
    because the budget could not be enforced. A loopback host without prices is priced at 0
    (``local-zero``). The budget is compared with the spend summed from the log, so it is cumulative
    per workspace.
    """

    def __init__(
        self,
        log: EgressLog,
        budget_usd: float,
        price_in_per_mtok: float | None,
        price_out_per_mtok: float | None,
        remote: bool,
    ) -> None:
        budget = _check_amount(BUDGET_ENV, budget_usd)
        price_in = _check_price(PRICE_IN_ENV, price_in_per_mtok)
        price_out = _check_price(PRICE_OUT_ENV, price_out_per_mtok)
        if (price_in is None) != (price_out is None):
            raise ValueError(f"Set both {PRICE_IN_ENV} and {PRICE_OUT_ENV}, or neither.")
        self._log = log
        self._key = _ledger_key(log.path)
        self._budget_usd = budget
        self._price_in = price_in
        self._price_out = price_out
        self._remote = bool(remote)

    @property
    def log(self) -> EgressLog:
        """The log the spend is read from and records are written to."""
        return self._log

    @property
    def budget_usd(self) -> float:
        """Maximum cumulative spend for the workspace, in USD."""
        return self._budget_usd

    @property
    def remote(self) -> bool:
        """Whether the priced host is remote."""
        return self._remote

    @property
    def prices_configured(self) -> bool:
        """Whether input and output prices are set."""
        return self._price_in is not None and self._price_out is not None

    @property
    def cost_source(self) -> CostSource:
        """Cost source for records that carry no provider-reported cost."""
        return "price-table" if self._remote or self.prices_configured else "local-zero"

    def spent_usd(self) -> float:
        """Return the cumulative spend recorded in the log."""
        return self._log.spent_usd()

    def pending_usd(self) -> float:
        """Return the worst-case estimates currently reserved against this log by calls in flight."""
        with _PENDING_LOCK:
            return math.fsum(_PENDING.get(self._key, {}).values())

    def estimate_worst_case(self, prompt_tokens: int, max_tokens: int) -> float:
        """Return ``prompt_tokens`` x input price + ``max_tokens`` x output price, in USD.

        Raises:
            ValueError: a token count is negative or ``max_tokens`` is below 1.
            PricingUnknown: the host is remote and no prices are configured.
        """
        _check_tokens("prompt_tokens", prompt_tokens, 0)
        _check_tokens("max_tokens", max_tokens, 1)
        if self._price_in is None or self._price_out is None:
            if self._remote:
                raise PricingUnknown(spent_usd=self.spent_usd(), budget_usd=self._budget_usd)
            return 0.0
        return (prompt_tokens * self._price_in + max_tokens * self._price_out) / _MTOK

    def check_before_call(self, prompt_tokens: int, max_tokens: int) -> float:
        """Refuse a call whose worst case does not fit in the remaining budget.

        Estimates already reserved by calls in flight count as spent.

        Returns:
            The worst-case estimate in USD.

        Raises:
            BudgetExceeded: spent + reserved + estimate would exceed the budget.
            PricingUnknown: the host is remote and no prices are configured.
            EgressLogError: the log cannot be totalled.
        """
        with _PENDING_LOCK:
            return self._check_locked(prompt_tokens, max_tokens)

    def reserve(self, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Check the budget like :meth:`check_before_call` and hold the estimate until released."""
        with _PENDING_LOCK:
            estimate = self._check_locked(prompt_tokens, max_tokens)
            reservation_id = next(_RESERVATION_IDS)
            _PENDING.setdefault(self._key, {})[reservation_id] = estimate
        return Reservation(self._key, reservation_id, estimate)

    def _check_locked(self, prompt_tokens: int, max_tokens: int) -> float:
        estimate = self.estimate_worst_case(prompt_tokens, max_tokens)
        spent = self._log.spent_usd()
        pending = math.fsum(_PENDING.get(self._key, {}).values())
        if spent + pending + estimate > self._budget_usd:
            raise BudgetExceeded(
                f"LLM call refused before sending: spent ${spent:.6f} + in flight ${pending:.6f} + "
                f"worst case ${estimate:.6f} ({prompt_tokens} prompt tokens, {max_tokens} max tokens) "
                f"would exceed the budget of ${self._budget_usd:.2f} ({BUDGET_ENV}).",
                spent_usd=spent,
                estimate_usd=estimate,
                budget_usd=self._budget_usd,
            )
        return estimate

    def cost_of(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Return tokens x configured price in USD, or 0.0 for a loopback host without prices.

        Raises:
            PricingUnknown: the host is remote and no prices are configured.
        """
        _check_tokens("prompt_tokens", prompt_tokens, 0)
        _check_tokens("completion_tokens", completion_tokens, 0)
        if self._price_in is None or self._price_out is None:
            if self._remote:
                raise PricingUnknown(spent_usd=self.spent_usd(), budget_usd=self._budget_usd)
            return 0.0
        return (prompt_tokens * self._price_in + completion_tokens * self._price_out) / _MTOK

    def settle(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        provider_cost_usd: float | None,
        *,
        prompt_estimate: int,
        max_tokens: int,
    ) -> Settlement:
        """Price a completed call.

        The provider-reported cost wins when present. Otherwise the cost is tokens x configured price
        (``price-table``), or 0 for a loopback host without prices (``local-zero``). Token counts the
        server did not report are replaced by the pre-call worst case (``prompt_estimate`` and
        ``max_tokens``), and the settlement says so.
        """
        estimated = prompt_tokens is None or completion_tokens is None
        prompt = prompt_estimate if prompt_tokens is None else prompt_tokens
        completion = max_tokens if completion_tokens is None else completion_tokens
        if provider_cost_usd is not None:
            cost = _check_amount("provider_cost_usd", provider_cost_usd)
            return Settlement(cost, "provider", prompt, completion, estimated)
        return Settlement(self.cost_of(prompt, completion), self.cost_source, prompt, completion, estimated)


def unique_ids(doc_ids: Iterable[str]) -> tuple[str, ...]:
    """Return ``doc_ids`` without duplicates, in first-seen order.

    Raises:
        TypeError: ``doc_ids`` is a single string (which would otherwise be split into characters) or
            contains something other than strings.
    """
    if isinstance(doc_ids, (str, bytes)):
        raise TypeError("doc_ids must be a collection of document ids, not a single string.")
    ids: list[str] = []
    for doc_id in doc_ids:
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_ids must contain only strings, got {_short_repr(doc_id)}.")
        ids.append(doc_id)
    return tuple(dict.fromkeys(ids))
