"""Process-level network guard.

wellbrief is built to run on machines without network access. The command-line
entry point installs this guard at startup so that any accidental outbound
connection or name lookup made through Python's socket layer fails loudly instead
of leaving the machine. The guard replaces four entry points of :mod:`socket`:

- ``socket.socket.connect``
- ``socket.socket.connect_ex``
- ``socket.create_connection``
- ``socket.getaddrinfo``

Allowed without configuration:

- ``AF_UNIX`` sockets;
- loopback targets: ``127.0.0.0/8``, ``::1``, IPv4-mapped loopback such as
  ``::ffff:127.0.0.1``, and the name ``localhost`` (RFC 6761 reserves it for the
  loopback interface), matched without regard to case but only without a trailing
  dot: glibc answers ``localhost`` from ``/etc/hosts`` but sends ``localhost.`` to
  the configured DNS server, so that spelling is always blocked;
- the unspecified addresses ``0.0.0.0``, ``::`` and the empty host, which the kernel
  treats as the local host when used as a connect destination;
- ``getaddrinfo`` for ``None``, ``""``, ``localhost`` and the literals above.

Hosts passed to :func:`install` through ``allow_hosts`` are allowed as well. An IP
literal is allowed directly. A host name is allowed for ``getaddrinfo`` and for
connections that name it, with or without one trailing dot; the addresses its
lookups return are remembered, so a later connect to one of them is allowed too.
The command-line entry point passes the configured LLM host here only when the
``llm`` narrator is enabled.

Everything else raises :class:`EgressBlocked` before the original function runs,
so no packet is sent and no name is resolved, and the attempt is counted
(:func:`stats`). ``connect_ex`` raises as well instead of returning an error
number, so a caller cannot silently ignore a blocked attempt.

Importing this module changes nothing. A caller who wants the guard calls
:func:`install` explicitly; :func:`uninstall` puts the original objects back.

The guard is a tripwire for accidental egress through the standard socket layer,
not a sandbox. It does not intercept direct use of ``_socket`` (including
``socket.SocketType``, which is ``_socket.socket``), ``sendto`` or ``sendmsg`` on an
unconnected datagram socket, ``sendto`` or ``sendmsg`` with ``MSG_FASTOPEN`` on a
Linux stream socket (which opens a TCP connection without ``connect``), the
``gethostbyname``/``gethostbyaddr``/``getnameinfo`` family (and therefore
``socket.getfqdn``), a subprocess, or a C extension with its own networking.
Running with the network disabled at the operating-system or container level
remains the stronger control.
"""

from __future__ import annotations

import contextlib
import errno
import ipaddress
import re
import socket
import threading
import types
import weakref
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NoReturn, TypedDict

__all__ = [
    "RECENT_LIMIT",
    "SELF_TEST_TARGET",
    "BlockKind",
    "BlockedAttempt",
    "EgressBlocked",
    "GuardStats",
    "SelfTestResult",
    "install",
    "is_installed",
    "reset_stats",
    "self_test",
    "stats",
    "uninstall",
]

BlockKind = Literal["connect", "connect_ex", "create_connection", "getaddrinfo"]
"""The guarded entry point through which a blocked attempt was made."""

SELF_TEST_TARGET: Final[tuple[str, int]] = ("192.0.2.1", 80)
"""Positive-control target of :func:`self_test`: TEST-NET-1 (RFC 5737), port 80."""

RECENT_LIMIT: Final = 20
"""Number of most recent blocked attempts kept for :func:`stats`."""

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_SockAddr = tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes]
_AddrInfo = tuple[socket.AddressFamily, socket.SocketKind, int, str, _SockAddr]
_ConnectFn = Callable[[socket.socket, Any], None]
_ConnectExFn = Callable[[socket.socket, Any], int]
_CreateConnectionFn = Callable[..., socket.socket]
_GetAddrInfoFn = Callable[..., list[_AddrInfo]]
_Decision = Literal["local", "ip", "name", "block"]

_AF_UNIX: Final[int | None] = getattr(socket, "AF_UNIX", None)
_INET_FAMILIES: Final = frozenset({socket.AF_INET, socket.AF_INET6})
_LOCAL_NAME: Final = "localhost"
_HOSTNAME_RE: Final = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*")
_TARGET_MAX_CHARS: Final = 200


class BlockedAttempt(TypedDict):
    """One blocked attempt as reported by :func:`stats`."""

    kind: BlockKind
    target: str


class GuardStats(TypedDict):
    """Snapshot returned by :func:`stats`.

    ``installed`` is true only while every patch is in place. ``blocked`` counts
    blocked attempts since the last :func:`reset_stats`; ``recent`` holds the last
    :data:`RECENT_LIMIT` of them, oldest first. ``allowed_hosts`` lists the
    configured extra hosts and ``resolved`` the addresses remembered for each
    allowed host name.
    """

    installed: bool
    blocked: int
    recent: list[BlockedAttempt]
    allowed_hosts: list[str]
    resolved: dict[str, list[str]]


class SelfTestResult(TypedDict):
    """Outcome of :func:`self_test`: probes attempted and probes blocked."""

    attempted: int
    blocked: int


class EgressBlocked(OSError):
    """Raised instead of an outbound connection or name lookup.

    It subclasses :class:`OSError` with ``errno`` set to ``EACCES``, so code that
    already handles network errors (for example ``urllib``, which wraps it in
    ``URLError``) reports it without special handling.

    Attributes:
        kind: The guarded entry point that was called.
        target: The destination, formatted as ``host:port``, ``[v6]:port`` or ``host``.
    """

    kind: BlockKind
    target: str

    def __init__(self, kind: BlockKind, target: str) -> None:
        self.kind = kind
        self.target = target
        super().__init__(
            errno.EACCES,
            f"netguard blocked {kind} to {target}: outbound network access is disabled in this "
            "process (allowed: loopback, Unix sockets and explicitly allowed hosts)",
        )

    def __reduce__(self) -> tuple[type[EgressBlocked], tuple[BlockKind, str]]:
        return (type(self), (self.kind, self.target))


@dataclass(frozen=True)
class _Patch:
    """One attribute the current install replaced with a wrapper."""

    owner: object
    name: str
    wrapper: Callable[..., Any]


@dataclass(frozen=True)
class _Replaced:
    """What a wrapper replaced: the value it delegates to, and whether the owner held it itself.

    ``owned`` is false for an attribute that was inherited (``socket.socket`` inherits
    ``connect`` and ``connect_ex`` from ``_socket.socket``); restoring it means deleting
    the wrapper rather than assigning the original.
    """

    original: Any
    owned: bool


@dataclass
class _State:
    installed: bool = False
    patches: list[_Patch] = field(default_factory=list)
    allowed_names: frozenset[str] = frozenset()
    allowed_ips: frozenset[_IPAddress] = frozenset()
    resolved: dict[str, set[_IPAddress]] = field(default_factory=dict)
    resolved_ips: frozenset[_IPAddress] = frozenset()
    blocked: int = 0
    recent: deque[BlockedAttempt] = field(default_factory=lambda: deque(maxlen=RECENT_LIMIT))


# Reentrant because self_test() holds it while its probe runs through the wrappers.
_lock = threading.RLock()
_state = _State()
# Every wrapper the guard has created and not yet garbage-collected, mapped to what it
# replaced. A wrapper can outlive its install: other code may hold a reference to it (an
# HTTP connection object) or put it back after uninstall() (a monkeypatch undo that
# recorded it as the value to restore). The map lets install() and uninstall()
# recognise such a wrapper and put back what it replaced.
_replaced: weakref.WeakKeyDictionary[Callable[..., Any], _Replaced] = weakref.WeakKeyDictionary()


# --- public API ----------------------------------------------------------------------


def install(*, allow_hosts: Iterable[str] = ()) -> None:
    """Install the guard, or update its allow list if it is already installed.

    Calling it again does not wrap the functions twice; it replaces the set of
    allowed hosts with ``allow_hosts``, forgets the addresses remembered for names
    that are no longer allowed, and puts back any patch that other code replaced
    in the meantime (the originals recorded by the first call are kept).

    A fresh install records whatever each attribute holds as the original, except
    that a wrapper left over from an earlier install (for example one that a
    ``monkeypatch`` undo put back after :func:`uninstall`) is first replaced by what
    it wrapped, so the guard never wraps or delegates to itself.

    Args:
        allow_hosts: Extra hosts that may be reached, as host names or IP literals
            (an IPv6 literal may be bracketed). Pass the host only, without a
            scheme, port or path.

    Raises:
        TypeError: ``allow_hosts`` is a single string or holds a non-string.
        ValueError: An entry is empty or is not a host name or IP literal.
            Validation happens before anything is patched.
    """
    names, ips = _parse_allow_hosts(allow_hosts)
    with _lock:
        _state.allowed_names = names
        _state.allowed_ips = ips
        if _state.installed:
            _state.resolved = {n: a for n, a in _state.resolved.items() if n in names}
            _refresh_resolved_ips()
            # Re-assert any patch that other code replaced since the first install.
            for patch in _state.patches:
                if getattr(patch.owner, patch.name) is not patch.wrapper:
                    setattr(patch.owner, patch.name, patch.wrapper)
            return
        _state.resolved = {}
        _refresh_resolved_ips()
        _remove_wrappers()
        # Read every original before patching anything, so a failure leaves nothing half-installed.
        found = [
            (owner, name, wrap, getattr(owner, name), name in vars(owner)) for owner, name, wrap in _TARGETS
        ]
        for owner, name, wrap, original, owned in found:
            wrapper = wrap(original)
            _replaced[wrapper] = _Replaced(original=original, owned=owned)
            _state.patches.append(_Patch(owner=owner, name=name, wrapper=wrapper))
            setattr(owner, name, wrapper)
        _state.installed = True


def uninstall() -> None:
    """Remove the guard and restore the original objects.

    Every one of the four attributes that still holds a wrapper of this module gets
    back exactly what the wrapper replaced. An attribute that was inherited before
    :func:`install` (``socket.socket`` inherits ``connect`` and ``connect_ex`` from
    ``_socket.socket``) is deleted again rather than re-assigned. An attribute that
    other code has replaced or deleted since :func:`install` is left as it is, so a
    ``monkeypatch`` undo that already restored the real function is not reverted.
    Wrappers of earlier installs that other code put back are removed too.

    The allow list and remembered addresses are cleared; the counters are kept until
    :func:`reset_stats`. Calling it while the guard is not installed only removes
    such left-over wrappers, if there are any.
    """
    with _lock:
        try:
            _remove_wrappers()
        finally:
            _state.patches.clear()
            _state.installed = False
            _state.allowed_names = frozenset()
            _state.allowed_ips = frozenset()
            _state.resolved = {}
            _refresh_resolved_ips()


def is_installed() -> bool:
    """Return true when the guard is installed and all four patches are still in place."""
    with _lock:
        return _installed_locked()


def stats() -> GuardStats:
    """Return a snapshot of the guard state and its blocked-attempt counters."""
    with _lock:
        allowed = sorted(_state.allowed_names) + sorted(str(ip) for ip in _state.allowed_ips)
        return {
            "installed": _installed_locked(),
            "blocked": _state.blocked,
            "recent": [{"kind": a["kind"], "target": a["target"]} for a in _state.recent],
            "allowed_hosts": allowed,
            "resolved": {
                name: sorted(str(ip) for ip in addresses)
                for name, addresses in sorted(_state.resolved.items())
            },
        }


def reset_stats() -> None:
    """Set the blocked counter to zero and clear the recent-attempt list."""
    with _lock:
        _state.blocked = 0
        _state.recent.clear()


def self_test() -> SelfTestResult:
    """Prove that the installed guard blocks and counts an outbound connection.

    As a positive control it calls ``socket.create_connection`` on
    :data:`SELF_TEST_TARGET` with a one-second timeout and checks that
    :class:`EgressBlocked` was raised for that target and that the blocked counter
    went up. The probe is counted like any other blocked attempt.

    The probe must never leave the machine, so it is not run at all unless the
    guard is installed with every patch in place and the target is not allowed by
    the current configuration. The lock is held for the duration of the probe so
    that another thread cannot uninstall the guard in the meantime.

    Returns:
        ``{"attempted": 1, "blocked": 1}`` when the guard works; ``blocked`` is 0
        when the probe was not blocked by the guard.

    Raises:
        RuntimeError: The guard is not installed, a patch was replaced by other
            code, or the probe target is allowed; the probe was not run.
    """
    host, port = SELF_TEST_TARGET
    with _lock:
        if not _installed_locked():
            raise RuntimeError(
                "netguard is not installed (or a patch was replaced): refusing to run the "
                f"self-test probe to {host}:{port}, because it must never leave the machine"
            )
        if _decide_host(host)[0] != "block":
            raise RuntimeError(
                f"netguard allows {host} in its current configuration: refusing to run the "
                "self-test probe, because it must never leave the machine"
            )
        before = _state.blocked
        blocked = False
        try:
            probe = socket.create_connection(SELF_TEST_TARGET, timeout=1)
        except EgressBlocked as exc:
            blocked = (
                exc.kind == "create_connection"
                and exc.target == _format_target(host, port)
                and _state.blocked > before
            )
        except OSError:
            blocked = False
        else:
            probe.close()
        return {"attempted": 1, "blocked": 1 if blocked else 0}


# --- wrappers --------------------------------------------------------------------------
#
# Each install creates fresh wrappers bound to the originals it found. A wrapper applies
# the policy while the guard is installed and otherwise calls its original directly, so
# a reference captured while the guard was active (for example by an HTTP connection
# object) behaves like the original after uninstall(). Binding the original per wrapper,
# rather than sharing one module-level slot, means that a wrapper created by one install
# never calls the originals of a later one, which would loop if other code had wrapped
# the earlier wrapper in between.


def _wrap_connect(original: _ConnectFn) -> _ConnectFn:
    def guarded_connect(self: socket.socket, address: Any, /) -> None:
        """Check the destination, then call the original ``socket.socket.connect``."""
        if _state.installed:
            _check_socket_address("connect", self, address)
        original(self, address)

    return guarded_connect


def _wrap_connect_ex(original: _ConnectExFn) -> _ConnectExFn:
    def guarded_connect_ex(self: socket.socket, address: Any, /) -> int:
        """Check the destination, then call the original ``socket.socket.connect_ex``.

        A blocked target raises :class:`EgressBlocked` instead of returning an error
        number.
        """
        if _state.installed:
            _check_socket_address("connect_ex", self, address)
        return original(self, address)

    return guarded_connect_ex


def _wrap_create_connection(original: _CreateConnectionFn) -> _CreateConnectionFn:
    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        """Check the destination, then call the original ``socket.create_connection``."""
        if _state.installed:
            # The original unpacks ``host, port = address``, so a two-item list is valid too.
            if isinstance(address, tuple | list) and len(address) == 2:
                host, port = address
                if _decide_host(host)[0] == "block":
                    _block("create_connection", _format_target(host, port))
            else:
                _block("create_connection", _display(address))
        return original(address, *args, **kwargs)

    return guarded_create_connection


def _wrap_getaddrinfo(original: _GetAddrInfoFn) -> _GetAddrInfoFn:
    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> list[_AddrInfo]:
        """Check the host, then call the original ``socket.getaddrinfo``.

        Lookups of allowed host names are passed through and their addresses are
        remembered, so a later connect to them is allowed.
        """
        if not _state.installed:
            return original(host, port, *args, **kwargs)
        decision, name = _decide_host(host)
        if decision == "block":
            _block("getaddrinfo", _format_target(host, port))
        result = original(host, port, *args, **kwargs)
        if decision == "name":
            _remember(name, result)
        return result

    return guarded_getaddrinfo


_TARGETS: Final[tuple[tuple[object, str, Callable[[Any], Callable[..., Any]]], ...]] = (
    (socket.socket, "connect", _wrap_connect),
    (socket.socket, "connect_ex", _wrap_connect_ex),
    (socket, "create_connection", _wrap_create_connection),
    (socket, "getaddrinfo", _wrap_getaddrinfo),
)
"""The patched attributes, as (owner, attribute name, wrapper factory)."""


def _remove_wrappers() -> None:
    """Put back what each wrapper of this module still in place replaced.

    Only an attribute that the owner holds itself and that is one of this module's
    wrappers is touched; anything else, including a value other code assigned after
    :func:`install`, is left alone. The caller holds the lock.
    """
    for owner, name, _wrap in _TARGETS:
        current = vars(owner).get(name)
        replaced = _replaced.get(current) if isinstance(current, types.FunctionType) else None
        if replaced is None:
            continue
        if replaced.owned:
            setattr(owner, name, replaced.original)
        else:
            delattr(owner, name)


# --- policy ----------------------------------------------------------------------------


def _check_socket_address(kind: BlockKind, sock: socket.socket, address: object) -> None:
    """Raise :class:`EgressBlocked` unless ``sock`` may connect to ``address``."""
    family = sock.family
    if _AF_UNIX is not None and family == _AF_UNIX:
        return
    if family in _INET_FAMILIES:
        if isinstance(address, tuple) and len(address) >= 2:
            if _decide_host(address[0])[0] != "block":
                return
            _block(kind, _format_target(address[0], address[1]))
        _block(kind, _display(address))
    _block(kind, f"{_family_label(family)} {_display(address)}")


def _decide_host(host: object) -> tuple[_Decision, str]:
    """Classify a host argument without resolving it.

    Returns the decision and, for host names, the normalised name. ``"local"``
    covers ``None``, the empty host, loopback and unspecified addresses and
    ``localhost``; ``"ip"`` an allowed or remembered IP literal; ``"name"`` an
    allowed host name; ``"block"`` everything else, including ``localhost.`` with a
    trailing dot (even when ``localhost`` is allowed), arguments of an unexpected
    type and IP spellings that :mod:`ipaddress` does not accept.
    """
    if host is None:
        return "local", ""
    if isinstance(host, bytes | bytearray):
        text = bytes(host).decode("ascii", errors="backslashreplace")
    elif isinstance(host, str):
        text = host
    else:
        return "block", ""
    if text == "":
        return "local", ""
    ip = _parse_ip(text)
    with _lock:
        if ip is not None:
            if ip.is_loopback or ip.is_unspecified:
                return "local", ""
            if ip in _state.allowed_ips or ip in _state.resolved_ips:
                return "ip", ""
            return "block", ""
        name, rooted = _normalize_name(text)
        if name == _LOCAL_NAME and rooted:
            # glibc answers "localhost" from /etc/hosts but sends "localhost." to the
            # configured DNS server, so the rooted spelling would leave the machine.
            return "block", name
        if name in _state.allowed_names:
            return "name", name
    if name == _LOCAL_NAME:
        return "local", name
    return "block", name


def _remember(name: str, infos: list[_AddrInfo]) -> None:
    """Record the addresses a lookup of the allowed host ``name`` returned."""
    found: set[_IPAddress] = set()
    for family, _type, _proto, _canonname, sockaddr in infos:
        if family in _INET_FAMILIES and isinstance(sockaddr[0], str):
            ip = _parse_ip(sockaddr[0])
            if ip is not None:
                found.add(ip)
    if not found:
        return
    with _lock:
        if name in _state.allowed_names:
            _state.resolved.setdefault(name, set()).update(found)
            _refresh_resolved_ips()


def _block(kind: BlockKind, target: str) -> NoReturn:
    """Count a blocked attempt and raise :class:`EgressBlocked` for it."""
    with _lock:
        _state.blocked += 1
        _state.recent.append({"kind": kind, "target": target})
    raise EgressBlocked(kind, target)


def _installed_locked() -> bool:
    return _state.installed and all(
        getattr(patch.owner, patch.name) is patch.wrapper for patch in _state.patches
    )


def _refresh_resolved_ips() -> None:
    _state.resolved_ips = frozenset(ip for addresses in _state.resolved.values() for ip in addresses)


# --- parsing and formatting ------------------------------------------------------------


def _parse_allow_hosts(hosts: Iterable[str]) -> tuple[frozenset[str], frozenset[_IPAddress]]:
    """Validate and normalise ``allow_hosts`` into host names and IP addresses."""
    if isinstance(hosts, str | bytes):
        raise TypeError("allow_hosts must be an iterable of host names, not a single string")
    names: set[str] = set()
    ips: set[_IPAddress] = set()
    for entry in hosts:
        if not isinstance(entry, str):
            raise TypeError(f"allow_hosts entries must be strings, got {type(entry).__name__}")
        text = entry.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            raise ValueError("allow_hosts entries must not be empty")
        ip = _parse_ip(text)
        if ip is not None:
            ips.add(ip)
            continue
        name, _rooted = _normalize_name(text)
        if not _HOSTNAME_RE.fullmatch(name):
            raise ValueError(
                f"allow_hosts entry {entry!r} is not a host name or IP literal; "
                "pass the host only, without scheme, port or path"
            )
        names.add(name)
    return frozenset(names), frozenset(ips)


def _parse_ip(text: str) -> _IPAddress | None:
    """Parse a canonical IP literal.

    An IPv6 zone id is dropped and an IPv4-mapped IPv6 address is reduced to its
    IPv4 address, so that policy checks compare plain addresses. Spellings the C
    resolver would also accept, such as ``127.1``, are not parsed here and are
    therefore treated as host names, which fails closed.
    """
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip.scope_id is not None:
            return ipaddress.IPv6Address(int(ip))
    return ip


def _normalize_name(text: str) -> tuple[str, bool]:
    """Return a host name as the resolver would receive it, lower-cased and without one trailing dot.

    A non-ASCII name is IDNA-encoded first, as :mod:`socket` does before it calls the
    C resolver; this also turns the full-width and ideographic full stops into ``.``.
    The flag tells whether a trailing dot was removed.
    """
    name = text
    if not name.isascii():
        with contextlib.suppress(UnicodeError):
            name = name.encode("idna").decode("ascii")
    name = name.lower()
    if name.endswith(".") and len(name) > 1:
        return name[:-1], True
    return name, False


def _format_target(host: object, port: object) -> str:
    text = _display(host)
    if ":" in text and not text.startswith("["):
        text = f"[{text}]"
    if port is None:
        return text
    return f"{text}:{_display(port)}"


def _display(value: object) -> str:
    if isinstance(value, bytes | bytearray):
        text = bytes(value).decode("ascii", errors="backslashreplace")
    elif isinstance(value, str):
        text = value
    else:
        text = repr(value)
    if not text.isprintable():
        text = repr(text)
    if len(text) > _TARGET_MAX_CHARS:
        text = text[: _TARGET_MAX_CHARS - 3] + "..."
    return text


def _family_label(family: int) -> str:
    try:
        return socket.AddressFamily(family).name
    except ValueError:
        return f"address family {family}"
