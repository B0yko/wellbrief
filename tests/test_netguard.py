"""Tests for the process-level network guard.

No test reaches the real network. Destinations outside the machine are either
blocked by the guard before the original socket function runs, or handled by a fake
transport installed underneath the guard (the guard records whatever is in place at
install time as the original and delegates to it).
"""

from __future__ import annotations

import contextlib
import errno
import http.server
import os
import pickle
import re
import socket
import socketserver
import subprocess
import sys
import tempfile
import textwrap
import threading
import traceback
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from wellbrief import netguard
from wellbrief.netguard import EgressBlocked

TEST_NET = "192.0.2.1"  # RFC 5737 TEST-NET-1
TEST_NET_OTHER = "192.0.2.2"  # a second TEST-NET-1 address
DOC_V6 = "2001:db8::1"  # RFC 3849 documentation prefix


# --- fixtures and helpers --------------------------------------------------------------


@pytest.fixture(autouse=True)
def pristine_guard() -> Iterator[None]:
    """Start and end every test with the guard removed and the counters at zero.

    pytest undoes ``monkeypatch`` before this teardown calls ``uninstall()``, as it
    would with an autouse fixture in the repository's ``conftest.py``. The final
    check proves that no test leaves a wrapper or a fake in the socket module.
    """
    netguard.uninstall()
    netguard.reset_stats()
    yield
    netguard.uninstall()
    netguard.reset_stats()
    assert _same(_PRISTINE, _snapshot()), "a socket attribute was left patched"
    assert _owned() == (False, False)


class FakeTransport:
    """Stands in for the resolver and ``connect`` underneath the guard."""

    def __init__(self, resolutions: dict[str, list[str]]) -> None:
        self.resolutions = resolutions
        self.lookups: list[object] = []
        self.connects: list[object] = []

    def getaddrinfo(
        self, host: Any, port: Any, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
    ) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[Any, ...]]]:
        self.lookups.append(host)
        key = host.decode("ascii") if isinstance(host, bytes) else host
        addresses = self.resolutions.get(key, [key or "127.0.0.1"]) if isinstance(key, str) else []
        number = port if isinstance(port, int) else 0
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, number, 0, 0))
            if ":" in address
            else (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, number))
            for address in addresses
        ]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """Replace name resolution and ``connect``/``connect_ex`` with a recording fake."""
    transport = FakeTransport({"llm.example": [TEST_NET], "empty.example": []})

    def fake_connect(sock: socket.socket, address: Any) -> None:
        transport.connects.append(address)

    def fake_connect_ex(sock: socket.socket, address: Any) -> int:
        transport.connects.append(address)
        return 0

    monkeypatch.setattr(socket, "getaddrinfo", transport.getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", fake_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", fake_connect_ex)
    return transport


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data = self.request.recv(1024)
        self.request.sendall(data.upper())


@pytest.fixture
def echo_port() -> Iterator[int]:
    """A TCP echo server on 127.0.0.1 with an ephemeral port."""
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _round_trip(sock: socket.socket) -> bytes:
    with sock:
        sock.settimeout(5)
        sock.sendall(b"ping")
        return sock.recv(1024)


def _snapshot() -> tuple[object, ...]:
    return (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)


def _owned() -> tuple[bool, bool]:
    return ("connect" in vars(socket.socket), "connect_ex" in vars(socket.socket))


def _same(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return all(a is b for a, b in zip(left, right, strict=True))


_PRISTINE = _snapshot()


def _entered_socket_module(caught: pytest.ExceptionInfo[EgressBlocked]) -> bool:
    """True when the traceback passes through ``socket.py`` (the original functions)."""
    socket_py = Path(socket.__file__).resolve()
    return any(Path(frame.filename).resolve() == socket_py for frame in traceback.extract_tb(caught.tb))


def _inet6_socket() -> socket.socket:
    try:
        return socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        pytest.skip("IPv6 sockets are not available")


def _tripwire(*args: object, **kwargs: object) -> socket.socket:
    raise AssertionError("the self-test probe must not run")


def _noop_connect(sock: socket.socket, address: Any) -> None:
    pass


def _replay_undo_after_uninstall() -> None:
    """Patch over the installed guard, uninstall it, then undo the patch.

    The undo puts back what the patch recorded as the old values: the guard's own
    wrappers, although the guard is no longer installed. This is what happens when
    a test's ``monkeypatch`` is undone after a fixture has called ``uninstall()``.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", _noop_connect)
        mp.setattr(socket, "getaddrinfo", FakeTransport({}).getaddrinfo)
        netguard.uninstall()


# --- import and lifecycle --------------------------------------------------------------


def test_import_installs_nothing() -> None:
    src = Path(netguard.__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import socket

        def snapshot():
            return (
                socket.socket.connect, socket.socket.connect_ex, socket.create_connection,
                socket.getaddrinfo, "connect" in vars(socket.socket), "connect_ex" in vars(socket.socket),
            )

        before = snapshot()
        import wellbrief
        import wellbrief.netguard as guard
        after = snapshot()
        assert all(a is b for a, b in zip(before, after)), "importing changed the socket module"
        assert not guard.is_installed()
        assert guard.stats()["installed"] is False
        print("unchanged")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unchanged"


def test_uninstall_restores_the_original_objects() -> None:
    before, owned_before = _snapshot(), _owned()
    assert owned_before == (False, False)  # socket.socket inherits connect/connect_ex from _socket
    netguard.install()
    assert not any(a is b for a, b in zip(before, _snapshot(), strict=True))
    assert netguard.is_installed()
    netguard.uninstall()
    assert _same(before, _snapshot())
    assert _owned() == owned_before
    assert not netguard.is_installed()
    netguard.uninstall()  # a second call does nothing
    assert _same(before, _snapshot())


def test_install_twice_is_idempotent() -> None:
    before = _snapshot()
    netguard.install(allow_hosts=["llm.example"])
    first = _snapshot()
    netguard.install(allow_hosts=["other.example"])
    assert _same(first, _snapshot())
    assert netguard.stats()["allowed_hosts"] == ["other.example"]
    netguard.uninstall()
    assert _same(before, _snapshot())


def test_install_again_repairs_a_replaced_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    netguard.install()
    wrapper = socket.create_connection
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    assert not netguard.is_installed()
    assert netguard.stats()["installed"] is False
    netguard.install()
    assert socket.create_connection is wrapper
    assert netguard.is_installed()


def test_uninstall_clears_the_allow_list(fake: FakeTransport) -> None:
    netguard.install(allow_hosts=["llm.example"])
    socket.getaddrinfo("llm.example", 443)
    assert netguard.stats()["resolved"] == {"llm.example": [TEST_NET]}
    netguard.uninstall()
    snapshot = netguard.stats()
    assert snapshot["allowed_hosts"] == []
    assert snapshot["resolved"] == {}
    assert snapshot["installed"] is False


def test_captured_wrapper_passes_through_after_uninstall(fake: FakeTransport) -> None:
    netguard.install()
    captured = socket.getaddrinfo
    netguard.uninstall()
    captured("example.com", 443)  # behaves like the original, which here is the fake
    assert fake.lookups == ["example.com"]
    assert netguard.stats()["blocked"] == 0


def test_uninstall_keeps_originals_that_a_monkeypatch_undo_restored() -> None:
    real_getaddrinfo = socket.getaddrinfo
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", FakeTransport({}).getaddrinfo)
        mp.setattr(socket.socket, "connect", _noop_connect)
        netguard.install()  # records the fakes as originals, as the CLI entry point would in a test
        assert netguard.is_installed()
    # The undo ran first and put the real objects back over the guard's wrappers.
    assert socket.getaddrinfo is real_getaddrinfo
    assert "connect" not in vars(socket.socket)
    netguard.uninstall()  # must not put the fakes back
    assert _same(_PRISTINE, _snapshot())
    assert _owned() == (False, False)
    assert not netguard.is_installed()


def test_uninstall_removes_wrappers_put_back_while_not_installed() -> None:
    netguard.install()
    _replay_undo_after_uninstall()
    assert "connect" in vars(socket.socket)  # a wrapper of the finished install
    assert not netguard.is_installed()
    netguard.uninstall()
    assert _same(_PRISTINE, _snapshot())
    assert _owned() == (False, False)


def test_reinstall_over_wrappers_put_back_after_uninstall(echo_port: int) -> None:
    netguard.install()
    _replay_undo_after_uninstall()
    netguard.install()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(("127.0.0.1", echo_port))  # the wrapper must not delegate to itself
    assert _round_trip(sock) == b"PING"
    with pytest.raises(EgressBlocked):
        socket.getaddrinfo("example.com", 443)
    assert netguard.stats()["blocked"] == 1
    netguard.uninstall()
    assert _same(_PRISTINE, _snapshot())
    assert _owned() == (False, False)


def test_reinstall_under_a_foreign_wrapper_of_the_guard(fake: FakeTransport) -> None:
    netguard.install()
    inner = socket.getaddrinfo

    def outer(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        """Stands in for a library that wraps whatever ``socket.getaddrinfo`` it finds."""
        return inner(host, port, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", outer)
        netguard.uninstall()
        assert socket.getaddrinfo is outer  # not the guard's, so uninstall() leaves it
        netguard.install()  # wraps outer, which still calls the first install's wrapper
        socket.getaddrinfo("127.0.0.1", 80)
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("example.com", 443)
        assert netguard.stats()["blocked"] == 1
        netguard.uninstall()
        assert socket.getaddrinfo is outer
    assert fake.lookups == ["127.0.0.1"]


def test_uninstall_tolerates_a_wrapper_removed_by_other_code() -> None:
    netguard.install()
    delattr(socket.socket, "connect_ex")  # other code restoring the inherited method its own way
    netguard.uninstall()
    assert not netguard.is_installed()
    assert _same(_PRISTINE, _snapshot())
    assert _owned() == (False, False)
    netguard.install()
    assert netguard.is_installed()


# --- blocked egress --------------------------------------------------------------------


def test_blocks_tcp_connect_to_test_net_before_the_original_runs() -> None:
    netguard.install()
    with pytest.raises(EgressBlocked) as caught:
        socket.create_connection((TEST_NET, 80), timeout=1)
    assert caught.value.kind == "create_connection"
    assert caught.value.target == "192.0.2.1:80"
    assert caught.value.errno == errno.EACCES
    assert "192.0.2.1:80" in str(caught.value)
    assert not _entered_socket_module(caught)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(EgressBlocked) as caught_connect,
    ):
        sock.connect((TEST_NET, 80))
    assert caught_connect.value.kind == "connect"
    snapshot = netguard.stats()
    assert snapshot["blocked"] == 2
    assert snapshot["recent"] == [
        {"kind": "create_connection", "target": "192.0.2.1:80"},
        {"kind": "connect", "target": "192.0.2.1:80"},
    ]


def test_connect_ex_raises_instead_of_returning_an_errno() -> None:
    netguard.install()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(EgressBlocked) as caught:
        sock.connect_ex((TEST_NET, 80))
    assert caught.value.kind == "connect_ex"
    assert netguard.stats()["recent"] == [{"kind": "connect_ex", "target": "192.0.2.1:80"}]


def test_blocks_dns_lookup_of_a_public_name_before_resolution() -> None:
    netguard.install()
    with pytest.raises(EgressBlocked) as caught:
        socket.getaddrinfo("example.com", 443)
    assert caught.value.kind == "getaddrinfo"
    assert caught.value.target == "example.com:443"
    assert not _entered_socket_module(caught)
    # A host name inside a connect address would be resolved by the C layer; it is refused first.
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(EgressBlocked) as caught_connect,
    ):
        sock.connect(("example.com", 80))
    assert caught_connect.value.target == "example.com:80"
    with pytest.raises(EgressBlocked) as caught_create:
        socket.create_connection(("example.com", 443), timeout=1)
    assert caught_create.value.kind == "create_connection"
    assert not _entered_socket_module(caught_create)
    assert netguard.stats()["blocked"] == 3


@pytest.fixture
def direct_urllib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make urllib connect directly: no proxy from the environment or system settings."""
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    # A non-empty environment result also stops urllib from reading system proxy settings.
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr("urllib.request._opener", None)  # drop an opener cached by earlier calls


def test_urlopen_to_an_external_host_is_blocked_and_counted(direct_urllib: None) -> None:
    netguard.install()
    with pytest.raises((EgressBlocked, urllib.error.URLError)) as caught:
        urllib.request.urlopen("http://example.invalid/", timeout=1)
    error = caught.value
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    assert isinstance(reason, EgressBlocked)
    assert reason.kind == "create_connection"
    snapshot = netguard.stats()
    assert snapshot["blocked"] == 1
    assert snapshot["recent"] == [{"kind": "create_connection", "target": "example.invalid:80"}]


# --- allowed traffic -------------------------------------------------------------------


def test_loopback_round_trip_is_allowed(echo_port: int) -> None:
    netguard.install()
    assert _round_trip(socket.create_connection(("127.0.0.1", echo_port), timeout=5)) == b"PING"
    assert _round_trip(socket.create_connection(("localhost", echo_port), timeout=5)) == b"PING"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(("127.0.0.1", echo_port))
    assert _round_trip(sock) == b"PING"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    assert sock.connect_ex(("127.0.0.1", echo_port)) == 0
    assert _round_trip(sock) == b"PING"
    assert netguard.stats()["blocked"] == 0


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_loopback_http_server_and_urlopen_work_under_the_guard(direct_urllib: None) -> None:
    netguard.install()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)  # started while guarded
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        with urllib.request.urlopen(url, timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert netguard.stats()["blocked"] == 0


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="AF_UNIX is not available")
def test_unix_sockets_are_allowed() -> None:
    netguard.install()
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"ping")
        assert right.recv(16) == b"ping"
    # sun_path is limited to about 104 bytes on macOS, so keep the directory name short.
    with tempfile.TemporaryDirectory(prefix="wbng-") as tmp:
        path = str(Path(tmp) / "s.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(path)
            server.listen(2)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(path)
                conn, _ = server.accept()
                with conn:
                    sock.sendall(b"ping")
                    assert conn.recv(16) == b"ping"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                assert sock.connect_ex(path) == 0
                conn, _ = server.accept()
                conn.close()
    assert netguard.stats()["blocked"] == 0


def test_allowed_host_name_lookup_is_remembered_and_connectable(echo_port: int) -> None:
    netguard.install(allow_hosts=["localhost"])
    infos = socket.getaddrinfo("localhost", echo_port, type=socket.SOCK_STREAM)
    snapshot = netguard.stats()
    assert snapshot["allowed_hosts"] == ["localhost"]
    remembered = snapshot["resolved"]["localhost"]
    ipv4 = [info[4] for info in infos if info[0] == socket.AF_INET]
    assert ipv4, "localhost did not resolve to an IPv4 address"
    for sockaddr in ipv4:
        assert sockaddr[0] in remembered
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(sockaddr)
        assert _round_trip(sock) == b"PING"
    assert netguard.stats()["blocked"] == 0


def test_allowed_host_name_resolving_to_a_remote_address(fake: FakeTransport) -> None:
    netguard.install(allow_hosts=["LLM.Example."])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(EgressBlocked):
            sock.connect((TEST_NET, 443))  # not looked up yet
        infos = socket.getaddrinfo("llm.example", 443)
        assert [info[4][0] for info in infos] == [TEST_NET]
        assert netguard.stats()["resolved"] == {"llm.example": [TEST_NET]}
        sock.connect((TEST_NET, 443))
        sock.connect(("llm.example", 443))
        with pytest.raises(EgressBlocked):
            sock.connect((TEST_NET_OTHER, 443))
    with pytest.raises(EgressBlocked):
        socket.getaddrinfo("other.example", 443)
    socket.getaddrinfo("LLM.example.", 443)  # an allowed name may carry one trailing dot
    socket.create_connection(("llm.example", 443), timeout=1).close()
    assert fake.lookups == ["llm.example", "LLM.example.", "llm.example"]
    assert fake.connects == [(TEST_NET, 443), ("llm.example", 443), (TEST_NET, 443)]
    assert netguard.stats()["blocked"] == 3


def test_rooted_localhost_is_blocked_even_when_localhost_is_allowed(fake: FakeTransport) -> None:
    netguard.install(allow_hosts=["localhost."])
    assert netguard.stats()["allowed_hosts"] == ["localhost"]
    socket.getaddrinfo("localhost", 80)
    # glibc sends "localhost." to the configured DNS server instead of reading /etc/hosts.
    hosts: list[Any] = ["localhost.", "LocalHost.", b"localhost.", "localhost\uff0e"]
    for host in hosts:
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo(host, 80)
        with pytest.raises(EgressBlocked) as caught:
            socket.create_connection((host, 80), timeout=1)
        assert not _entered_socket_module(caught)
    assert fake.lookups == ["localhost"]
    assert fake.connects == []
    assert netguard.stats()["blocked"] == 8


def test_reinstall_forgets_addresses_of_names_no_longer_allowed(fake: FakeTransport) -> None:
    netguard.install(allow_hosts=["llm.example", "empty.example"])
    socket.getaddrinfo("llm.example", 443)
    assert socket.getaddrinfo("empty.example", 443) == []
    assert netguard.stats()["resolved"] == {"llm.example": [TEST_NET]}
    netguard.install(allow_hosts=["llm.example"])
    assert netguard.stats()["resolved"] == {"llm.example": [TEST_NET]}
    netguard.install(allow_hosts=["other.example"])
    assert netguard.stats()["resolved"] == {}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(EgressBlocked):
        sock.connect((TEST_NET, 443))


def test_allowed_ip_literals_are_allowed_directly(fake: FakeTransport) -> None:
    netguard.install(allow_hosts=[TEST_NET, f"[{DOC_V6}]"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((TEST_NET, 8080))
    socket.create_connection((TEST_NET, 8080), timeout=1).close()
    with _inet6_socket() as sock6:
        sock6.connect((DOC_V6, 8080, 0, 0))
        sock6.connect((f"::ffff:{TEST_NET}", 8080, 0, 0))
        with pytest.raises(EgressBlocked):
            sock6.connect((f"::ffff:{TEST_NET_OTHER}", 8080, 0, 0))
    assert fake.lookups == [TEST_NET]
    assert fake.connects == [
        (TEST_NET, 8080),
        (TEST_NET, 8080),
        (DOC_V6, 8080, 0, 0),
        (f"::ffff:{TEST_NET}", 8080, 0, 0),
    ]
    snapshot = netguard.stats()
    assert snapshot["allowed_hosts"] == [TEST_NET, DOC_V6]
    assert snapshot["blocked"] == 1


# --- policy tables (fake transport underneath) -----------------------------------------

LOCAL_INET: list[tuple[object, int]] = [
    ("127.0.0.1", 9),
    ("127.0.0.2", 9),  # all of 127.0.0.0/8 is loopback
    ("localhost", 9),
    ("LocalHost", 9),
    ("0.0.0.0", 9),
    ("", 9),
    (b"127.0.0.1", 9),
    (b"localhost", 9),
]
BLOCKED_INET: list[tuple[object, int]] = [
    (TEST_NET, 9),
    (b"192.0.2.1", 9),
    ("127.1", 9),  # accepted by the C resolver, not by ipaddress: fails closed
    ("2130706433", 9),
    ("localhost.", 9),  # glibc sends the rooted name to DNS
    ("LOCALHOST.", 9),
    (b"localhost.", 9),
    ("localhost\u3002", 9),  # IDNA turns the ideographic full stop into a trailing dot
    ("localhost.localdomain", 9),
    ("foo.localhost", 9),
    ("<broadcast>", 9),
]
LOCAL_INET6: list[tuple[str, int, int, int]] = [
    ("::1", 9, 0, 0),
    ("::1%lo0", 9, 0, 0),
    ("::ffff:127.0.0.1", 9, 0, 0),
    ("::", 9, 0, 0),
    ("localhost", 9, 0, 0),
]
BLOCKED_INET6: list[tuple[str, int, int, int]] = [
    (DOC_V6, 9, 0, 0),
    (f"::ffff:{TEST_NET}", 9, 0, 0),
    (f"{DOC_V6}%lo0", 9, 0, 0),  # a zone id does not make an address local
    ("localhost.", 9, 0, 0),
]


@pytest.mark.parametrize("address", LOCAL_INET)
def test_local_ipv4_targets_pass_through(fake: FakeTransport, address: tuple[object, int]) -> None:
    netguard.install()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)
        assert sock.connect_ex(address) == 0
    assert fake.connects == [address, address]


@pytest.mark.parametrize("address", BLOCKED_INET)
def test_other_ipv4_targets_are_blocked(fake: FakeTransport, address: tuple[object, int]) -> None:
    netguard.install()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(EgressBlocked):
            sock.connect(address)
        with pytest.raises(EgressBlocked):
            sock.connect_ex(address)
    assert fake.connects == []
    assert netguard.stats()["blocked"] == 2


@pytest.mark.parametrize("address", LOCAL_INET6)
def test_local_ipv6_targets_pass_through(fake: FakeTransport, address: tuple[str, int, int, int]) -> None:
    netguard.install()
    with _inet6_socket() as sock:
        sock.connect(address)
    assert fake.connects == [address]


@pytest.mark.parametrize("address", BLOCKED_INET6)
def test_other_ipv6_targets_are_blocked(fake: FakeTransport, address: tuple[str, int, int, int]) -> None:
    netguard.install()
    with _inet6_socket() as sock, pytest.raises(EgressBlocked) as caught:
        sock.connect(address)
    host = address[0]
    assert caught.value.target == (f"[{host}]:9" if ":" in host else f"{host}:9")
    assert fake.connects == []


@pytest.mark.parametrize(
    "host",
    [None, "", "localhost", "LocalHost", b"localhost", "127.0.0.1", "::1", "::ffff:127.0.0.1", "0.0.0.0"],
)
def test_local_lookups_pass_through(fake: FakeTransport, host: Any) -> None:
    netguard.install()
    socket.getaddrinfo(host, 80)
    assert fake.lookups == [host]
    assert netguard.stats()["resolved"] == {}


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        TEST_NET,
        DOC_V6,
        "localhost.",
        "LocalHost.",
        b"localhost.",
        "localhost\uff0e",
        "localhost.localdomain",
        "foo.localhost",
        "127.1",
        12345,
    ],
)
def test_other_lookups_are_blocked_without_resolution(fake: FakeTransport, host: Any) -> None:
    netguard.install()
    with pytest.raises(EgressBlocked) as caught:
        socket.getaddrinfo(host, None)
    assert caught.value.kind == "getaddrinfo"
    assert fake.lookups == []


def test_malformed_and_foreign_addresses_fail_closed(fake: FakeTransport) -> None:
    netguard.install()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(EgressBlocked):
            sock.connect(cast(Any, (TEST_NET,)))
        with pytest.raises(EgressBlocked):
            sock.connect(cast(Any, 12345))
    with pytest.raises(EgressBlocked):
        socket.create_connection(cast(Any, "192.0.2.1:80"))
    with pytest.raises(EgressBlocked) as caught:
        socket.create_connection((cast(Any, 12345), 80))
    assert caught.value.target == "12345:80"
    assert fake.connects == []
    assert fake.lookups == []


def test_create_connection_accepts_a_two_item_list(fake: FakeTransport) -> None:
    netguard.install()
    socket.create_connection(cast(Any, ["127.0.0.1", 9]), timeout=1).close()
    with pytest.raises(EgressBlocked) as caught:
        socket.create_connection(cast(Any, [TEST_NET, 9]), timeout=1)
    assert caught.value.target == "192.0.2.1:9"
    assert fake.lookups == ["127.0.0.1"]
    assert fake.connects == [("127.0.0.1", 9)]


class _ForeignFamilySocket(socket.socket):
    """An AF_INET socket that reports an address family the guard does not know."""

    @property
    def family(self) -> socket.AddressFamily:
        return cast(socket.AddressFamily, 9999)


def test_unknown_address_families_are_blocked(fake: FakeTransport) -> None:
    netguard.install()
    with (
        _ForeignFamilySocket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(EgressBlocked) as caught,
    ):
        sock.connect(("127.0.0.1", 9))
    assert caught.value.target == "address family 9999 ('127.0.0.1', 9)"
    assert fake.connects == []


def test_targets_are_printable_and_bounded() -> None:
    netguard.install()
    with pytest.raises(EgressBlocked) as caught:
        socket.getaddrinfo("bad\nhost", 80)
    assert caught.value.target == "'bad\\nhost':80"
    with pytest.raises(EgressBlocked) as caught_long:
        socket.getaddrinfo("a" * 500 + ".example", b"http")
    assert caught_long.value.target.endswith("...:http")
    assert len(caught_long.value.target) < 220


# --- configuration ---------------------------------------------------------------------


def test_allow_hosts_are_normalised() -> None:
    netguard.install(allow_hosts=[" LLM.Example. ", "[::1]", f"::ffff:{TEST_NET}", "bücher.example"])
    assert netguard.stats()["allowed_hosts"] == ["llm.example", "xn--bcher-kva.example", TEST_NET, "::1"]


@pytest.mark.parametrize(
    ("hosts", "error"),
    [
        ("llm.example", TypeError),
        ([b"llm.example"], TypeError),
        ([""], ValueError),
        (["[]"], ValueError),
        (["https://llm.example/v1"], ValueError),
        (["llm.example:443"], ValueError),
        (["two words"], ValueError),
        (["*"], ValueError),
        (["ü" + "a" * 70 + ".example"], ValueError),  # not encodable as IDNA
    ],
)
def test_install_rejects_bad_allow_hosts(hosts: Any, error: type[Exception]) -> None:
    before = _snapshot()
    with pytest.raises(error):
        netguard.install(allow_hosts=hosts)
    assert not netguard.is_installed()
    assert _same(before, _snapshot())


# --- counters, self-test and exception -------------------------------------------------


def test_reset_stats_clears_counters_but_keeps_the_guard() -> None:
    netguard.install()
    for _ in range(3):
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("example.com", 80)
    assert netguard.stats()["blocked"] == 3
    netguard.reset_stats()
    snapshot = netguard.stats()
    assert snapshot["blocked"] == 0
    assert snapshot["recent"] == []
    assert snapshot["installed"] is True


def test_counters_are_exact_under_concurrency(fake: FakeTransport) -> None:
    netguard.install()
    workers, attempts = 8, 50
    barrier = threading.Barrier(workers)

    def attempt_many() -> None:
        barrier.wait()
        for _ in range(attempts):
            with contextlib.suppress(EgressBlocked):
                socket.create_connection((TEST_NET, 80), timeout=1).close()

    threads = [threading.Thread(target=attempt_many) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    snapshot = netguard.stats()
    assert snapshot["blocked"] == workers * attempts
    assert len(snapshot["recent"]) == netguard.RECENT_LIMIT
    assert fake.lookups == []
    assert fake.connects == []


def test_self_test_blocks_and_counts_one_probe() -> None:
    netguard.install()
    result = netguard.self_test()
    assert result == {"attempted": 1, "blocked": 1}
    line = f"guard self-test: blocked {result['blocked']}/{result['attempted']}"
    assert line == "guard self-test: blocked 1/1"
    snapshot = netguard.stats()
    assert snapshot["blocked"] == 1
    assert snapshot["recent"] == [{"kind": "create_connection", "target": "192.0.2.1:80"}]


def test_self_test_refuses_to_probe_without_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    with pytest.raises(RuntimeError, match="not installed"):
        netguard.self_test()
    assert netguard.stats()["blocked"] == 0


def test_self_test_refuses_when_a_patch_was_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    netguard.install()
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    with pytest.raises(RuntimeError, match="not installed"):
        netguard.self_test()
    assert netguard.stats()["blocked"] == 0


def test_self_test_refuses_when_its_target_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    netguard.install(allow_hosts=[TEST_NET])
    with pytest.raises(RuntimeError, match=re.escape("allows 192.0.2.1")):
        netguard.self_test()
    assert netguard.stats()["blocked"] == 0


def test_egress_blocked_is_an_oserror_that_survives_pickling() -> None:
    exc = EgressBlocked("getaddrinfo", "example.com:443")
    assert isinstance(exc, OSError)
    assert exc.errno == errno.EACCES
    assert "getaddrinfo" in str(exc)
    assert "example.com:443" in str(exc)
    clone = pickle.loads(pickle.dumps(exc))
    assert type(clone) is EgressBlocked
    assert (clone.kind, clone.target, clone.errno, str(clone)) == (exc.kind, exc.target, exc.errno, str(exc))
