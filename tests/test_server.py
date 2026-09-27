"""The local HTTP API (`wellbrief serve`): endpoints, headers, error paths and the
no-authentication warning.

Every test builds its own tiny workspace (`mini_workspace`, a hand-built store small enough to
total by hand -- see its own docstring) and runs the real `server.create_server` server on an
ephemeral loopback port in a background thread, so these are round trips over real HTTP, not
calls into the handler's Python objects.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import mini_workspace as mini
from wellbrief import cli, server
from wellbrief.models import Document
from wellbrief.narrate import OfflineNarrator
from wellbrief.workspace import Workspace

XSS_TEXT = "END OF WELL REPORT\n<script>alert(1)</script><img src=x onerror=alert(2)>\n"


@pytest.fixture(scope="module")
def ws(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    workspace = Workspace(home=tmp_path_factory.mktemp("server-home"), name="ws")
    workspace.root.mkdir(parents=True, exist_ok=True)
    store = mini.store(workspace.root)
    store.put_documents([Document("EOWR-XSS-001", "eowr", "XSS-001", "Orrindale", "2024-01-01",
                                  "END OF WELL REPORT", XSS_TEXT)])
    store.close()
    return workspace


@pytest.fixture(scope="module")
def ghost_ws(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """A field with NPT ledger rows but no daily reports, so `brief` raises
    `MissingDdrDataError` (the API's 422 path)."""
    from wellbrief.models import NptEvent
    from wellbrief.store import Store

    workspace = Workspace(home=tmp_path_factory.mktemp("ghost-home"), name="ws")
    workspace.root.mkdir(parents=True, exist_ok=True)
    store = Store(workspace.db_path)
    store.put_documents([Document("EOWR-GST-001", "eowr", "GST-001", "GhostField", "2024-01-01",
                                  "END OF WELL REPORT", "END OF WELL REPORT\nNo lessons recorded.")])
    store.put_npt([NptEvent("EOWR-GST-001", "GST-001", "GhostField", "2024-01-01", "STUCK_PIPE", 5.0,
                            '12 1/4"', "Some Formation", 1000.0, 1.30, "Rig-1", "Packed off briefly.")])
    store.close()
    return workspace


@pytest.fixture(scope="module")
def running_server(ws: Workspace) -> Iterator[str]:
    narrator = OfflineNarrator()
    httpd = server.create_server(ws, narrator, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(base: str, path: str, *, method: str = "GET",
            body: dict[str, Any] | None = None) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(base + path, data=data, method=method,
                  headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _json(base: str, path: str, *, method: str = "GET",
         body: dict[str, Any] | None = None) -> tuple[int, dict[str, str], Any]:
    status, headers, raw = _request(base, path, method=method, body=body)
    return status, headers, json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Security headers, on every response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "method"), [
    ("/", "GET"), ("/api/status", "GET"), ("/nope", "GET"), ("/api/ask", "GET"),
])
def test_every_response_carries_the_security_headers(running_server: str, path: str, method: str) -> None:
    _status, headers, _body = _request(running_server, path, method=method)
    assert headers["Content-Security-Policy"] == "default-src 'self'"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"


@pytest.mark.parametrize("path", ["/api/status", "/api/npt", "/nope"])
def test_api_responses_are_never_cached(running_server: str, path: str) -> None:
    _status, headers, _body = _request(running_server, path)
    assert headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------


# A same-origin, no-external-reference regression must also catch a protocol-relative URL
# (`//cdn.example.com/...`), which carries no `http://`/`https://` substring at all but is
# fetched over the network exactly like one and would be blocked by `default-src 'self'` only
# once a browser actually loaded the page -- this test should catch it long before that.
_PROTOCOL_RELATIVE_RE = re.compile(
    r'''(?:href|src)\s*=\s*["\']//|url\(\s*["\']?//|fetch\(\s*["\']//''', re.IGNORECASE)
# An inline `style="..."` attribute and an inline `<script>...</script>` body (as opposed to
# `<script src="...">`) are both blocked at runtime by `default-src 'self'` (it has no
# `'unsafe-inline'`), so either would silently stop working under the shipped CSP; catching them
# here is cheaper than finding out from a browser dropping the rule.
_INLINE_STYLE_ATTR_RE = re.compile(r'''<[a-zA-Z][a-zA-Z0-9]*\b[^>]*\bstyle\s*=\s*["\']''')
_SCRIPT_TAG_RE = re.compile(r'<script\b[^>]*>', re.IGNORECASE)


@pytest.mark.parametrize(("path", "content_type"), [
    ("/", "text/html"), ("/index.html", "text/html"), ("/app.js", "text/javascript"),
    ("/app.css", "text/css"),
])
def test_static_files_are_served_same_origin_with_no_external_reference(
        running_server: str, path: str, content_type: str) -> None:
    status, headers, body = _request(running_server, path)
    assert status == 200
    assert headers["Content-Type"].startswith(content_type)
    text = body.decode("utf-8")
    assert "http://" not in text and "https://" not in text
    assert not _PROTOCOL_RELATIVE_RE.search(text)


# app.js documents the DOM-safety rule in its own header comment, which names these APIs to
# explain why they are never called; a plain substring search would trip over that comment, so
# every "//" line comment is stripped first and the remaining source is checked for the actual
# sink patterns (an assignment or a call), not just the bare words.
_UNSAFE_DOM_SINKS = [
    re.compile(r"\.innerHTML\s*="),
    re.compile(r"\.outerHTML\s*="),
    re.compile(r"\.insertAdjacentHTML\s*\("),
    re.compile(r"document\s*\.\s*write\s*\("),
]


def test_app_js_never_assigns_unsafe_markup() -> None:
    lines = (server.WEB_DIR / "app.js").read_text(encoding="utf-8").splitlines()
    code_only = "\n".join(re.sub(r"//.*$", "", line) for line in lines)
    for pattern in _UNSAFE_DOM_SINKS:
        assert not pattern.search(code_only), f"app.js must never use {pattern.pattern}"


def test_index_html_has_no_inline_style_attributes() -> None:
    text = (server.WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert not _INLINE_STYLE_ATTR_RE.search(text)


def test_index_html_scripts_all_carry_a_src_attribute() -> None:
    text = (server.WEB_DIR / "index.html").read_text(encoding="utf-8")
    tags = _SCRIPT_TAG_RE.findall(text)
    assert tags, "expected at least one <script> tag in index.html"
    for tag in tags:
        assert re.search(r"\bsrc\s*=", tag, re.IGNORECASE), f"inline <script> with no src: {tag!r}"


def test_index_html_has_the_three_keyboard_accessible_tabs() -> None:
    text = (server.WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert text.count('role="tab"') == 3
    assert text.count('role="tabpanel"') == 3
    for name in ("ask", "brief", "npt"):
        assert f'id="tab-{name}"' in text
        assert f'id="panel-{name}"' in text
        assert f'aria-controls="panel-{name}"' in text


def test_index_html_gates_brief_downloads_disabled_until_a_brief_is_built() -> None:
    text = (server.WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="brief-download-md"' in text
    assert 'id="brief-download-json"' in text
    # Neither download link carries an href until app.js builds one for a verified brief.
    assert not re.search(r'id="brief-download-(?:md|json)"[^>]*\bhref=', text)


def test_app_js_calls_every_json_endpoint() -> None:
    # Not a browser test (none is available here): a plain source check that the page actually
    # wires up every endpoint `server.py` implements, so a rename on one side would show up here.
    text = (server.WEB_DIR / "app.js").read_text(encoding="utf-8")
    for endpoint in ("/api/status", "/api/ask", "/api/brief", "/api/npt", "/api/doc/", "/api/brief."):
        assert endpoint in text, f"app.js never references {endpoint}"


def test_static_file_post_is_method_not_allowed(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/app.js", method="POST", body={})
    assert status == 405
    assert "error" in payload


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


def test_status_reports_the_workspace_and_network_mode(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/status")
    assert status == 200
    assert payload["workspace"] == "ws"
    assert payload["network_mode"] == "offline"
    assert isinstance(payload["outbound_connection_attempts"], int)
    fields = {f["field"] for f in payload["fields"]}
    assert {"Orrindale", "Vessra South"} <= fields
    # Each field row also carries the material its own Ask-tab example questions are built
    # from (`analytics.example_topics`); this is the one key `status_payload` itself does not
    # produce (see `_status_payload_for`), so wiring it in is worth its own assertion here on
    # top of `test_analytics.py`'s direct coverage of the function.
    for row in payload["fields"]:
        assert set(row["example_topics"]) == {
            "top_avoidable_code_label", "top_section_for_code", "top_formation",
        }
        assert all(isinstance(v, str) for v in row["example_topics"].values())
    orrindale = next(f for f in payload["fields"] if f["field"] == "Orrindale")
    assert orrindale["example_topics"]["top_avoidable_code_label"]


def test_status_post_is_method_not_allowed(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/status", method="POST", body={})
    assert status == 405
    assert "error" in payload


# ---------------------------------------------------------------------------
# POST /api/ask
# ---------------------------------------------------------------------------


def test_ask_answers_a_known_question(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/ask", method="POST",
                                      body={"question": "Stuck pipe on Orrindale"})
    assert status == 200
    assert payload["abstained"] is False
    assert payload["citations"]
    assert payload["question"] == "Stuck pipe on Orrindale"


def test_ask_field_parameter_scopes_the_figures(running_server: str) -> None:
    question = "What weather delays were recorded?"
    _status, _headers, both = _json(running_server, "/api/ask", method="POST", body={"question": question})
    _status, _headers, scoped = _json(running_server, "/api/ask", method="POST",
                                      body={"question": question, "field": "Vessra South"})
    assert both["figures"]["total_hours"] == 10.0
    assert scoped["figures"]["total_hours"] == 6.0
    assert scoped["query_plan"]["fields"] == ["Vessra South"]
    assert scoped["question"].startswith("Vessra South field.")


def test_ask_rejects_a_missing_question(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/ask", method="POST", body={})
    assert status == 400
    assert "question" in payload["error"]


def test_ask_rejects_an_unknown_field(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/ask", method="POST",
                                      body={"question": "Stuck pipe", "field": "Nowhereland"})
    assert status == 400
    assert "Nowhereland" in payload["error"]


def test_ask_rejects_a_non_positive_top_k(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/ask", method="POST",
                                      body={"question": "Stuck pipe on Orrindale", "top_k": 0})
    assert status == 400
    assert "top_k" in payload["error"]


def test_ask_get_is_method_not_allowed(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/ask")
    assert status == 405
    assert "error" in payload


def test_ask_records_an_audit_line(running_server: str, ws: Workspace) -> None:
    from wellbrief import audit

    before = len(audit.read_lines(ws.audit_path))
    _json(running_server, "/api/ask", method="POST", body={"question": "Stuck pipe on Orrindale"})
    lines = audit.read_lines(ws.audit_path)
    assert len(lines) == before + 1
    assert lines[-1]["command"] == "ask"


def test_ask_with_an_xss_payload_question_comes_back_as_json_data_not_markup(running_server: str) -> None:
    # A question is never parsed as markup anywhere on this path: the API only ever returns it
    # inside a JSON string, and the page only ever renders it with `textContent` (see app.js).
    status, headers, payload = _json(running_server, "/api/ask", method="POST", body={"question": XSS_TEXT})
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert payload["question"] == XSS_TEXT


# ---------------------------------------------------------------------------
# GET /api/doc/<id>
# ---------------------------------------------------------------------------


def test_doc_returns_the_document(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/doc/EOWR-ORD-101")
    assert status == 200
    assert payload["doc_id"] == "EOWR-ORD-101"
    assert payload["field"] == "Orrindale"
    assert payload["pages"] is None
    assert "LESSONS LEARNED" in payload["text"]


def test_doc_with_html_payload_text_comes_back_as_json_data_not_markup(running_server: str) -> None:
    status, headers, payload = _json(running_server, "/api/doc/EOWR-XSS-001")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert payload["text"] == XSS_TEXT


def test_doc_not_found(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/doc/NO-SUCH-DOC")
    assert status == 404
    assert "NO-SUCH-DOC" in payload["error"]


def test_doc_post_is_method_not_allowed(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/doc/EOWR-ORD-101", method="POST", body={})
    assert status == 405
    assert "error" in payload


# ---------------------------------------------------------------------------
# GET /api/npt
# ---------------------------------------------------------------------------


def test_npt_rolls_up_by_field(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/npt?field=Vessra+South")
    assert status == 200
    assert payload["total_hours"] == pytest.approx(sum(
        r["hours"] for r in payload["by_code"]
    ))
    assert "since" not in payload


def test_npt_carries_the_since_filter(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/npt?since=2024-01-01")
    assert status == 200
    assert payload["since"] == "2024-01-01"


# ---------------------------------------------------------------------------
# POST /api/brief and the download endpoints
# ---------------------------------------------------------------------------


def test_brief_returns_a_well_formed_register(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/brief", method="POST",
                                      body={"field": "Orrindale", "well": "ORD-NEXT", "td": 3000})
    assert status == 200
    assert payload["well_name"] == "ORD-NEXT"
    assert payload["field_name"] == "Orrindale"
    assert isinstance(payload["risks"], list)
    assert payload["disclaimer"]
    assert "not_verified" in payload
    assert payload["verification"]["ok"] is True
    assert payload["provenance"]["narrator"] == "offline"


def test_brief_rejects_an_unknown_field(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/brief", method="POST",
                                      body={"field": "Nowhereland", "well": "X-1", "td": 1000})
    assert status == 400
    assert "Nowhereland" in payload["error"]


def test_brief_rejects_a_non_positive_td(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/brief", method="POST",
                                      body={"field": "Orrindale", "well": "X-1", "td": 0})
    assert status == 400
    assert "td" in payload["error"]


def test_brief_rejects_a_missing_well(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/brief", method="POST",
                                      body={"field": "Orrindale", "td": 1000})
    assert status == 400
    assert "well" in payload["error"]


def test_brief_rejects_a_td_that_overflows_to_infinity(running_server: str) -> None:
    # Ordinary, standards-compliant JSON syntax -- no non-standard "Infinity" token needed --
    # since Python's float() silently overflows a literal this large to inf.
    host, _, port_str = running_server.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port_str), timeout=10)
    try:
        body = b'{"field": "Orrindale", "well": "X-1", "td": 1e400}'
        conn.request("POST", "/api/brief", body=body, headers={"Content-Length": str(len(body))})
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 400
        assert "td" in payload["error"]
    finally:
        conn.close()


def test_brief_rejects_a_nan_spread_rate(running_server: str) -> None:
    # json.loads accepts the bare NaN/Infinity/-Infinity tokens by default; a value that would
    # never round-trip through a strict JSON.parse must not reach the response or the audit log.
    host, _, port_str = running_server.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port_str), timeout=10)
    try:
        body = b'{"field": "Orrindale", "well": "X-1", "td": 3000, "spread_rate": NaN}'
        conn.request("POST", "/api/brief", body=body, headers={"Content-Length": str(len(body))})
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 400
        assert "spread_rate" in payload["error"]
    finally:
        conn.close()


@pytest.mark.parametrize("value", ["nan", "inf", "-infinity"])
def test_brief_download_rejects_a_non_finite_td_in_the_query_string(running_server: str, value: str) -> None:
    # Python's float() also accepts these strings, so the query-string path needs the same
    # isfinite check as the JSON-body path above.
    status, _headers, payload = _json(
        running_server, f"/api/brief.md?field=Orrindale&well=X-1&td={value}")
    assert status == 400
    assert "td" in payload["error"]


def test_brief_on_a_field_with_no_daily_reports_is_unprocessable(running_server: str,
                                                                 ghost_ws: Workspace) -> None:
    narrator = OfflineNarrator()
    httpd = server.create_server(ghost_ws, narrator, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        status, _headers, payload = _json(base, "/api/brief", method="POST",
                                          body={"field": "GhostField", "well": "GST-NEXT", "td": 100})
        assert status == 422
        assert "daily reports" in payload["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_brief_md_download_has_the_attachment_header(running_server: str) -> None:
    status, headers, body = _request(
        running_server, "/api/brief.md?field=Orrindale&well=ORD-NEXT&td=3000")
    assert status == 200
    assert headers["Content-Type"].startswith("text/markdown")
    assert headers["Content-Disposition"] == 'attachment; filename="brief-ORD-NEXT.md"'
    assert headers["Cache-Control"] == "no-store"
    assert body.decode("utf-8").startswith("# Offset-well risk brief: ORD-NEXT")


def test_brief_json_download_has_the_attachment_header(running_server: str) -> None:
    status, headers, payload = _json(
        running_server, "/api/brief.json?field=Orrindale&well=ORD-NEXT&td=3000")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Content-Disposition"] == 'attachment; filename="brief-ORD-NEXT.json"'
    assert payload["well_name"] == "ORD-NEXT"


def test_brief_download_requires_field_well_and_td(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/api/brief.md?well=ORD-NEXT&td=3000")
    assert status == 400
    assert "field" in payload["error"]


def test_brief_records_an_audit_line(running_server: str, ws: Workspace) -> None:
    from wellbrief import audit

    before = len(audit.read_lines(ws.audit_path))
    _json(running_server, "/api/brief", method="POST",
         body={"field": "Orrindale", "well": "ORD-NEXT", "td": 3000})
    lines = audit.read_lines(ws.audit_path)
    assert len(lines) == before + 1
    assert lines[-1]["command"] == "brief"


# ---------------------------------------------------------------------------
# Request body size cap and malformed requests
# ---------------------------------------------------------------------------


def test_oversized_request_body_is_rejected(running_server: str) -> None:
    # `Content-Length` alone is enough to reject the request (see `_read_json_body`), so this
    # declares an oversized length but never actually writes a body: a client that checked
    # first, rather than one that already streamed the whole thing and hit a reset.
    host, _, port_str = running_server.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port_str), timeout=10)
    try:
        conn.putrequest("POST", "/api/ask", skip_host=False, skip_accept_encoding=True)
        conn.putheader("Content-Length", str(server.MAX_BODY_BYTES + 1))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        payload = json.loads(resp.read().decode("utf-8"))
        assert "error" in payload
    finally:
        conn.close()


def test_missing_content_length_is_rejected(running_server: str) -> None:
    host, _, port_str = running_server.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port_str), timeout=10)
    try:
        conn.putrequest("POST", "/api/ask", skip_host=False, skip_accept_encoding=True)
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 411
        resp.read()
    finally:
        conn.close()


def test_malformed_json_body_is_rejected(running_server: str) -> None:
    host, _, port_str = running_server.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port_str), timeout=10)
    try:
        body = b"{not json"
        conn.request("POST", "/api/ask", body=body, headers={"Content-Length": str(len(body))})
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
    finally:
        conn.close()


def test_a_request_body_that_is_not_a_json_object_is_rejected(running_server: str) -> None:
    # A syntactically valid JSON array, not the "missing Content-Length" case above.
    host, _, port_str = running_server.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port_str), timeout=10)
    try:
        body = b"[1, 2, 3]"
        conn.request("POST", "/api/ask", body=body, headers={"Content-Length": str(len(body))})
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Only GET and POST; unknown paths 404
# ---------------------------------------------------------------------------


def test_unknown_path_is_404(running_server: str) -> None:
    status, _headers, payload = _json(running_server, "/no/such/endpoint")
    assert status == 404
    assert "error" in payload


@pytest.mark.parametrize("method", ["DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])
def test_other_http_methods_are_rejected(running_server: str, method: str) -> None:
    # A HEAD response must carry no body (`http.client` enforces this on the read side too),
    # so this only checks the status, not a JSON payload.
    status, _headers, _body = _request(running_server, "/api/status", method=method)
    assert status == 405


# ---------------------------------------------------------------------------
# Binding: the no-authentication warning, and never calling socket.getfqdn
# ---------------------------------------------------------------------------


def test_bind_warning_is_none_for_loopback_hosts() -> None:
    assert server.bind_warning("127.0.0.1") is None
    assert server.bind_warning("localhost") is None


def test_bind_warning_names_the_host_for_anything_else() -> None:
    message = server.bind_warning("0.0.0.0")
    assert message is not None
    assert "0.0.0.0" in message
    assert "authentication" in message


def test_create_server_prints_the_warning_for_a_non_loopback_host(
        ws: Workspace, capsys: pytest.CaptureFixture[str]) -> None:
    narrator = OfflineNarrator()
    capsys.readouterr()
    httpd = server.create_server(ws, narrator, host="0.0.0.0", port=0)
    try:
        err = capsys.readouterr().err
        assert "no authentication" in err
    finally:
        httpd.server_close()


def test_create_server_is_silent_for_the_loopback_default(
        ws: Workspace, capsys: pytest.CaptureFixture[str]) -> None:
    narrator = OfflineNarrator()
    capsys.readouterr()
    httpd = server.create_server(ws, narrator, host="127.0.0.1", port=0)
    try:
        err = capsys.readouterr().err
        assert err == ""
    finally:
        httpd.server_close()


def test_server_bind_never_calls_getfqdn(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(host: str) -> str:
        raise AssertionError("socket.getfqdn must not be called while binding")

    monkeypatch.setattr(socket, "getfqdn", boom)
    narrator = OfflineNarrator()
    httpd = server.create_server(ws, narrator, host="127.0.0.1", port=0)
    try:
        assert httpd.server_name == "127.0.0.1"
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Concurrent /api/ask against a never-indexed workspace: `ThreadingHTTPServer` gives every
# request its own thread, and building/persisting a field's on-disk index has to be safe when
# several of those threads reach it at once.
# ---------------------------------------------------------------------------


def test_concurrent_asks_against_a_never_indexed_workspace_do_not_race(
        tmp_path_factory: pytest.TempPathFactory) -> None:
    """A workspace that was ingested but never explicitly indexed (a plain `wellbrief ingest`
    followed by `wellbrief serve`, with no `wellbrief index` in between) builds its field indexes
    lazily, on the first request that needs them. A burst of concurrent `/api/ask` requests must
    all still get a clean 200, not an intermittent 500 from two threads writing the same index
    files at once.
    """
    workspace = Workspace(home=tmp_path_factory.mktemp("race-home"), name="ws")
    workspace.root.mkdir(parents=True, exist_ok=True)
    store = mini.store(workspace.root)
    store.close()
    assert not workspace.fields_dir.exists()  # confirms this really is the never-indexed case

    narrator = OfflineNarrator()
    httpd = server.create_server(workspace, narrator, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"

        def ask(_: int) -> int:
            status, _headers, _payload = _json(base, "/api/ask", method="POST",
                                               body={"question": "Stuck pipe on Orrindale"})
            return status

        with ThreadPoolExecutor(max_workers=24) as pool:
            statuses = list(pool.map(ask, range(24)))
        assert statuses == [200] * 24
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_the_serve_subcommand_defaults_to_loopback(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBRIEF_HOME", str(tmp_path / "home"))
    args = cli.build_parser().parse_args(["serve"])
    assert args.func is cli.cmd_serve
    assert args.host == server.DEFAULT_HOST
    assert args.port == server.DEFAULT_PORT


def test_the_serve_subcommand_accepts_host_and_port() -> None:
    args = cli.build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "9001"])
    assert args.host == "0.0.0.0"
    assert args.port == 9001
