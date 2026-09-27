"""Local HTTP API and static UI.

`wellbrief serve` starts a stdlib `http.server.ThreadingHTTPServer` bound to `--host`/`--port`
(default `127.0.0.1:8765`) that serves two things on one origin: the JSON API under `/api/...`,
and the static single-page UI (`web/index.html`, `web/app.js`, `web/app.css`, packaged inside the
wheel next to this module) at every other path.

Every JSON endpoint returns exactly the payload its CLI equivalent's `--json` output does --
`Answer.to_dict()` for `ask` (`qa.ask` itself), `riskbrief.brief_json` for `brief`,
`workspace.status_payload` for `status`, `analytics.npt_payload` for `npt` -- built by the same
functions the CLI commands call, not reimplemented here, so a caller never sees the CLI and the
API disagree. `ask` and `brief` requests append to the workspace's `audit.jsonl` exactly like
their CLI commands (`audit.record_ask`/`audit.record_brief`).

Endpoints:

- `GET  /api/status`
- `POST /api/ask`             `{"question": str, "field"?: str, "top_k"?: int}`
- `POST /api/brief`           `{"field": str, "well": str, "td": number, "rig"?: str, "mwd"?: str,
                                "spread_rate"?: number}`
- `GET  /api/npt`             `?field=&well=&code=&since=`
- `GET  /api/doc/<id>`
- `GET  /api/brief.md`        same parameters as `POST /api/brief`, as a query string;
  `GET  /api/brief.json`      answered with `Content-Disposition: attachment`

A `field` given to `/api/ask` scopes the question the same way naming the field in the question
text would: `search.plan_query` reads any known field name it finds anywhere in the text, so this
prefixes it (`"<field> field. <question>"`) rather than reimplementing that matching here; the
resulting `Answer.question` (and the audit record) show the question actually planned.

Security:

- Every response carries `Content-Security-Policy: default-src 'self'`,
  `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`; every API response (JSON
  or a brief download) also carries `Cache-Control: no-store`.
- A request body over `MAX_BODY_BYTES`, or one with no `Content-Length` at all, is rejected before
  it is read.
- Every numeric parameter (`td`, `top_k`, `spread_rate`, whether given as JSON or on a query
  string) is checked with `math.isfinite`: `NaN`/`Infinity`/`-Infinity` are rejected with a 400
  rather than flowing into a brief and its JSON response (where they would not round-trip through
  a standards-compliant `JSON.parse`) or into the audit log.
- Only `GET` and `POST` are implemented; every other method gets a clean 405, not the stdlib's
  default HTML error page.
- Binding to a host that is not a loopback literal prints a warning (`bind_warning`): the API has
  no login, no session and no per-caller authorization, so anyone who can reach that host can call
  every endpoint.
- The network guard `cli.main()` installs stays in effect while serving; this module does not
  touch it.
- `WellbriefHTTPServer.server_bind` never calls `socket.getfqdn` (`http.server.HTTPServer`'s own
  `server_bind` does, to set `server_name`, which can trigger a reverse DNS lookup); this server
  has no use for a resolved name, so `server_name` is set to the bound host literal instead.
- The static files carry no external reference of any kind (no CDN script, no web font, no
  `http://`/`https://` URL, no inline script or style attribute): everything this origin serves,
  it also owns.

Every request opens its own store (`Workspace.open_store`) and, for `ask`/`brief`/`npt`, its own
resolved settings and (for `ask`/`brief`) searcher, used only within that request's handler call
and closed before it returns -- the same "open, use, close" shape every CLI command already has,
and the simplest way to give a `ThreadingHTTPServer`'s per-request threads a SQLite connection
each may use on its own (a connection may only be used on the thread that opened it).
"""

from __future__ import annotations

import json
import math
import re
import socketserver
import sys
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__, analytics, audit, riskbrief
from .egress import is_loopback_host
from .models import RiskBrief
from .narrate import Narrator
from .qa import ask as ask_qa
from .settings import SettingsError, resolve_settings
from .workspace import Workspace, index_files_hash, status_payload, wire_searcher

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_BODY_BYTES",
    "WEB_DIR",
    "ApiError",
    "WellbriefHTTPServer",
    "bind_warning",
    "create_server",
]

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 1_000_000  # generous for a question or a brief request, small enough to cap abuse

_STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}

NO_AUTH_WARNING = ("binding to {host} exposes the JSON API on that host with no authentication; "
                   "only do this on a trusted network")


class ApiError(Exception):
    """A clean JSON error response: raised by a route handler, turned into one by the request
    handler's dispatch loop instead of letting a raw traceback reach the caller."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Request parameter validation
# ---------------------------------------------------------------------------


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be a non-empty string")
    return value


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be a string")
    return value or None


def _require_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be a finite number")
    if positive and number <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be greater than zero")
    return number


def _optional_number(value: Any, name: str, *, positive: bool = False) -> float | None:
    return None if value is None else _require_number(value, name, positive=positive)


def _parse_float(raw: str, name: str, *, positive: bool = False) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be a number") from None
    if not math.isfinite(value):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be a finite number")
    if positive and value <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name!r} must be greater than zero")
    return value


def _query_first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


@dataclass(frozen=True)
class _AskParams:
    question: str
    field: str | None
    top_k: int | None


def _ask_params_from_json(body: dict[str, Any]) -> _AskParams:
    question = _require_str(body.get("question"), "question")
    field = _optional_str(body.get("field"), "field")
    top_k_raw = body.get("top_k")
    top_k: int | None = None
    if top_k_raw is not None:
        if isinstance(top_k_raw, bool) or not isinstance(top_k_raw, int) or top_k_raw < 1:
            raise ApiError(HTTPStatus.BAD_REQUEST, "'top_k' must be a positive integer")
        top_k = top_k_raw
    return _AskParams(question=question, field=field, top_k=top_k)


@dataclass(frozen=True)
class _BriefParams:
    field: str
    well: str
    td: float
    rig: str | None
    mwd: str | None
    spread_rate: float | None


def _brief_params_from_json(body: dict[str, Any]) -> _BriefParams:
    return _BriefParams(
        field=_require_str(body.get("field"), "field"),
        well=_require_str(body.get("well"), "well"),
        td=_require_number(body.get("td"), "td", positive=True),
        rig=_optional_str(body.get("rig"), "rig"),
        mwd=_optional_str(body.get("mwd"), "mwd"),
        spread_rate=_optional_number(body.get("spread_rate"), "spread_rate", positive=True),
    )


def _brief_params_from_query(query: dict[str, list[str]]) -> _BriefParams:
    field = _query_first(query, "field")
    well = _query_first(query, "well")
    td_raw = _query_first(query, "td")
    if not field:
        raise ApiError(HTTPStatus.BAD_REQUEST, "'field' is required")
    if not well:
        raise ApiError(HTTPStatus.BAD_REQUEST, "'well' is required")
    if not td_raw:
        raise ApiError(HTTPStatus.BAD_REQUEST, "'td' is required")
    spread_rate_raw = _query_first(query, "spread_rate")
    return _BriefParams(
        field=field, well=well, td=_parse_float(td_raw, "td", positive=True),
        rig=_query_first(query, "rig") or None, mwd=_query_first(query, "mwd") or None,
        spread_rate=_parse_float(spread_rate_raw, "spread_rate", positive=True) if spread_rate_raw else None,
    )


_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _brief_filename(well: str, fmt: str) -> str:
    safe = _UNSAFE_FILENAME_RE.sub("-", well).strip("-") or "well"
    return f"brief-{safe}.{fmt}"


# ---------------------------------------------------------------------------
# Payload builders: each opens its own store, so a request never shares a SQLite
# connection with another thread's (see the module docstring).
# ---------------------------------------------------------------------------


def _doc_payload(ws: Workspace, doc_id: str) -> dict[str, Any]:
    store = ws.open_store()
    try:
        doc = store.get_document(doc_id)
        if doc is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such document: {doc_id}")
        return {
            "doc_id": doc.doc_id, "doc_type": doc.doc_type, "well": doc.well,
            "field": doc.field_name, "date": doc.date, "title": doc.title,
            "text": doc.text, "pages": doc.page_map,
        }
    finally:
        store.close()


def _status_payload_for(ws: Workspace, narrator: Narrator) -> dict[str, Any]:
    """`workspace.status_payload`, plus one key the web UI alone needs: each field row's
    `example_topics` (`analytics.example_topics`), the material its own example questions are
    built from. Added here rather than in `status_payload` itself so `wellbrief status --json`,
    which nothing in the UI reads, stays exactly the workspace/index health report it always was.
    """
    store = ws.open_store()
    try:
        payload = status_payload(ws, store, narrator.name)
        for row in payload["fields"]:
            row["example_topics"] = analytics.example_topics(store, row["field"])
        return payload
    finally:
        store.close()


def _npt_payload_for(ws: Workspace, query: dict[str, list[str]]) -> dict[str, Any]:
    store = ws.open_store()
    try:
        settings = resolve_settings(workspace_root=ws.root)
        return analytics.npt_payload(
            store, settings.spread_rate_usd_per_day,
            field_name=_query_first(query, "field"), well=_query_first(query, "well"),
            code=_query_first(query, "code"), since=_query_first(query, "since"),
        )
    finally:
        store.close()


def _handle_ask(body: dict[str, Any], ws: Workspace, narrator: Narrator) -> dict[str, Any]:
    params = _ask_params_from_json(body)
    store = ws.open_store()
    try:
        known_fields = store.field_names()
        if params.field is not None and params.field not in known_fields:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"unknown field {params.field!r}; known: {', '.join(known_fields) or '-'}")
        settings = resolve_settings(workspace_root=ws.root, top_k=params.top_k)
        searcher = wire_searcher(ws, store, settings)
        effective_question = f"{params.field} field. {params.question}" if params.field else params.question
        answer = ask_qa(effective_question, store, searcher, narrator=narrator,
                        top_k=settings.retrieval.top_k, spread_rate=settings.spread_rate_usd_per_day,
                        keywords=settings.taxonomy.keywords)
        audit.record_ask(ws.audit_path, store, effective_question, answer)
        return answer.to_dict()
    finally:
        store.close()


def _build_brief(ws: Workspace, narrator: Narrator, params: _BriefParams,
                 fmt: str) -> tuple[RiskBrief, dict[str, Any]]:
    store = ws.open_store()
    try:
        known_fields = store.field_names()
        if params.field not in known_fields:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"unknown field {params.field!r}; known: {', '.join(known_fields) or '-'}")
        settings = resolve_settings(workspace_root=ws.root, spread_rate=params.spread_rate)
        risk_thresholds = {k: v for k, v in asdict(settings.risk).items() if k != "max_risks"}
        verification_placeholder: dict[str, Any] = {}
        provenance = riskbrief.build_provenance(
            store, params.field, settings.spread_rate_usd_per_day, narrator.name, verification_placeholder,
            index_manifest_hash=index_files_hash(ws, params.field), risk_thresholds=risk_thresholds,
        )
        try:
            brief = riskbrief.build_brief(
                store, params.well, params.field, params.td,
                spread_rate=settings.spread_rate_usd_per_day, narrator=narrator,
                max_risks=settings.risk.max_risks, plan_rig=params.rig, plan_mwd=params.mwd,
                provenance=provenance, risk_thresholds=risk_thresholds, keywords=settings.taxonomy.keywords,
            )
        except riskbrief.MissingDdrDataError as exc:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
        check = riskbrief.verify_brief(brief, store)
        verification_placeholder.update(check)
        audit.record_brief(ws.audit_path, brief, check, parameters={
            "field": params.field, "well": params.well, "td_m": params.td,
            "rig": params.rig, "mwd": params.mwd,
            "spread_rate_usd_per_day": settings.spread_rate_usd_per_day, "format": fmt,
        })
        return brief, check
    finally:
        store.close()


def _static_bytes(filename: str) -> bytes:
    try:
        return (WEB_DIR / filename).read_bytes()
    except OSError as exc:
        raise ApiError(HTTPStatus.NOT_FOUND, f"missing static file: {filename}") from exc


# ---------------------------------------------------------------------------
# The request handler
# ---------------------------------------------------------------------------


def _build_handler_class(ws: Workspace, narrator: Narrator) -> type[BaseHTTPRequestHandler]:
    """One `BaseHTTPRequestHandler` subclass bound to `ws`/`narrator` by closure (rather than by
    an attribute on the server object), so every route handler reaches them as plain, correctly
    typed local variables."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"wellbrief/{__version__}"

        def log_message(self, log_format: str, *args: Any) -> None:
            pass  # a local developer tool's own request log adds noise, not information

        # -- sending responses --------------------------------------------------

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, *, no_store: bool,
                  attachment: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'self'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if no_store:
                self.send_header("Cache-Control", "no-store")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            if self.command != "HEAD":  # a HEAD response must carry the headers only, never a body
                self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: Any, *, attachment: str | None = None) -> None:
            body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", no_store=True, attachment=attachment)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json(status, {"error": message})

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            try:
                length = int(raw_length)
            except ValueError:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length is not a number") from None
            if length < 0:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length must not be negative")
            if length > MAX_BODY_BYTES:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               f"request body exceeds {MAX_BODY_BYTES} bytes")
            raw_body = self.rfile.read(length) if length else b""
            if not raw_body:
                return {}
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"request body is not valid JSON: {exc}") from None
            if not isinstance(data, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
            return data

        # -- routing --------------------------------------------------------

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _not_get_or_post(self) -> None:
            self._send_error(HTTPStatus.METHOD_NOT_ALLOWED,
                             f"{self.command} is not allowed; only GET and POST are")

        do_HEAD = _not_get_or_post
        do_PUT = _not_get_or_post
        do_DELETE = _not_get_or_post
        do_PATCH = _not_get_or_post
        do_OPTIONS = _not_get_or_post

        def _expect(self, method: str, expected: str, path: str) -> None:
            if method != expected:
                raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED,
                               f"{method} not allowed on {path}; expected {expected}")

        def _dispatch(self, method: str) -> None:
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)
            try:
                self._route(method, path, query)
            except ApiError as exc:
                self._send_error(exc.status, exc.message)
            except SettingsError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:  # noqa: BLE001 -- a clean 500 for the caller, never a raw traceback
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

        def _route(self, method: str, path: str, query: dict[str, list[str]]) -> None:
            if path in _STATIC_FILES:
                self._expect(method, "GET", path)
                filename, content_type = _STATIC_FILES[path]
                self._send(HTTPStatus.OK, _static_bytes(filename), content_type, no_store=False)
                return
            if path.startswith("/api/doc/") and len(path) > len("/api/doc/"):
                self._expect(method, "GET", path)
                self._send_json(HTTPStatus.OK, _doc_payload(ws, unquote(path[len("/api/doc/"):])))
                return
            if path == "/api/status":
                self._expect(method, "GET", path)
                self._send_json(HTTPStatus.OK, _status_payload_for(ws, narrator))
                return
            if path == "/api/ask":
                self._expect(method, "POST", path)
                self._send_json(HTTPStatus.OK, _handle_ask(self._read_json_body(), ws, narrator))
                return
            if path == "/api/brief":
                self._expect(method, "POST", path)
                brief, check = _build_brief(ws, narrator, _brief_params_from_json(self._read_json_body()),
                                            "json")
                self._send_json(HTTPStatus.OK, riskbrief.brief_json(brief, check))
                return
            if path == "/api/npt":
                self._expect(method, "GET", path)
                self._send_json(HTTPStatus.OK, _npt_payload_for(ws, query))
                return
            if path in ("/api/brief.md", "/api/brief.json"):
                self._expect(method, "GET", path)
                fmt = "md" if path.endswith(".md") else "json"
                params = _brief_params_from_query(query)
                brief, check = _build_brief(ws, narrator, params, fmt)
                filename = _brief_filename(params.well, fmt)
                if fmt == "json":
                    self._send_json(HTTPStatus.OK, riskbrief.brief_json(brief, check), attachment=filename)
                else:
                    self._send(HTTPStatus.OK, riskbrief.render_markdown(brief, check).encode("utf-8"),
                              "text/markdown; charset=utf-8", no_store=True, attachment=filename)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such endpoint: {path}")

    return Handler


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


class WellbriefHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        """Bind without ever calling `socket.getfqdn` (see the module docstring).

        Reuses `socketserver.TCPServer`'s own bind (address reuse, then the actual `bind`), and
        sets `server_name`/`server_port` the way `http.server.HTTPServer.server_bind` does, minus
        the reverse-DNS lookup this server has no use for.
        """
        socketserver.TCPServer.server_bind(self)
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])


def bind_warning(host: str) -> str | None:
    """A warning to print before binding to `host`, when it is not a loopback literal: the API
    has no login, no session and no per-caller authorization, so anyone who can reach that host
    can call every endpoint. `None` for a loopback host (the default), where only this machine
    can connect.
    """
    return None if is_loopback_host(host) else NO_AUTH_WARNING.format(host=host)


def create_server(ws: Workspace, narrator: Narrator, host: str = DEFAULT_HOST,
                  port: int = DEFAULT_PORT) -> WellbriefHTTPServer:
    """Build and bind (but do not yet run) the local HTTP server for `ws`: the JSON API under
    `/api/...`, plus the static single-page UI at every other path (see the module docstring).

    Prints `bind_warning(host)` to stderr when it is not `None`.

    Raises:
        OSError: `host`/`port` cannot be bound (for example the port is already in use).
    """
    warning = bind_warning(host)
    if warning is not None:
        print(f"wellbrief serve: warning: {warning}", file=sys.stderr)
    return WellbriefHTTPServer((host, port), _build_handler_class(ws, narrator))
