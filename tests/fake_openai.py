"""A fake OpenAI-compatible chat-completions server for tests, bound to an ephemeral loopback port."""

from __future__ import annotations

import json
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class FakeResponse:
    """One canned answer. ``body`` may be a JSON-ready object, text, or raw bytes."""

    body: Any = None
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    drop: bool = False
    """Close the connection without answering."""


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def completion(
    text: str = "Stuck pipe was recorded on 3 wells [DDR-ORD-101-009].",
    *,
    prompt_tokens: int | None = 120,
    completion_tokens: int | None = 40,
    cost: float | None = None,
    model: str = "fake-model",
    finish_reason: str = "stop",
    extra_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a chat-completion body shaped like the OpenAI, mlx-lm and llama.cpp responses."""
    usage: dict[str, Any] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        usage["total_tokens"] = prompt_tokens + completion_tokens
    if cost is not None:
        usage["cost"] = cost
    usage.update(extra_usage or {})
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        body["usage"] = usage
    return body


class FakeOpenAIServer:
    """A threaded HTTP server that records requests and replays queued responses."""

    def __init__(self, tls: ssl.SSLContext | None = None) -> None:
        self.requests: list[RecordedRequest] = []
        self._queue: deque[FakeResponse] = deque()
        self.default = FakeResponse(completion())
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with server._lock:
                    server.requests.append(
                        RecordedRequest(
                            method=self.command,
                            path=self.path,
                            headers={k.lower(): v for k, v in self.headers.items()},
                            body=body,
                        )
                    )
                    reply = server._queue.popleft() if server._queue else server.default
                if reply.delay_s:
                    time.sleep(reply.delay_s)
                if reply.drop:
                    self.close_connection = True
                    return
                if isinstance(reply.body, bytes):
                    payload = reply.body
                elif isinstance(reply.body, str):
                    payload = reply.body.encode("utf-8")
                else:
                    payload = json.dumps(reply.body).encode("utf-8")
                try:
                    self.send_response(reply.status)
                    headers = {"Content-Type": "application/json", **reply.headers}
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    pass

            do_POST = _handle
            do_GET = _handle

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request: Any, address: Any) -> None:
                return

        self._httpd = Server(("127.0.0.1", 0), Handler)
        self.scheme = "http"
        if tls is not None:
            self._httpd.socket = tls.wrap_socket(self._httpd.socket, server_side=True)
            self.scheme = "https"
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05})
        self._thread.daemon = True

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://127.0.0.1:{self.port}/v1"

    def queue(self, *responses: FakeResponse) -> None:
        with self._lock:
            self._queue.extend(responses)

    def start(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
