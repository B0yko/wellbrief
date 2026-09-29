# ADR 0004: Standard-library HTTP server and a vanilla JS UI

## Status

Accepted.

## Context

`wellbrief serve` and `wellbrief demo` need a local web UI: a question box with citations, a risk
brief form and table, and an NPT rollup view, reachable from a browser without adding a second
language's toolchain or a build step to a project that is otherwise a single Python wheel.

## Decision

`http.server.ThreadingHTTPServer` serves a small JSON API (`/api/status`, `/api/ask`,
`/api/brief`, `/api/npt`, `/api/doc/<id>`, `/api/brief.md`, `/api/brief.json`) and one page of
plain HTML, CSS and JS, all packaged inside the wheel. No framework, no bundler, no CDN and no
web font: every response carries `Content-Security-Policy: default-src 'self'`, which blocks
inline scripts and inline style attributes, so `app.js` and `app.css` are served as their own
same-origin files and bar widths are set through `element.style` rather than inline `style=`
attributes. All document text the UI renders goes through `textContent`, never innerHTML, so a
report's own text can never execute as markup. The server also checks the `Host` header of every
request (a loopback name, or the configured bind host with the bound port) and, on `POST`, the
`Origin` and `Content-Type: application/json`, which closes DNS-rebinding and cross-site request
paths to the unauthenticated API without adding a login.

## Consequences

The UI adds nothing to the dependency surface (ADR 0001) and nothing to the air-gapped install:
the same wheel that carries the CLI carries the whole UI, and there is no separate frontend
build to keep in sync with the API. The cost is a plainer UI than a component framework would
give — no client-side routing beyond hash fragments for deep links, and any interaction beyond a
form and a fetch call is written out by hand.
