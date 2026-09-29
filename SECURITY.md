# Security policy

## Supported versions

The current `0.1.x` release line receives security fixes.

## Reporting a vulnerability

Please report a security issue privately through GitHub's security advisories for this
repository (the repository's Security tab, "Report a vulnerability"), rather than opening a
public issue. Include the version, the command or API call that triggers it, and, where possible,
a minimal way to reproduce it. Expect an acknowledgement within a few days.

## Scope

wellbrief is a single-user, local tool: `wellbrief serve` and `wellbrief demo` bind to
`127.0.0.1` by default, and there is no authentication of any kind. The following are in scope as
security issues:

- A way to make the network guard or the egress rule allow an outbound connection they are
  documented to refuse (a host that is not loopback and was not explicitly allowed with
  `WELLBRIEF_ALLOW_REMOTE=1`, or a connection the guard should have counted and blocked).
- Cross-site scripting in the local web UI: any way for a well file's own content, once ingested,
  to execute as script or markup rather than being rendered as plain text.
- Path traversal: any way for an ingest path, a document id, or an API parameter to read or write
  a file outside the intended workspace or ingest folder.
- Cross-site or DNS-rebinding access to the local API: the server refuses a request whose `Host`
  header is not a loopback name or the configured `--host`, and a `POST` whose `Origin` is missing
  or foreign or whose `Content-Type` is not `application/json`. A way around those checks from a
  web page is in scope.
- A way for the cost ledger's budget stop to be bypassed, letting the `llm` narrator spend past
  `WELLBRIEF_BUDGET_USD`.

Explicitly out of scope: running `wellbrief serve --host 0.0.0.0` (or any other non-loopback
bind) on an untrusted network — the CLI prints a warning that this exposes an unauthenticated
API, and doing it anyway is a deployment choice, not a vulnerability in the tool. The tool ships
no authentication and no multi-user access control by design (see the README's Limitations
section); a request for either is a feature request, not a security report.

## Disclosure

Please allow time to investigate and release a fix before any public disclosure. Credit is given
in the release notes unless you ask otherwise.
