# ADR 0009: A process-level network guard, and an egress rule decided without DNS

## Status

Accepted.

## Context

wellbrief is meant to run with no network access at all, and the one feature that could send
data out — the optional `llm` narrator — needs a rule for which hosts it may ever contact. A
lookup to resolve that host's name is itself outbound network traffic, so the rule cannot depend
on knowing what a host name resolves to; and every other command needs a way to prove, not just
assert, that it never made an outbound connection.

## Decision

Two independent mechanisms:

- **The egress rule** (`egress.check_egress`) allows a configured LLM base URL only when its host
  is a loopback literal (`127.0.0.0/8`, `::1`, or the name `localhost`), decided on the URL text
  alone. A non-loopback host is refused unless `WELLBRIEF_ALLOW_REMOTE=1` is set explicitly. No
  DNS lookup is performed to make this decision, because the lookup's answer could differ between
  the check and the connection, and because performing it would itself be the outbound traffic
  the rule exists to prevent.
- **The process-level network guard** (`netguard.install`), installed by the CLI's entry point
  before any command runs, replaces `socket.socket.connect`, `connect_ex`, `create_connection`
  and `getaddrinfo`. Loopback targets, Unix sockets and the unspecified addresses are always
  allowed; the one configured LLM host is allowed as well, but only while the invocation's
  narrator is `llm`. Every other connection or name lookup raises before anything is sent and is
  counted, which is what `status`, `eval` and `selfcheck` report as "outbound connection
  attempts: N". `selfcheck` also runs a positive control — one connection attempt to a documented
  test address — and confirms the guard actually blocked it, so the zero above is a guard that was
  proven to work during this run, not a guard that never had anything to block.

The guard's limits are stated openly rather than left implicit: it does not intercept direct use
of the low-level `_socket` module, `sendto`/`sendmsg` on an unconnected datagram socket,
`sendto`/`sendmsg` with `MSG_FASTOPEN` on Linux (which opens a connection without calling
`connect`), the `gethostbyname`/`gethostbyaddr`/`getnameinfo` family, a subprocess, or a C
extension with its own networking. Importing the package installs nothing; only the CLI's own
entry point calls `install()`.

## Consequences

A caller can prove a given run made no outbound connection by reading one counter, and
`docker run --network none ... selfcheck` gives the same proof enforced by the operating system
independently of the guard. The guard is a tripwire against accidental egress through the
standard socket layer, not a sandbox: the README and this ADR both say so, and running with the
network disabled at the operating-system or container level (as the air-gapped install path does)
remains the control to rely on when that matters more than convenience.
