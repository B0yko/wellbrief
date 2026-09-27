# ADR 0005: Configuration precedence

## Status

Accepted.

## Context

Every site's report template is a little different: different label text on a daily report,
different section headings on an end-of-well report, its own short NPT codes. The tool has to
adapt to that without a code change, while still behaving predictably when more than one source
could set the same value (a command-line flag, an environment variable, a file passed to
`ingest --config`, a workspace's own settings file).

## Decision

Precedence, highest first: CLI flags; the environment variables that have one
(`WELLBRIEF_HOME`, `WELLBRIEF_WORKSPACE`, `WELLBRIEF_SPREAD_RATE_USD_PER_DAY`); `ingest --config
PATH`, or, when that is not given, the ingested folder's own `wellbrief.toml`; the workspace's own
`wellbrief.toml`; the built-in defaults. The parse, detection, taxonomy, CSV and retrieval/risk
tables are `wellbrief.toml`-only (no environment variable), since they only ever matter through a
file a site actually wrote, and the ingest-time layers only apply to `ingest` itself — a command
like `ask` or `brief` reads the workspace's own file and the defaults, nothing more. A whole table
is merged key by key across layers (a lower-precedence file's untouched keys survive), while a
list-valued key such as a code's keyword list is replaced wholesale by whichever layer last set
it, not appended to, so a site can shrink or reorder a list without fighting the default's own
entries.

Every table and key is validated against a fixed, known shape: an unrecognised table, an
unrecognised key inside a known table, or a value of the wrong type is a `SettingsError` naming
the file and the exact key, never a silently ignored typo that leaves a site's override
unapplied without any sign of it.

## Consequences

A site maps its own report template once, in one `wellbrief.toml`, and every command that reads
that workspace picks it up automatically; `ingest --config` lets a single ingest of a
differently-templated folder override that without touching the workspace's own file.
`examples/alt-template/` and `examples/npt-ledger.csv` exist specifically to prove this against a
template that differs from the canonical one in every mapped field. The cost is a longer
`settings.py`: every table's validation is written by hand rather than left to a generic loader,
in exchange for an error message that always names the actual mistake.
