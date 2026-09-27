# Contributing

## Setup

```bash
uv sync
```

This installs the runtime dependency (`pypdf`) and the development group (`pytest`,
`pytest-cov`, `ruff`, `mypy`) into a local virtual environment. Python 3.12 is required.

## Before opening a change

```bash
uv run pytest -q                    # unit + integration tests
uv run ruff check .                 # lint
uv run mypy                         # strict type check on src/
uv run wellbrief eval --suite all   # the full offline eval suite, default seed
```

All four must pass. `mypy` runs in strict mode on `src/`; a change should not increase the error
count, and code it touches should be typed cleanly rather than suppressed. Tests must not touch
the network and must write only to a temporary directory: point `WELLBRIEF_HOME` at a `tmp_path`
in any test that opens a workspace, the way the existing tests do.

## Evaluation cases are not tuning knobs

`evals/cases/*.toml` holds the product's own claims about itself: retrieval precision, brief
precision, citation verification, and so on. Once a suite has had its first recorded run, no case
is removed and no case is loosened — a question, a relevance set, a gold query or a target stays
as it was. A case the current code cannot pass is marked non-gating with a stated reason and
still reported as a failure, never quietly dropped. Any change to a case file after its first
recorded run — including a stricter case, or a fix to a gold query that contradicted its own
comment — is logged in `evals/CHANGES.md`: the date, the case id, what changed, and why.

If a change makes an eval number worse, that is worth knowing and saying in the pull request, not
a reason to edit the case.

## Style

- Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `build:`, `ci:`,
  `chore:`, `perf:`), one concern per commit.
- Every module, class and public function carries a docstring explaining what it does and why,
  not just its signature.
- A new environment variable, TOML key or CLI flag is not complete until the README's
  configuration reference lists it.

## Adding a report template

A different site's report layout does not need a code change: map its labels, section headings
and NPT code aliases through `wellbrief.toml` (see the README's configuration reference and
`examples/alt-template/`). If you do need to change a parser, add a fixture under `examples/` or
a generated corpus in a test, and a test proving the new template's output matches the canonical
template's semantics, the way the existing alt-template test does.

## Reporting a security issue

See [SECURITY.md](SECURITY.md); please do not open a public issue for anything that touches the
network guard, the egress guard, or the local UI's security boundary.
