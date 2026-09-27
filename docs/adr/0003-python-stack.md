# ADR 0003: Python version, build backend, layout and entry point

## Status

Accepted.

## Context

The project needed a packaging setup that installs cleanly with `pip` or `uvx` from a plain git
checkout, produces a small pure-Python wheel suitable for an air-gapped wheelhouse, and gives one
command-line entry point rather than a script the caller has to locate and invoke with `python`.

## Decision

Python 3.12, required by `pyproject.toml`. `hatchling` as the build backend: it needs no plugin
for a pure-Python `src`-layout package, and its `force-include` table is enough to also place the
eval case files (which live at the repository's own `evals/cases/`, outside the package
directory) inside the built wheel, so a built package finds them the same way a repository
checkout does. `src/wellbrief/` layout, so the package cannot be imported by accident from a
checkout's working directory without being installed first. One console script,
`wellbrief = wellbrief.cli:main`, registered in `[project.scripts]`.

## Consequences

`pip install`, `uv pip install`, and `uvx --from <checkout-or-git-url> wellbrief ...` all work the
same way, from a fresh clone or from a built wheel, with no separate step to make the eval suite
runnable. The `src` layout costs a little more configuration (packages have to be told where to
look) in exchange for tests always exercising the installed package, not an accidental local
import.
