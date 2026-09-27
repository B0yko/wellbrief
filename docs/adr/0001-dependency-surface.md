# ADR 0001: Dependency surface

## Status

Accepted.

## Context

wellbrief is meant to run on a laptop or a sealed server with no network access, sometimes
reviewed by someone who has to be able to audit everything it installs before it touches a
site's own well files. Every dependency is something that review has to cover, and every
dependency is something an air-gapped install has to carry across on removable media.

## Decision

One runtime dependency: `pypdf`, pure Python and BSD-3-Clause, used only to extract text from PDF
reports. Everything else comes from the standard library: `sqlite3` for the store, `http.server`
for the local API and UI, `tomllib` for configuration, `zipfile` and `xml.etree` for writing and
reading the built-in DOCX support, `urllib` and `ssl` for the optional LLM narrator, `array` for
the on-disk vector file. The floor version of `pypdf` in `pyproject.toml` excludes every release
with a published advisory on the text-extraction code path this project's `readers.pdf.read_pdf`
uses.

Development-only dependencies (`pytest`, `pytest-cov`, `ruff`, `mypy`) never ship in the wheel and
never reach a production install.

## Consequences

A security reviewer in a sealed environment has one pure-Python package to read, not a
dependency tree. An air-gapped install's wheelhouse is exactly two files (the project's own wheel
and `pypdf`'s), which `docs/airgapped-install.md` verifies end to end inside a
`--network none` container. The cost is writing a few things by hand that a dependency would
otherwise provide — a minimal DOCX writer, a hashing embedder instead of a downloaded model, a
hand-rolled HTTP client for the optional LLM narrator — each documented in its own ADR.
