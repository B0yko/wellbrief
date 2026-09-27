# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-27

### Added

- `ingest`: incremental ingestion (content-hash keyed) of `.txt`, `.md`, `.pdf`, `.docx` and NPT
  ledger `.csv` files into a named workspace, with document-type detection, a per-run coverage
  table, and `--prune`/`--dry-run`.
- `ask`: cited answers over a field's well files. A query planner extracts field, well, hole
  section, formation, NPT code, depth and document type; structured filters run before hybrid
  BM25 + hashing-embedder retrieval; figures are computed with SQL over the NPT ledger; every
  citation is checked verbatim against its source before the answer is shown; an unmatched
  question abstains explicitly instead of guessing.
- `brief`: a pre-spud offset-well risk brief, ranking non-productive-time patterns (by hole
  section and formation, and by rig or downhole tool) with probability, expected hours and cost
  at a configurable spread rate, a named driver where the data supports one, and mitigations
  quoted verbatim from a well that avoided the problem or from an incident report's corrective
  actions for the same code.
- `npt` and `patterns`: non-productive-time rollups with cost, and the detected recurring
  patterns behind a brief.
- Two narrators: a deterministic `offline` template narrator (the default, no network and no
  model), and an `llm` narrator that talks to any OpenAI-compatible `/chat/completions` server
  behind an egress guard, a process-level network guard, and a cost ledger with a budget stop. A
  verifier checks any narrator's output against the evidence and falls back to the deterministic
  narrator, with a visible banner, on any failure.
- `corpus generate`: a seeded synthetic well-file corpus with a ground-truth sidecar, for trying
  the product with no real data and for the evaluation suites below.
- `eval` and `bench`: data-driven evaluation suites (retrieval precision, arithmetic, abstention,
  brief recall/driver/mitigation precision and recall, parser fidelity, format parity, a retrieval
  ablation, and a narrator-backend comparison) and performance benchmarks, both runnable as
  ordinary commands and in CI.
- `selfcheck`: an end-to-end offline check (corpus generation, ingest, indexing, the eval suites,
  brief verification) plus a network-guard positive control, exiting non-zero on any failure —
  the command a network-disabled proof runs against.
- `serve` and `demo`: a local JSON API and a single-page vanilla-JS UI (Ask, Brief and NPT tabs,
  deep links, a highlighted citation in the source document, downloadable Markdown/JSON briefs),
  and one command that generates a demo corpus, ingests it and opens the UI.
- Site-specific report templates and NPT code aliases through `wellbrief.toml`, with strict
  validation of every key; `examples/alt-template/` and `examples/npt-ledger.csv` demonstrate a
  differently labelled report template and a non-default CSV ledger layout.
- A multi-arch Docker image (`ghcr.io/b0yko/wellbrief`) with a pre-generated demo workspace, and
  a verified air-gapped install path (wheelhouse, and `docker save`/`docker load`) documented in
  `docs/airgapped-install.md`.
