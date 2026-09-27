# ADR 0010: Test and CI strategy

## Status

Accepted.

## Context

The product's numbers (retrieval precision, brief precision, citation verification, performance)
are part of what the repository claims about itself, so they need to be reproducible by anyone,
on every push, without needing a paid API key or a GPU, while still exercising the real language
model path when that path is deliberately being measured.

## Decision

Unit tests cover the tokenizer, section-spelling canonicalisation, each report parser, the
PDF/DOCX/CSV readers, BM25, the hashing embedder, reciprocal rank fusion, the query planner, the
risk statistics and lift/ratio gates, pattern de-duplication and event ownership, the
practice/failure/neutral classifier, the citation verifier (including a fake narrator that
hallucinates a document id and a number, to prove the verifier actually catches it), the network
guard, the egress guard and cost ledger (against a fake OpenAI-compatible server on an ephemeral
loopback port, never a real one), and the HTTP API (including its Content-Security-Policy header
and an XSS regression). Integration tests cover `demo --no-browser` end to end on a reduced
corpus, ingest of `examples/alt-template/`, and the CLI's command surface end to end. Continuous
integration runs `ruff` and `mypy`, `pytest` on both `ubuntu-latest` and `macos-latest`,
`wellbrief eval --suite all` on the default seed (which must exit 0 and makes no outbound calls),
and a Docker job that builds the image and runs `docker run --rm --network none <image>
selfcheck`. No job calls a paid API; the narrator comparison against a real cloud endpoint is a
manual, budget-capped run whose recorded results are committed to `results/v0.1.0/`, not something
CI repeats on every push.

Once an eval suite's cases have had their first recorded run, no case is removed and no case is
loosened: a question, a relevance set, a gold query or a target stays as it was, and any change is
logged in `evals/CHANGES.md` with its reason. A case the current code still fails is marked
non-gating with a stated reason and reported as a failure, not silently excluded from the count.

## Consequences

Anyone can clone the repository and get the same green CI with no credentials at all. A
regression in retrieval, statistics or the verifier fails a test before it ever reaches an eval
number, and an eval case cannot be quietly redefined to make a regression disappear. The cost is
that CI cannot, by itself, catch a regression specific to the `llm` narrator against a real
endpoint; that path is covered by its own tests against a fake server plus the manual, recorded
comparison run.
