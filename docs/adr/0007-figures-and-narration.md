# ADR 0007: Figures from SQL, the language model only narrates, verification with a fallback

## Status

Accepted.

## Context

An answer or a brief has to state numbers a drilling engineer will act on — hours of NPT, a
dollar cost, a well count. A language model asked to compute or recall a number from a pile of
retrieved text is exactly the kind of task language models get subtly wrong, and a subtly wrong
number in a decision-support tool is worse than an obviously missing one.

## Decision

Every number in an answer or a brief is computed with SQL over the `npt_events` ledger before any
narrator sees the question: totals, per-code and per-well breakdowns, avoidable share, cost at the
configured spread rate, and every risk-brief statistic (ADR 0011). A narrator — the deterministic
`offline` templating narrator, or the `llm` narrator behind the egress guard (ADR 0009) — only
phrases the figures that were already computed and the mitigations that were already mined; it is
never asked to compute a number itself or to decide which documents are relevant. Whatever text a
narrator produces, a verifier checks it against the evidence before it is shown: every cited
document id must be in the evidence pack, every number in the text (after normalising thousands
separators, rounding and units) must trace to a quote, a computed figure, or the question itself,
and no well or field name outside the evidence may appear. Text that fails any check is replaced
by the deterministic offline narrator's own answer, with a visible banner, and the rejected text
and the reasons are kept in the JSON output rather than discarded.

## Consequences

A wrong number is a bug in the SQL or the parser, something a unit test can pin down, rather than
a language model's fluent-sounding guess that only a domain expert would catch by eye. The
`offline` narrator alone is enough to ship a fully working product with no language model at all,
and it doubles as every other narrator's safety net. The cost is that the tool can never answer a
question that requires judgement no formula in `analytics.py` or `riskbrief.py` expresses; it
abstains instead of guessing.
