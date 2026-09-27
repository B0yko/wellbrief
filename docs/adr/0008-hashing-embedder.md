# ADR 0008: A hashing embedder behind the `Embedder` interface

## Status

Accepted.

## Context

Hybrid retrieval needs a dense side alongside BM25, but downloading and running a neural
embedding model would add a model-download step to a tool whose default path is meant to need
zero gigabytes of downloads and no GPU, and would add a much larger dependency (a tensor runtime,
model weights) to the one-dependency surface ADR 0001 sets out.

## Decision

The dense side is a hashing embedder: every chunk's words and character 3/4-grams are hashed into
a fixed-size signed vector (512 dimensions), with no training and no model file. It sits behind an
`Embedder` interface with one other implementation possible later, so a future embeddings backend
(an OpenAI-compatible embeddings endpoint, for instance) is a new class behind the same interface,
not a rewrite of `search.py` or `workspace.py`. The retrieval ablation
(`wellbrief eval --ablation`) measures the hashing embedder against BM25 alone and against the
hybrid combination, and the product's documentation states plainly that it captures fuzzy lexical
similarity — a misspelling or a reordered phrase still scores well — not conceptual or semantic
similarity.

## Consequences

The default path downloads nothing and needs no accelerator, and hybrid retrieval still beats
either ranker alone once the query planner's structured filters are applied ahead of both (see
the ablation table in the README). A hashing embedder will not find a paraphrase that shares no
words or fragments with the query the way a trained model might; the query planner's own
synonym matching is what covers a fixed, known set of paraphrases (an NPT code's common
alternative names, section-size spellings) instead.
