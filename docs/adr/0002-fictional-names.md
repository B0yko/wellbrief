# ADR 0002: Fictional names

## Status

Accepted.

## Context

Every operator, field, well, rig, formation and downhole tool name in the demonstration corpus
has to be obviously fictional to anyone reading the repository, and it also has to not collide,
by accident, with a real operator or a real field somewhere. A generated report that happened to
reuse a real field's name would be confusing at best and could be read as a claim about a real
asset at worst.

## Decision

Every name used by the corpus generator (`Quillfen Energy`, `Orrindale`, `Vessra South`, their
rigs, MWD tools and formations, and the names introduced later by `examples/`) was checked
against public search results before it was used, specifically for a collision with a real
operator, field, rig contractor or downhole-tool vendor. The queries and results are not
published in this repository, so that a real entity a query happened to turn up is never itself
written down here.

One collision was found and acted on: an example CSV ledger's field name collided with a real,
currently producing gas field, and was renamed. Every other name was kept as originally chosen. Both fields' generated documents also end with a fixed footer line stating
that the operator, fields, wells, rigs and vendors are fictional.

## Consequences

A name collision is caught before a name is ever committed, not after. The check is a point-in-time
web search, not a trademark or legal clearance, and a name search can miss an obscure or very
recently registered real entity; the footer on every generated document, and this ADR, both state
plainly that everything the corpus names is invented. New names introduced later (a new example,
a new field in a demonstration) go through the same check before their first commit.
