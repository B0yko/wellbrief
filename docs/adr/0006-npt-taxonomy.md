# ADR 0006: NPT taxonomy

## Status

Accepted.

## Context

Non-productive time has to be classified into codes before it can be counted, ranked or compared
across wells, but drilling operators do not share one universal code list, and the tool has no
way to know a given site's own short codes without being told.

## Decision

Thirteen built-in codes, each carrying a human-readable label, an `avoidable` flag and a family
(`hole`, `equipment`, `well_construction`, `logistics`, `external`). This is described in the
product and its documentation as "a simplified taxonomy modelled on common industry practice",
never as an official or standard code list, because it is neither: it is a working classification
sized for this tool's own statistics and pattern detection, not a substitute for a site's real
NPT coding standard.

Weather, third-party standby and HSE stop-work are marked not avoidable: no engineering change on
the operator's side prevents them, so they are excluded from avoidable-NPT totals and from
becoming risks in a brief, and are instead summed into one background line. A site's own code
spelling maps onto this taxonomy through `[taxonomy.aliases]` in `wellbrief.toml`; an
unrecognised, unmapped code becomes `OTHER` rather than being dropped or raising an error, so a
code the taxonomy does not anticipate still shows up in the rollups instead of silently
disappearing.

## Consequences

A field can be analysed the moment its codes are mapped, without waiting on the tool to learn a
new code list, and the avoidable/not-avoidable split keeps a brief from recommending against
weather. The taxonomy's size and boundaries are a design choice, not a citation of a standard, and
a site whose own coding is much finer-grained will see some of that detail collapse into `OTHER`
until it is mapped.
