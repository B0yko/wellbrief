# ADR 0011: Risk statistics and pattern scopes

## Status

Accepted.

## Context

The offset-well risk brief ranks non-productive time (NPT) patterns found in
a field's own drilling history; nothing in it comes from a language model.
The formulas and thresholds below decide which patterns are real enough to
show a drilling engineer before spud, and how each one is priced. They are
written down once, in one place, so every number in the config has a stated
reason instead of looking arbitrary.

## Decision

### Pattern scopes

Two kinds of pattern are detected, independently, over the field's NPT
ledger:

- **Interval**: keyed on (NPT code, hole section, formation). A place in the
  well that keeps costing the same kind of time, regardless of which rig or
  tool drilled it.
- **Equipment**: keyed on (NPT code, rig) or (NPT code, MWD tool). A piece of
  kit that costs more time than the rest of the same field's fleet, wherever
  it is run.

### Exposure and per-affected-well statistics

- **Exposed wells** are the probability denominator: for an interval
  pattern, offset wells with at least one daily report in that hole section
  and formation; for an equipment pattern, wells of the field drilled on
  that rig, or run with that tool.
- **Affected wells** are the exposed wells with at least one qualifying NPT
  event of the pattern.
- `probability = wells_affected / wells_total` (the exposed count).
- **Mean and P90** are taken over each affected well's own hour *sum*, never
  over individual event hours: an affected well's total is what a drilling
  engineer actually carries, and averaging events instead would let one well
  with many short events outweigh one well with a single long one.
- **Expected hours** = `probability * mean_hours_per_affected_well`.
- **Expected cost** = `expected_hours / 24 * spread_rate_usd_per_day`.

### Interval lift

An interval candidate has to run hotter than the code's field-wide rate:

```
lift = (pattern_hours / drilling_days_in_that_section_and_formation)
       / (field_hours_for_the_code / field_drilling_days)
```

"Drilling day" is one daily report. A candidate qualifies only when
`wells_affected >= min_wells`, `wells_affected / wells_exposed >=
min_support`, and `lift >= min_lift`.

### Equipment rule

An equipment candidate has to clear three tests at once against the rest of
the same field's fleet:

- hours per drilling day on the category, divided by hours per drilling day
  on the rest of the fleet, at least `equipment_min_ratio`;
- share of the category's own wells affected at least `equipment_min_rate`;
- affected wells at least `equipment_min_wells` (so a two-well fleet cannot
  become a pattern by itself).

A share-of-wells-affected rule alone misses a category that fails harder,
not more often; the ratio test catches that case.

### Event ownership

Every NPT event counts toward at most one listed pattern. Two qualifying
equipment patterns of the same code are merged into the larger one first,
whenever they share more than half of the smaller one's events. An interval
candidate that shares more than half of its own events with a surviving
equipment pattern of the same code is folded into it entirely; at or below
that share, only the shared events move to the equipment pattern, and the
interval candidate is re-tested against the same thresholds on what
remains, kept only if it still qualifies on its own.

### Unavoidable background

Codes marked not avoidable in the taxonomy (weather, third-party standby,
HSE stop) never become risks. Their hours are summed field-wide and shown
as one background line instead.

### Applies-to-plan

An interval risk always applies to a planned well, since it does not depend
on rig or tool. An equipment risk applies when the plan names the same rig
or tool, does not apply when it names a different one, and is marked "plan
did not specify" when the plan gives neither a rig nor an MWD tool.

### Total-depth filter

A risk is dropped when its depth window starts more than 50 m below the
planned total depth, since the planned well will not reach that part of the
hole.

### Ranking and cap

Surviving risks are sorted by expected cost, descending, and capped at
`max_risks`.

## Threshold values

| Name | Value |
| --- | --- |
| `RISK_MIN_SUPPORT` | 0.3 |
| `RISK_MIN_WELLS` | 2 |
| `RISK_MIN_LIFT` | 2.0 |
| `EQUIPMENT_MIN_RATIO` | 2.0 |
| `EQUIPMENT_MIN_RATE` | 0.4 |
| `EQUIPMENT_MIN_WELLS` | 3 |
| `RISK_MAX_RISKS` | 8 |

Chosen on seed 20260731 only and frozen in this commit before seeds 7 and 42
were run.

## Classifier caveat

The mitigation miner's practice/failure/neutral classifier is surface-pattern
matching, not a model, and its phrase lists were written against the
synthetic corpus's own sentence templates. Its measured accuracy describes
how well it reads sentences shaped the way this corpus's generator writes
them, not how it would read a real field report's prose.

## Consequences

Every number the risk brief shows can be recomputed by hand from the NPT
ledger with the formulas above; nothing here depends on a language model or a
hidden constant. The thresholds only work in one direction (a pattern must
clear every one of its tests to be shown at all), so a field with too little
history gets a shorter or empty brief rather than a speculative one. The
values are not re-tuned against later data: a threshold that turns out to
need adjusting is a decision for a later revision of this document, made
deliberately and recorded as such, not a quiet edit.
