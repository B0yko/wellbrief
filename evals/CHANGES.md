# Changes to the evaluation cases

The cases in `evals/cases/` were written before the fixes they measure. Once a
suite has had its first recorded run, no case is removed and no case is
loosened: a question, a relevance set, a gold query, a threshold or a target
stays as it was. A case the final run still fails keeps its expectation and
gets `gate = false` with a `reason`, and it is reported as a failure.

Every change to a case file after its first run is listed below with the date,
the case ids, what changed and why. A change that makes a case stricter, or
that corrects a gold query which contradicts the case's own comment, is listed
the same way.

## Log

No changes yet.
