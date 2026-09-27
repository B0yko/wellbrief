"""Verifier: does a narrator's text trace back to its evidence?

Three checks, run against any narrator's output (`ask`'s or `brief`'s):

(a) every `[DOC-ID]` cited in the text names a document the evidence carries;
(b) every formatted number in the text -- after removing citation brackets,
    document ids, well ids and dates -- is found, at the precision the text
    itself shows, among the numbers the evidence pack, the computed figures
    or the quoted source text carry, or among the numbers the question
    itself already stated; hole sizes (`12 1/4"`, `12.25"`, `12-1/4 in`,
    `12 1/4 in`) are never checked, since they describe the plan, not the
    archive, and a digit run fused directly to a preceding letter (`P90`, a
    percentile label, not the measurement 90) is never read as a number;
(c) no well id, and (when the caller names the field universe) no field
    name, outside the evidence appears.

An empty or whitespace-only reply fails outright, rather than passing every
check vacuously for having nothing to check.

Number normalisation handles the formatted shapes the narrators write:
`$1.49 M` (millions, compared at the precision its own decimals give, here
the nearest $10,000), `29 %` (checked both as a share, 0.29, and as a point
value, 29, since a computed figure may be stored either way) and `744.3 h`
(checked at 0.1 h). A plain number with thousands separators (`1,402`) is
checked at its own decimal precision. A document id (`DDR-ORD-101-009`,
`EOWR-ORD-101`) is stripped whole, even outside a citation bracket, so its
own digits are never mistaken for a measurement.

`OfflineNarrator` always passes every check, because everything it writes
comes straight from the evidence pack; `LlmNarrator` (`narrate.llm`) uses a
failing `VerificationResult` to fall back to it, with a banner. The
fault-injection helpers at the end corrupt an already-verified text in one of
`FAULT_KINDS` and are used by the `verifier-faults` eval suite (and this
module's own tests) to measure the catch rate.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import FIGURE_SOURCE_LIMIT
from .base import extract_cited_ids

if TYPE_CHECKING:
    from ..models import Answer, RiskBrief
    from ..store import Store

__all__ = [
    "FAULT_KINDS",
    "Evidence",
    "VerificationResult",
    "evidence_from_answer",
    "evidence_from_brief",
    "evidence_from_pack",
    "inject_fault",
    "verify",
]

_CITE_BRACKET_RE = re.compile(r"\[[A-Za-z0-9_\-]+\]")
_WELL_RE = re.compile(r"\b[A-Z]{2,5}-\d{2,4}\b")
# A full document id (`DDR-ORD-101-009`, `EOWR-ORD-101`): a type prefix, a well id, and an
# optional sequence number. Stripped whole before number extraction so a plain-prose mention
# (one without citation brackets) never leaves an orphan digit group -- the trailing "-009" of
# "DDR-ORD-101-009" -- behind for `_TOKEN_RE` to mistake for a measurement.
_DOC_ID_RE = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,5}-\d{2,4}(?:-\d{2,4})?\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_HOLE_SIZE_RE = re.compile(
    r'\d+(?:\.\d+)?(?:[\s-]\d+/\d+)?\s*(?:"|in\.?\b|¼|½|¾|⅛|⅜|⅝|⅞)', re.IGNORECASE
)
# The `(?<![A-Za-z0-9])` guard keeps a number token from starting right after a letter or a
# digit with no separator. A letter immediately before rules out a percentile label like "P90"
# (never read as the measurement 90: the text still checks "P90 37.7 h" against the evidence
# for 37.7, just not for a bare 90); ruling out a digit immediately before as well stops the
# same "90" from being re-tried, and matched, one character in as a bare "0" once the first
# attempt (at the "9") is turned away by the letter it followed.
_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<dollar>\$)?\s?(?P<num>\d[\d,]*(?:\.\d+)?)(?:\s?(?P<unit>M\b|%|h\b))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of `verify`: `ok` and, when it is False, why."""

    ok: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class Evidence:
    """Everything a narrator's text is allowed to name or state: document ids, well
    and field names, and every number drawn from the evidence pack, the computed
    figures and the quoted source text. A well name is evidence when it is one of
    the cited documents' own wells, when a figures breakdown (`by_well`) counted it,
    or when it is mentioned inside a quoted source text (a genuine end-of-well
    report cross-referencing another well by name is not the narrator inventing
    one)."""

    doc_ids: frozenset[str]
    wells: frozenset[str]
    fields: frozenset[str]
    numbers: tuple[float, ...]


# ---------------------------------------------------------------------------
# Building an Evidence from what `qa.ask` and `riskbrief.build_brief` have
# ---------------------------------------------------------------------------


def _numbers_from_json(value: Any) -> list[float]:
    """Every numeric leaf of a JSON-shaped structure (a figures dict, `RiskBrief.to_dict()`), plus
    the length of every list found in it: a report that says "5 of 20 reports" or "3 mitigations"
    is stating the size of a list the evidence already carries, not a new fact."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        out: list[float] = []
        for v in value.values():
            out.extend(_numbers_from_json(v))
        return out
    if isinstance(value, (list, tuple)):
        out = [float(len(value))]
        for v in value:
            out.extend(_numbers_from_json(v))
        return out
    return []


def _strings_by_key(value: Any, key: str) -> set[str]:
    """Every string found anywhere under a dict key named `key` in a JSON-shaped structure: a
    figures breakdown (`by_well`, `by_field`) names a well or field the SQL aggregation counted
    even when no single cited report is that well's, so those names are evidence too, not a
    narrator's invention."""
    out: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key and isinstance(v, str) and v:
                out.add(v)
            else:
                out |= _strings_by_key(v, key)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out |= _strings_by_key(v, key)
    return out


def _wells_in_text(text: str | None) -> set[str]:
    """Well ids mentioned inside a piece of quoted source text: a genuine end-of-well
    report legitimately cross-references another well ("follows the recommendation
    in the ORD-105 report"), and repeating that verbatim is not inventing a name."""
    return set(_WELL_RE.findall(text)) if text else set()


def _numbers_from_text(text: str | None) -> list[float]:
    """Every number a piece of quoted or narrated text states, in every candidate form."""
    if not text:
        return []
    out: list[float] = []
    for m in _TOKEN_RE.finditer(_strip_for_numbers(text)):
        out.extend(value for value, _step in _token_candidates(m))
    return out


def evidence_from_pack(pack: list[dict[str, Any]], summary: dict[str, Any]) -> Evidence:
    """The evidence an `ask` answer may draw on: the pack `qa.ask` builds (each entry
    carries `doc_id`, `well`, `field` and `quote`) and the summary passed to
    `Narrator.answer` alongside it (`figures`, `figure_sources`, `report_count` and
    `mitigations`, walked whole so a per-report figure or a mitigation's own numbers
    count as evidence too)."""
    doc_ids = {e["doc_id"] for e in pack if e.get("doc_id")}
    wells = {e["well"] for e in pack if e.get("well")} | _strings_by_key(summary, "well")
    fields = {e["field"] for e in pack if e.get("field")} | _strings_by_key(summary, "field")
    numbers = _numbers_from_json(summary)
    for e in pack:
        numbers.extend(_numbers_from_text(e.get("quote")))
        wells |= _wells_in_text(e.get("quote"))
    for m in summary.get("mitigations") or []:
        numbers.extend(_numbers_from_text(m.get("text")))
        wells |= _wells_in_text(m.get("text"))
    return Evidence(frozenset(doc_ids), frozenset(wells), frozenset(fields), tuple(numbers))


def evidence_from_brief(brief: RiskBrief) -> Evidence:
    """The evidence a risk brief's narrative may draw on: its own field, its offset
    wells, every citation's document, and every number the brief itself carries
    (thresholds and spread rate are provenance, not narrated content, and are left
    out; everything under `risks` and the totals is included). The narrative states
    the total of `unavoidable_hours` (a sum across its codes), so that sum is added
    explicitly: `_numbers_from_json` walks a dict's individual values, not a total
    the narrator itself computes over them."""
    doc_ids: set[str] = set()
    wells = set(brief.generated_from_wells)
    numbers = _numbers_from_json({
        "risks": [r.to_dict() for r in brief.risks],
        "total_expected_npt_hours": brief.total_expected_npt_hours,
        "total_exposure_usd": brief.total_exposure_usd,
        "unavoidable_hours": brief.unavoidable_hours,
        "planned_td_m": brief.planned_td_m,
        "spread_rate_usd_per_day": brief.spread_rate_usd_per_day,
    })
    if brief.unavoidable_hours:
        numbers.append(float(sum(brief.unavoidable_hours.values())))
    for risk in brief.risks:
        for cite in risk.citations:
            doc_ids.add(cite.doc_id)
            numbers.extend(_numbers_from_text(cite.quote))
            wells |= _wells_in_text(cite.quote)
    return Evidence(frozenset(doc_ids), frozenset(wells), frozenset({brief.field_name}), tuple(numbers))


def evidence_from_answer(answer: Answer, store: Store) -> Evidence:
    """The same evidence as `evidence_from_pack`, reconstructed from the public
    `Answer` a caller (a test, or the `verifier-faults` eval harness) holds instead
    of the private pack `qa.ask` builds it from.

    One thing the public contract does not carry exactly: `ask` only cites the
    largest `FIGURE_SOURCE_LIMIT` contributing reports, but `Answer.figure_sources`
    lists every one of them, so this adds that cited count explicitly rather than
    reconstructing it from a shape only `qa.ask` itself sees.
    """
    figures_summary: dict[str, Any] = {"figures": answer.figures, "figure_sources": answer.figure_sources,
                                       "mitigations": answer.mitigations}
    doc_ids = {c.doc_id for c in answer.citations}
    wells = {c.well for c in answer.citations if c.well} | _strings_by_key(figures_summary, "well")
    fields: set[str] = set(_strings_by_key(figures_summary, "field"))
    for doc_id in doc_ids:
        doc = store.get_document(doc_id)
        if doc is not None and doc.field_name:
            fields.add(doc.field_name)
    numbers = _numbers_from_json(figures_summary)
    numbers.append(float(min(FIGURE_SOURCE_LIMIT, len(answer.figure_sources))))
    for c in answer.citations:
        numbers.extend(_numbers_from_text(c.quote))
        wells |= _wells_in_text(c.quote)
    for m in answer.mitigations:
        numbers.extend(_numbers_from_text(m.get("text")))
        wells |= _wells_in_text(m.get("text"))
    return Evidence(frozenset(doc_ids), frozenset(wells), frozenset(fields), tuple(numbers))


# ---------------------------------------------------------------------------
# Number normalisation
# ---------------------------------------------------------------------------


def _strip_for_numbers(text: str) -> str:
    text = _CITE_BRACKET_RE.sub(" ", text)
    text = _DATE_RE.sub(" ", text)
    text = _DOC_ID_RE.sub(" ", text)
    text = _WELL_RE.sub(" ", text)
    text = _HOLE_SIZE_RE.sub(" ", text)
    return text


def _token_candidates(match: re.Match[str]) -> list[tuple[float, float]]:
    """The value(s) a formatted-number token may mean, each with the rounding step
    its own display precision implies. A percentage yields both a share (0.29) and
    a point value (29), since a computed figure may be stored either way."""
    num = match.group("num")
    if not num:
        return []
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return []
    decimals = len(num.split(".", 1)[1]) if "." in num else 0
    step = 10.0 ** (-decimals)
    unit = (match.group("unit") or "").lower()
    if match.group("dollar") and unit == "m":
        return [(value * 1_000_000.0, step * 1_000_000.0)]
    if unit == "%":
        return [(value, step), (value / 100.0, step / 100.0)]
    return [(value, step)]


def _all_candidates(text: str) -> list[tuple[float, float]]:
    return [c for m in _TOKEN_RE.finditer(_strip_for_numbers(text)) for c in _token_candidates(m)]


def _matches(value: float, step: float, pool: Iterable[float]) -> bool:
    step = step if step > 0 else 1.0
    target = round(value / step)
    return any(round(v / step) == target for v in pool)


# ---------------------------------------------------------------------------
# The three checks
# ---------------------------------------------------------------------------


def verify(text: str, evidence: Evidence, question: str = "",
          known_field_names: frozenset[str] = frozenset()) -> VerificationResult:
    """Check `text` (a narrator's answer or risk-brief narrative) against `evidence`.

    `question` exempts a number the question itself already stated (a hole size or
    a depth the engineer typed in is not a hallucination). `known_field_names`, when
    given, is the full set of field names the workspace holds; without it check (c)
    still catches a well id outside the evidence, but not a field name.

    An empty or whitespace-only `text` fails outright: it cites nothing, states no
    number and names no well, so the three checks below would otherwise pass it
    vacuously. That reply carries no answer, and a caller (`LlmNarrator`) must fall
    back to the deterministic offline text rather than surface it as verified.
    """
    if not text or not text.strip():
        return VerificationResult(ok=False, reasons=["empty reply"])

    reasons: list[str] = []

    for doc_id in sorted(extract_cited_ids(text) - evidence.doc_ids):
        reasons.append(f"cited a document not in the evidence: {doc_id}")

    question_values = [v for v, _ in _all_candidates(question)]
    for m in _TOKEN_RE.finditer(_strip_for_numbers(text)):
        candidates = _token_candidates(m)
        if not candidates:
            continue
        if any(_matches(v, s, evidence.numbers) for v, s in candidates):
            continue
        if any(_matches(v, s, question_values) for v, s in candidates):
            continue
        reasons.append(f"number not traceable to the evidence or the question: {m.group(0).strip()!r}")

    without_citations = _CITE_BRACKET_RE.sub(" ", text)
    for well in sorted(set(_WELL_RE.findall(without_citations))):
        if well not in evidence.wells:
            reasons.append(f"names a well not in the evidence: {well}")
    for name in sorted(known_field_names):
        if name not in evidence.fields and re.search(rf"\b{re.escape(name)}\b", text):
            reasons.append(f"names a field not in the evidence: {name}")

    return VerificationResult(ok=not reasons, reasons=reasons)


# ---------------------------------------------------------------------------
# Fault injection: the "verifier catch rate" harness
# ---------------------------------------------------------------------------

#: The four fault kinds the README's "verifier catch rate" metric measures, each
#: expected to be caught 100 % of the time.
FAULT_KINDS = ("quote_char", "id_outside_pack", "untraceable_number", "foreign_well")


def inject_fault(text: str, evidence: Evidence, kind: str, *,
                 foreign_id: str = "ZZZ-999", foreign_well: str = "") -> str | None:
    """Corrupt an already-verified `text` in one of `FAULT_KINDS`.

    Returns the corrupted text, or None when `text` offers nothing this fault kind
    can corrupt (`quote_char` needs a number to alter; `foreign_well` needs a well
    name from another field, supplied by the caller since this module knows nothing
    about the corpus).

    Raises:
        ValueError: `kind` is not one of `FAULT_KINDS`.
    """
    if kind == "quote_char":
        return _flip_a_digit(text, evidence)
    if kind == "id_outside_pack":
        return f"{text.rstrip()} [{foreign_id}]"
    if kind == "untraceable_number":
        bogus = max((*evidence.numbers, 0.0)) + 12_345.67
        return f"{text.rstrip()} A further {bogus:.1f} h were also recorded."
    if kind == "foreign_well":
        if not foreign_well:
            return None
        return f"{text.rstrip()} This also affected {foreign_well}."
    raise ValueError(f"unknown fault kind: {kind!r}; expected one of {FAULT_KINDS}")


def _excluded_spans(text: str) -> list[tuple[int, int]]:
    """Spans of `text` that are a citation bracket, a well id, a date or a hole size --
    never the target of `_flip_a_digit`, since a well id or date is not a computed number
    and a hole size describes the plan, not the archive."""
    spans: list[tuple[int, int]] = []
    for pattern in (_CITE_BRACKET_RE, _DOC_ID_RE, _WELL_RE, _DATE_RE, _HOLE_SIZE_RE):
        spans.extend(m.span() for m in pattern.finditer(text))
    return spans


def _overlaps(span: tuple[int, int], excluded: list[tuple[int, int]]) -> bool:
    return any(span[0] < e[1] and span[1] > e[0] for e in excluded)


def _flip_a_digit(text: str, evidence: Evidence) -> str | None:
    """Change one digit of a formatted number `text` contains outside a citation, well
    id, date or hole size, so it no longer traces to the evidence (check b).

    Tries the last digit of each candidate number first (the smallest possible
    change) and moves on when a change happens to still round-match some other
    evidence number (a coincidence, most likely for a short, round count); returns
    None only when the text has no number this can corrupt.
    """
    excluded = _excluded_spans(text)
    for m in _TOKEN_RE.finditer(text):
        num = m.group("num")
        num_span = m.span("num")
        if not num or _overlaps(num_span, excluded):
            continue
        for i in range(len(num) - 1, -1, -1):
            if not num[i].isdigit():
                continue
            bumped = str((int(num[i]) + 1) % 10)
            new_num = num[:i] + bumped + num[i + 1:]
            candidate = text[:num_span[0]] + new_num + text[num_span[0] + len(num):]
            new_match = _TOKEN_RE.match(candidate, m.start())
            if new_match is None:
                continue
            candidates = _token_candidates(new_match)
            if candidates and not any(_matches(v, s, evidence.numbers) for v, s in candidates):
                return candidate
    return None
