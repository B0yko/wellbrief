"""Mitigation miner v2: one implementation, shared by `ask` and `brief`.

A "mitigation" is a sentence that says what to do differently, quoted
verbatim from a document that earns the claim. Two things have to be true
before a sentence qualifies:

1. **Where it comes from.** Only two sources ever count: the lessons and
   recommendations of a *clean* well's end of well report, or the corrective
   actions of an incident report filed under the same NPT code. "Clean" is
   defined by the caller, not by this module: `riskbrief.build_risk` passes
   the population a discovered `Pattern` already computed (offset wells
   exposed to the same section and formation, or on the field's other rig or
   tool, that never had the event); `qa.ask` computes the same population
   directly from the ledger when the question names enough to define it, and
   falls back to an unrestricted search otherwise (see `qa._mitigations_for`).
   A `MinerScope` with `clean_wells=None` is that unrestricted case: every
   well's end of well report is a candidate, not only ones known to have
   avoided anything.
2. **What it says.** The sentence has to read as a written practice, not a
   description of the failure itself or a throwaway remark: `classify_sentence`
   labels it `practice`, `failure` or `neutral` from surface wording alone,
   and only `practice` sentences ever become a mitigation. It also has to be
   about the risk in question: `[taxonomy.keywords]` names the words that
   put a sentence in the same subject as the code (mud pump, fluid end for a
   rig repair; MWD, temperature rating for a downhole tool failure), and an
   interval risk additionally requires the sentence, or the incident's own
   recorded section and formation, to place it at the right spot in the well.

The classifier is intentionally simple surface-pattern matching, not a model:
an imperative sentence ("Hold at least...", "Inspect and change...") or one
carrying an explicit success phrase ("held gauge", "only seepage", "no MWD
failures above...") is `practice`; failing that, a sentence naming the
failure itself ("packed off", "total losses were taken", "failed") is
`failure`; anything else is `neutral`. An explicit practice signal always
wins a sentence that also carries a failure word: the corrective action
"Place the jars so that the string can be backed off above the stuck point"
is a real example from this corpus, an instruction that happens to name the
very problem it prevents.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import TAXONOMY_KEYWORDS
from .models import Document
from .parse import parse_eowr, parse_incident
from .quotes import verbatim_quote
from .store import Store

__all__ = [
    "MITIGATION_SOURCE_EOWR",
    "MITIGATION_SOURCE_INCIDENT",
    "MinerScope",
    "Mitigation",
    "classify_many",
    "classify_sentence",
    "exposed_clean_wells",
    "mine_mitigations",
]


# ---------------------------------------------------------------------------
# Classifier: practice | failure | neutral
# ---------------------------------------------------------------------------

# A sentence that opens with one of these (no subject, base form) is a
# written instruction: every corrective action and pattern recommendation in
# the synthetic demonstration corpus, and presumably in a site's own reports, is phrased this way.
_IMPERATIVE_VERBS = frozenset({
    "run", "confirm", "hold", "condition", "inspect", "pump", "place", "keep",
    "weight", "record", "spot", "monitor", "replace", "log", "pressure",
    "plan", "call", "review", "raise", "reduce", "maintain", "install",
})
_FIRST_WORD = re.compile(r"^\s*([A-Za-z]+)\b")


def _phrase(words: str) -> re.Pattern[str]:
    """`words` on word boundaries, literal spacing (the classifier's phrases are short and fixed)."""
    parts = [re.escape(w) for w in words.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.I)


# Explicit success/prescriptive phrases, plus the two more the synthetic
# demonstration corpus's generator writes that are not spelled out verbatim in the list below:
# "spot"/"spotted" (an LCM pill is spotted, not just "spot"ted as a verb at
# the start) and "cut flow rate" (the lesson reads "Flow rate was cut to
# 430 gpm", not an imperative).
_PRACTICE_PHRASES = (
    "hold at least", "held gauge", "only seepage", "inspected and changed",
    "inspect and change", "rated to", "reduce flow", "cut to", "cut flow rate",
    "sweep", "spot", "spotted",
)
_PRACTICE_RX = tuple(_phrase(p) for p in _PRACTICE_PHRASES)
# "no MWD failures above 118 C", "no pump NPT": a negated failure word is a
# success claim, not a failure narrative.
_NO_FAILURES_RX = re.compile(r"\bno\b[^.;]{0,30}\bfailures?\b", re.I)
_NO_PUMP_NPT_RX = re.compile(r"\bno\s+pump\s+npt\b", re.I)

# Failure narrative phrases, plus "tight hole" and "leaked", the two failure
# templates in the synthetic demonstration corpus that describe what went
# wrong without using any of the phrases above (a salt-creep
# instability lesson with no pack-off, a leaking cement head seal).
_FAILURE_PHRASES = (
    "packed off", "total losses were taken", "failed", "washout",
    "required repeated", "stuck", "tight hole", "leaked",
)
_FAILURE_RX = tuple(_phrase(p) for p in _FAILURE_PHRASES)
_LOST_BBL_RX = re.compile(r"\blost\b[^.;]{0,20}\bbbl\b", re.I)


def classify_sentence(text: str) -> str:
    """`practice` | `failure` | `neutral`, from surface wording alone (see module docstring).

    Never reads a truth file or a label a generator or operator wrote: the
    classifier only ever sees the sentence text. An explicit
    practice signal -- one of the phrases above, or the imperative mood every
    written instruction in the synthetic demonstration corpus opens with -- always wins, even over a
    sentence that also names the failure it is instructing against; only
    when nothing marks a sentence as prescriptive does a failure word call it
    a failure narrative, and a sentence with neither is neutral.
    """
    folded = " ".join(text.split())
    practice = sum(1 for rx in _PRACTICE_RX if rx.search(folded))
    if _NO_FAILURES_RX.search(folded) or _NO_PUMP_NPT_RX.search(folded):
        practice += 1
    first = _FIRST_WORD.match(folded)
    if first is not None and first.group(1).lower() in _IMPERATIVE_VERBS:
        practice += 1
    if practice:
        return "practice"
    failure = sum(1 for rx in _FAILURE_RX if rx.search(folded))
    if _LOST_BBL_RX.search(folded):
        failure += 1
    return "failure" if failure else "neutral"


def classify_many(sentences: list[str]) -> list[str]:
    """`classify_sentence` over a batch (the evals adapter's `classify`)."""
    return [classify_sentence(s) for s in sentences]


# ---------------------------------------------------------------------------
# Relevance: the code's keywords, plus formation or section for an interval risk
# ---------------------------------------------------------------------------

_KeywordPatterns = dict[str, tuple[re.Pattern[str], ...]]


def _compile_keywords(keywords: Mapping[str, list[str]]) -> _KeywordPatterns:
    return {code: tuple(_phrase(k) for k in words) for code, words in keywords.items()}


#: The default, built from `TAXONOMY_KEYWORDS` (`config.py`); a site's own `[taxonomy.keywords]`
#: (`settings.TaxonomySettings.keywords`) is compiled fresh per call rather than cached here,
#: since it varies by workspace and this only ever runs a handful of times per `ask` or `brief`.
_DEFAULT_KEYWORD_PATTERNS: _KeywordPatterns = _compile_keywords(TAXONOMY_KEYWORDS)


def _keyword_hit(text: str, code: str, patterns: _KeywordPatterns) -> bool:
    return any(rx.search(text) for rx in patterns.get(code, ()))


def _keyword_score(text: str, code: str, patterns: _KeywordPatterns) -> int:
    return sum(1 for rx in patterns.get(code, ()) if rx.search(text))


def _names_formation_or_section(text: str, hole_section: str, formation: str) -> bool:
    low = text.lower()
    if formation and formation.lower() in low:
        return True
    if hole_section:
        digits = re.sub(r"[^0-9/]", "", hole_section)
        return bool(digits) and digits in re.sub(r"[^0-9/]", "", low)
    return False


def _relevant(text: str, code: str, hole_section: str, formation: str, patterns: _KeywordPatterns) -> bool:
    """The sentence's own text names the code's subject and, for an interval
    risk, the place in the well (for an equipment risk the keyword
    match alone is enough)."""
    if not _keyword_hit(text, code, patterns):
        return False
    if not (hole_section or formation):
        return True
    return _names_formation_or_section(text, hole_section, formation)


# ---------------------------------------------------------------------------
# Scope and candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinerScope:
    """One risk's identity, as the miner needs it: what to look for, where
    (an interval risk's hole section and formation; empty for an equipment
    risk, where the keyword match alone decides relevance), and, when known,
    the wells that avoided it.

    `clean_wells=None` means the caller could not determine who avoided the
    problem (a question that names no code, or a code with no known clean
    population); the miner then draws its end of well report candidates from
    every well in scope rather than a known-clean subset. Incident corrective
    actions are never restricted this way: a
    corrective action is a valid mitigation regardless of whether the well
    that filed the incident also avoided the problem elsewhere.
    """

    code: str
    field_name: str | list[str] | None = None
    hole_section: str = ""
    formation: str = ""
    clean_wells: frozenset[str] | None = None


#: `Mitigation.source`: which of the miner's two candidate pools a sentence came from -- an
#: end of well report's own lessons or recommendations (a clean well's, or an unrestricted
#: one when no clean population is known; see `MinerScope`), or a same-code incident report's
#: corrective actions (never restricted to clean wells: a corrective action is written by the
#: well that HAD the problem, and is a valid mitigation regardless). A caller that shows a
#: heading claiming the mitigation comes from "wells that avoided it" (`qa.ask`) must never
#: apply that claim to a `"incident"`-sourced entry.
MITIGATION_SOURCE_EOWR = "eowr"
MITIGATION_SOURCE_INCIDENT = "incident"


@dataclass(frozen=True)
class Mitigation:
    text: str
    doc_id: str
    well: str
    source: str = MITIGATION_SOURCE_EOWR

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "doc_id": self.doc_id, "well": self.well, "source": self.source}


@dataclass(frozen=True)
class _Candidate:
    score: int
    kind_rank: int          # lesson < recommendation < corrective_action, a tie-break only
    text: str
    doc_id: str
    well: str
    source: str


_KIND_RANK = {"lesson": 0, "recommendation": 1, "corrective_action": 2}
_KIND_SOURCE = {"lesson": MITIGATION_SOURCE_EOWR, "recommendation": MITIGATION_SOURCE_EOWR,
                "corrective_action": MITIGATION_SOURCE_INCIDENT}


def exposed_clean_wells(store: Store, field_name: str | None, code: str,
                        hole_section: str, formation: str) -> set[str]:
    """The wells `qa.ask` treats as having avoided `code` at this (section,
    formation): exposed to it (a daily report there) and no ledger event of
    the code there (computed directly for a
    query's scope rather than a discovered `analytics.Pattern`, which already
    carries its own `clean_wells`). Empty when neither dimension is given:
    there is then no population to define.
    """
    if not hole_section and not formation:
        return set()
    exposed = {
        d["well"] for d in store.ddr_features(field_name)
        if (not hole_section or d["hole_section"] == hole_section)
        and (not formation or d["formation"] == formation)
    }
    if not exposed:
        return set()
    had = {
        e.well for e in store.npt(field_name=field_name, code=code,
                                  hole_section=hole_section or None, formation=formation or None)
    }
    return exposed - had


def _eowr_meta(doc: Document) -> Mapping[str, Any]:
    """`doc.meta` when it already carries `parse.parse_eowr`'s output -- every document
    `ingest_folder` produced, parsed once at ingest time with that file's own
    `[parse.eowr.sections]` (a site's own `wellbrief.toml` or `ingest --config`, recorded per
    file). `"lessons"` is a key `parse_eowr` always sets (to `[]` when there is nothing to find),
    so its absence means this `Document` was built directly (a unit test's hand-written store
    row, not a real ingest), and is the one case this falls back to parsing `doc.text` with the
    canonical template's own default headings -- the only headings such a row ever uses.

    Re-parsing `doc.text` with the *current* caller's settings instead of trusting `doc.meta`
    would repeat the bug this module exists to avoid: a site whose section headings differ from
    whatever is asking the question would silently get no candidates.
    """
    if "lessons" in doc.meta:
        return doc.meta
    return parse_eowr(doc.text)


def _eowr_candidates(store: Store, scope: MinerScope,
                     patterns: _KeywordPatterns) -> list[_Candidate]:
    """Lessons and recommendations for each candidate end of well report (see `_eowr_meta`)."""
    out: list[_Candidate] = []
    for doc in store.documents(doc_type="eowr", field_name=scope.field_name):
        if scope.clean_wells is not None and doc.well not in scope.clean_wells:
            continue
        meta = _eowr_meta(doc)
        sentences = [("lesson", s) for s in meta.get("lessons", [])]
        sentences += [("recommendation", s) for s in meta.get("recommendations", [])]
        for kind, sentence in sentences:
            if not _relevant(sentence, scope.code, scope.hole_section, scope.formation, patterns):
                continue
            if classify_sentence(sentence) != "practice":
                continue
            quote = verbatim_quote(doc.text, sentence)
            if quote is None:
                continue
            out.append(_Candidate(_keyword_score(sentence, scope.code, patterns), _KIND_RANK[kind],
                                  quote, doc.doc_id, doc.well, _KIND_SOURCE[kind]))
    return out


def _incident_meta(doc: Document) -> Mapping[str, Any]:
    """`doc.meta` when it already carries `parse.parse_incident`'s output, else that parse run
    directly on `doc.text` with the canonical template's own default headings (see `_eowr_meta`;
    `"corrective_actions"` is the key `parse_incident` always sets)."""
    if "corrective_actions" in doc.meta:
        return doc.meta
    return parse_incident(doc.text)


def _incident_candidates(store: Store, scope: MinerScope,
                         patterns: _KeywordPatterns) -> list[_Candidate]:
    """Corrective actions for each candidate incident report (see `_incident_meta`): the code,
    section and formation matched below are the ones recorded at ingest time with that file's
    own settings, not values re-derived here from a possibly different current configuration."""
    out: list[_Candidate] = []
    for doc in store.documents(doc_type="incident", field_name=scope.field_name):
        meta = _incident_meta(doc)
        if meta.get("code") != scope.code:
            continue
        # An interval risk's incidents have to be the same section and
        # formation too (structural fields, not text: a corrective action
        # ("Run a caliper across the salt...") does not always name the
        # formation the way a lesson does).
        if scope.hole_section and meta.get("hole_section") != scope.hole_section:
            continue
        if scope.formation and meta.get("formation") != scope.formation:
            continue
        for action in meta.get("corrective_actions", []):
            if not _keyword_hit(action, scope.code, patterns):
                continue
            if classify_sentence(action) != "practice":
                continue
            quote = verbatim_quote(doc.text, action)
            if quote is None:
                continue
            out.append(_Candidate(_keyword_score(action, scope.code, patterns),
                                  _KIND_RANK["corrective_action"], quote, doc.doc_id, doc.well,
                                  MITIGATION_SOURCE_INCIDENT))
    return out


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _near_duplicate(a: str, b: str) -> bool:
    """Normalised token Jaccard >= 0.8, or the same first 90 normalised characters:
    the field-wide recommendation every well after the originator
    repeats verbatim collapses to the one instance the ranking prefers."""
    ta, tb = _tokens(a), _tokens(b)
    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.8:
        return True
    pa = re.sub(r"[^a-z0-9]", "", a.lower())[:90]
    pb = re.sub(r"[^a-z0-9]", "", b.lower())[:90]
    return bool(pa) and pa == pb


def mine_mitigations(store: Store, scope: MinerScope, limit: int = 3,
                     keywords: Mapping[str, list[str]] | None = None) -> list[Mitigation]:
    """The shared miner: practice sentences from `scope`'s
    candidate wells' end of well reports, plus same-code incident corrective
    actions, near-duplicates removed, at most `limit`, best first.

    `keywords` (`[taxonomy.keywords]`) defaults to the canonical template's own vocabulary; a
    caller with a resolved `settings.Settings` passes its `taxonomy.keywords` through instead.
    Candidate text and structured fields (lessons, recommendations, corrective actions, code,
    section, formation) come from each document's own `meta` whenever it has one, parsed once at
    ingest time with that file's own settings (`[parse.eowr.sections]` / `[parse.incident.sections]`,
    from an `ingest --config` file or the ingested folder's own `wellbrief.toml`) -- so this needs
    no section-heading configuration of its own, and is correct even when a document's own
    template differs from the current workspace `wellbrief.toml` (see `_eowr_meta`,
    `_incident_meta`). A `Document` built directly rather than through `ingest_folder` (no
    `meta`, a unit test's own store row) falls back to parsing `doc.text` with the canonical
    template's own default headings.
    """
    patterns = _compile_keywords(keywords) if keywords is not None else _DEFAULT_KEYWORD_PATTERNS
    candidates = (_eowr_candidates(store, scope, patterns)
                 + _incident_candidates(store, scope, patterns))
    candidates.sort(key=lambda c: (-c.score, c.kind_rank, c.doc_id))
    out: list[Mitigation] = []
    for c in candidates:
        if any(_near_duplicate(c.text, m.text) for m in out):
            continue
        out.append(Mitigation(c.text, c.doc_id, c.well, c.source))
        if len(out) >= limit:
            break
    return out
