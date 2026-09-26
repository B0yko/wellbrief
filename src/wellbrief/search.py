"""Query planning and hybrid retrieval: structured filters first, then keyword and dense ranking.

A drilling engineer's question is half prose and half database predicate.
"What went wrong in the 12 1/4 inch section on Orrindale" is a text query with
two hard filters buried in it. Running that as pure text search returns
Vessra South reports that happen to mention 12 1/4", which answers a different
question. So the query planner reads the structure first (fields, wells, hole
sections, formations, NPT codes, depth and document type), the candidate set
is narrowed on the parsed fields, and only then do BM25 and the dense index
compete inside that set.

A filter that matches nothing leaves an empty candidate set; it never widens
back to the whole archive. A name the workspace does not know (a well, field,
hole section, formation or NPT code) is recorded on the plan as unmatched, so
the caller can abstain instead of answering a different question.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field as dc_field
from typing import Any

from .config import CODE_SYNONYMS, DEFAULT_TOP_K, DEPTH_BAND_M, RRF_K
from .embed import Embedder
from .models import Document, NptEvent, SearchHit
from .quotes import evidence_quote
from .store import Store, chunk_document_id
from .text import canonical_section, normalise, tokenize
from .workspace import FieldIndex

# ---------------------------------------------------------------------------
# Vocabulary of the planner
# ---------------------------------------------------------------------------


# Between the words of a phrase: any run of spaces, hyphens or underscores, or none.
_WORD_GAP = r"(?:\s|[_-])*"


def _phrase_regex(phrase: str) -> str:
    """A phrase on word boundaries, any spaces, hyphens or underscores between words, optional plural."""
    words = [w for w in re.split(r"[\s_-]+", phrase.strip().lower()) if w]
    return r"\b" + _WORD_GAP.join(re.escape(w) for w in words) + r"(?:s|es)?\b"


def _synonym_table(synonyms: dict[str, list[str]]) -> list[tuple[str, re.Pattern[str]]]:
    """(code, pattern) pairs, the phrases with the most words (then the longest) first."""
    entries = {(code, phrase.lower()) for code, phrases in synonyms.items()
               for phrase in [*phrases, code]}
    ordered = sorted(entries, key=lambda e: (-len(re.split(r"[\s_-]+", e[1])), -len(e[1]), e[1], e[0]))
    return [(code, re.compile(_phrase_regex(phrase), re.I)) for code, phrase in ordered]


_SYNONYMS = _synonym_table(CODE_SYNONYMS)

_DOC_TYPE_CUES: dict[str, re.Pattern[str]] = {
    "eowr": re.compile(r"\blessons?\b|\brecommend\w*|\bend[\s-]+of[\s-]+well\b|\beowrs?\b", re.I),
    "incident": re.compile(r"\bincidents?\b|\broot[\s-]+causes?\b|\binvestigations?\b", re.I),
    "ddr": re.compile(r"\bdaily[\s-]+(?:drilling[\s-]+)?reports?\b|\bddrs?\b", re.I),
}

# A question about non-productive time in general ("What NPT was booked on
# ORD-104?") is answered from the NPT ledger even when it names no code.
_NPT_CUE = re.compile(r"\bnpt\b|\bnon[\s-]*productive\b|\bdown[\s-]*time\b|\blost[\s-]+time\b"
                      r"|\btime\s+(?:was\s+|were\s+)?lost\b", re.I)

# A well id, in any case. See `_read_wells` for which ids name a well.
_WELL_ID = re.compile(r"\b([A-Za-z]{2,5})-(\d{2,4})\b")

# A size in inches: 12 1/4", 12-1/4 in, 12 1/4 inch, 12.25", 8.5 in, 26", or
# a fraction or decimal with no unit (12 1/4, 12.25). `_read_sections` decides
# which of them name a hole section.
_SIZE = re.compile(
    r"(?<![\w./])(?:(?P<whole>\d{1,2})(?:\s+|\s*-\s*)?(?P<num>\d)/(?P<den>\d)|(?P<size>\d{1,2}(?:\.\d{1,3})?))"
    r"(?:\s*-?\s*(?P<unit>\"|''|inch(?:es)?\b|in\b))?", re.I)
# "... section", "... hole", "... open hole" right after the size.
_SECTION_WORD_AFTER = re.compile(r"\s*(?:open\s+)?(?:hole|section)s?\b", re.I)
# "section 6\"", "hole size 8 1/2\"", "Hole section: 12 1/4\"" right before it.
_SECTION_WORD_BEFORE = re.compile(r"\b(?:hole\s+)?(?:section|size)\s*:?\s*$", re.I)
# A casing, liner, shoe or drill-string size: a tubular word right after the
# size or after up to two describing words ("9 5/8\" production casing", "7\"
# liner", "5\" drill pipe"). A casing point belongs to the hole section it
# ends, "stuck pipe" is not a pipe size, and "12 1/4\" with casing wear" is
# still a hole.
_TUBULAR_AFTER = re.compile(
    r"(?:\s+(?!(?:stuck|with|and|or|in|on|at|of|for|to|from|while|before|after|below|above|the|a|an)\b)"
    r"[\w-]+){0,2}?\s+(?:casing(?!\s+point)|liners?|shoes?|(?:drill\s*)?pipe|dp|hwdp"
    r"|(?:drill\s+)?collars?|tubing|conductors?|tie-?backs?|csg)\b", re.I)
_CLAUSE_END = re.compile(r"\s*(?:$|[,.;:!?)])")

# "around 2,650 m", "near 2650 m", "at 2650m"; only "at" makes it a filter.
_DEPTH = re.compile(
    r"(?:(?P<qual>\b(?:around|about|near|approximately|approx|roughly|circa|at)\b\.?|~)\s*)?"
    r"(?<![\w.,/-])(?P<num>\d{1,3}(?:,\d{3})+|\d{3,5})\s*(?:m|metres?|meters?)\b", re.I)
_MAX_DEPTH_M = 15_000

# A capitalised word of a name ("Orrindale", "Keldra's"), and a name of up to
# three words ("Vessra South"). An apostrophe or hyphen only joins letters, so
# a closing quote is never part of a name.
_NAME_WORD = r"[A-Z][a-z]\w*(?:['-][A-Za-z]\w*)*"
_NAME = rf"{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}}"
_QUOTED = r"[\"'](?P<name>[A-Za-z][^\"']{0,38}[A-Za-z0-9])[\"']"
# "field" as a word of its own: not "fields", not "field-wide".
_FIELD_WORD = r"(?i:\bfield\b)(?!-)"

# (pattern, fewest words the name needs once leading ordinary words are
# dropped, whether a name the workspace does not hold must end its clause to be
# read as a name: "for field Zelmar" names a field, "the field Engineer
# report" does not).
_Phrase = tuple[re.Pattern[str], int, bool]
_FIELD_PHRASES: list[_Phrase] = [
    (re.compile(rf"(?P<name>{_NAME})\s+{_FIELD_WORD}"), 1, False),
    (re.compile(rf"{_FIELD_WORD}\s*:?\s*(?P<name>{_NAME})"), 1, True),
    (re.compile(rf"{_QUOTED}\s+{_FIELD_WORD}"), 1, False),
    (re.compile(rf"{_FIELD_WORD}\s*:?\s*{_QUOTED}"), 1, False),
]
# A formation is named the way formations usually are, a capitalised name and
# a rock word ("Keldra Salt", "<Name> Dolomite"), or with the word formation.
_ROCK = (r"(?:Dolomite|Salt|Shale|Sand|Sandstone|Carbonate|Limestone|Marl|Claystone|Mudstone|Siltstone"
         r"|Chalk|Anhydrite|Evaporites?|Overburden)")
_FORMATION_PHRASES: list[_Phrase] = [
    (re.compile(rf"\b(?P<name>{_NAME_WORD}(?:\s+{_NAME_WORD})?\s+{_ROCK})\b"), 2, False),
    (re.compile(rf"(?P<name>{_NAME})\s+(?i:formation)\b"), 1, False),
    (re.compile(rf"(?i:\bformation\b)\s*:?\s*(?P<name>{_NAME})"), 1, True),
    (re.compile(rf"{_QUOTED}\s+(?i:formation)\b"), 1, False),
]
# What may follow a name that ends its clause.
_NAME_END = re.compile(
    r"\s*(?:$|[,.;:!?)\"']|(?:wells?|fields?|area|in|on|at|for|with|and|or|from|during|since|between|to|of"
    r"|by|over|across|while|when|where|before|after|below|above)\b)", re.I)
# Leading words that make a capitalised phrase an ordinary one ("The field",
# "Oil field", "Show Salt"); they are dropped before a name is looked up.
_NAME_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "each", "every", "whole", "entire", "same",
    "other", "another", "any", "all", "one", "our", "their", "its", "oil", "gas", "offshore", "onshore",
    "mature", "new", "old", "which", "what", "where", "when", "why", "who", "how", "show", "list", "give",
    "find", "tell", "is", "are", "was", "were", "did", "does", "do", "in", "on", "at", "of", "for", "from",
    "by", "across", "into", "total", "upper", "lower", "middle", "top", "base",
})
# Words that describe a rock rather than name it ("Reactive Shale", "Mobile
# Salt"); leading ones are dropped from a formation phrase, as is any leading
# word of five letters or more ending in -ing, -ive, -ed or -ly ("Drilling Salt").
_DESCRIPTIVE = frozenset({
    "mobile", "plastic", "soft", "hard", "tight", "weak", "loose", "porous", "vuggy", "unstable", "massive",
    "thick", "thin", "shallow", "deep", "sticky", "brittle", "swelling", "reactive", "depleted", "fractured",
    "unconsolidated", "overpressured", "overlying", "underlying",
})
# Capitalised words that are not names: days, months, job titles, report and
# test words. An unknown phrase made only of these (and of NPT code synonyms,
# "Wellbore Instability") is not read as a field or formation name.
_ORDINARY_WORDS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "today", "yesterday", "morning", "evening", "night", "day", "days", "daily",
    "week", "weeks", "month", "months", "year", "years",
    "engineer", "engineers", "engineering", "supervisor", "superintendent", "manager", "geologist",
    "geologists", "driller", "drillers", "toolpusher", "representative", "rep", "company", "man", "crew",
    "team", "staff", "personnel", "office", "hand", "hands", "service", "services", "operator",
    "report", "reports", "reporting", "summary", "data", "notes", "log", "logs", "history", "review",
    "study", "survey", "results", "plan", "planning", "program", "programme", "operations", "development",
    "appraisal", "exploration", "campaign", "average", "wide",
    "pressure", "pressures", "test", "tests", "testing", "integrity", "strength", "temperature",
    "gradient", "depth", "tops", "evaluation", "damage", "water", "fluid", "fluids",
})


def _scan_codes(text: str) -> tuple[list[str], str]:
    """The NPT codes a text names, in order of first appearance, and the text with every match blanked.

    Phrases match on word boundaries, longer phrases first; a matched span is
    blanked so a shorter phrase inside it cannot match again.
    """
    work = normalise(text).lower()
    first: dict[str, int] = {}
    for code, rx in _SYNONYMS:
        spans = [m.span() for m in rx.finditer(work)]
        if not spans:
            continue
        first[code] = min(first.get(code, spans[0][0]), spans[0][0])
        for start, end in spans:
            work = work[:start] + " " * (end - start) + work[end:]
    return sorted(first, key=lambda c: (first[c], c)), work


def match_codes(text: str) -> list[str]:
    """NPT codes a text names through their synonyms (word boundaries), in order of first appearance."""
    return _scan_codes(text)[0]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unmatched:
    """A name in the question that the workspace does not know."""

    kind: str       # field | well | section | formation | code
    value: str


def _listed(kind: str, values: list[str]) -> str:
    return f"{kind}{'s' if len(values) > 1 else ''} {', '.join(values)}"


@dataclass
class QueryPlan:
    """What the system thinks the question is actually asking for.

    Every named field, well, hole section, formation and code is kept; a filter
    on several values matches any of them.
    """

    text: str
    fields: list[str] = dc_field(default_factory=list)
    wells: list[str] = dc_field(default_factory=list)
    doc_types: list[str] = dc_field(default_factory=list)
    hole_sections: list[str] = dc_field(default_factory=list)
    formations: list[str] = dc_field(default_factory=list)
    codes: list[str] = dc_field(default_factory=list)
    depth_m: float | None = None
    depth_exact: bool = False       # "at 2,650 m": the depth band filters instead of ranking
    npt: bool = False               # the question asks about non-productive time in general
    unmatched: list[Unmatched] = dc_field(default_factory=list)

    @property
    def depth_band(self) -> tuple[float, float] | None:
        if self.depth_m is None:
            return None
        return max(0.0, self.depth_m - DEPTH_BAND_M), self.depth_m + DEPTH_BAND_M

    @property
    def scoped(self) -> bool:
        """Whether the question names a hole section, formation or code, which only the ledger can match."""
        return bool(self.hole_sections or self.formations or self.codes)

    @property
    def uses_ledger(self) -> bool:
        """Whether the candidate documents are chosen through the NPT ledger."""
        return self.scoped or self.depth_exact or self.npt

    @property
    def countable(self) -> bool:
        """Whether the question scopes something the NPT ledger can total."""
        return bool(self.fields or self.wells or self.uses_ledger)

    def document_filters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.fields:
            out["field_name"] = list(self.fields)
        if self.wells:
            out["well"] = list(self.wells)
        if self.doc_types:
            out["doc_type"] = list(self.doc_types)
        return out

    def ledger_filters(self, depth: bool = True) -> dict[str, Any]:
        """Filters for the NPT ledger; the depth band only when the question says "at"."""
        out: dict[str, Any] = {}
        for key, values in (("field_name", self.fields), ("well", self.wells), ("code", self.codes),
                            ("hole_section", self.hole_sections), ("formation", self.formations)):
            if values:
                out[key] = list(values)
        band = self.depth_band
        if depth and self.depth_exact and band is not None:
            out["depth_min"], out["depth_max"] = band
        return out

    def filters_text(self, types: bool = True) -> str:
        """The structured filters in words, for the abstention line (document types only if `types`)."""
        bits: list[str] = []
        for kind, values in (("field", self.fields), ("well", self.wells), ("section", self.hole_sections),
                             ("formation", self.formations), ("code", self.codes)):
            if values:
                bits.append(_listed(kind, values))
        if types and self.doc_types:
            bits.append(f"type {', '.join(self.doc_types)}")
        band = self.depth_band
        if self.depth_exact and band is not None:
            bits.append(f"depth {band[0]:,.0f}-{band[1]:,.0f} m")
        return ", ".join(bits) or "no filter"

    def describe(self) -> str:
        bits = [f"{key}={','.join(values)}" for key, values in (
            ("field", self.fields), ("wells", self.wells), ("section", self.hole_sections),
            ("formation", self.formations), ("codes", self.codes), ("types", self.doc_types)) if values]
        if self.depth_m is not None:
            bits.append(f"depth{'=' if self.depth_exact else '~'}{self.depth_m:.0f}m")
        if self.npt:
            bits.append("npt")
        if self.unmatched:
            bits.append("unmatched=" + ",".join(f"{u.kind}:{u.value}" for u in self.unmatched))
        return " ".join(bits) or "no structural filter"

    def to_dict(self) -> dict[str, Any]:
        band = self.depth_band
        return {
            "fields": list(self.fields),
            "wells": list(self.wells),
            "hole_sections": list(self.hole_sections),
            "formations": list(self.formations),
            "codes": list(self.codes),
            "doc_types": list(self.doc_types),
            "depth_m": self.depth_m,
            "depth_band_m": list(band) if band is not None else None,
            "depth_mode": None if self.depth_m is None else ("filter" if self.depth_exact else "prefer"),
            "npt": self.npt,
            "unmatched": [{"kind": u.kind, "value": u.value} for u in self.unmatched],
            "summary": self.describe(),
        }


# ---------------------------------------------------------------------------
# Reading a question
# ---------------------------------------------------------------------------


def _prepare(question: str) -> str:
    """Normalise fractions and dashes, and fold typographic quotes and inch marks."""
    text = normalise(question)
    for mark in ("\u201c", "\u201d", "\u2033"):
        text = text.replace(mark, '"')
    for mark in ("\u2018", "\u2032"):
        text = text.replace(mark, "'")
    return text


def _descriptive(word: str) -> bool:
    low = word.lower()
    return low in _DESCRIPTIVE or (len(low) >= 5 and low.endswith(("ing", "ive", "ed", "ly")))


def _clean_name(name: str, descriptive: bool = False) -> str:
    """The name without quotes, a possessive 's, or leading ordinary (and, if asked, descriptive) words."""
    words = [w for w in (re.sub(r"'s$", "", w, flags=re.I).strip("'\"") for w in name.split()) if w]
    while words and (words[0].lower() in _NAME_STOPWORDS or (descriptive and _descriptive(words[0]))):
        words.pop(0)
    return " ".join(words)


def _ordinary(name: str) -> bool:
    """Whether a capitalised phrase is made only of ordinary words and NPT code synonyms."""
    return all(w.strip("'") in _ORDINARY_WORDS for w in _scan_codes(name)[1].split())


def _resolve(name: str, known: Iterable[str]) -> str | None:
    """The known name `name` refers to: the same name in any case, a known name followed by
    other words ("Vessra South Mud" read after "field"), or the one known name it abbreviates
    ("Keldra formation" for Keldra Salt)."""
    by_low = {k.lower(): k for k in known}
    low = " ".join(name.lower().split())
    if low in by_low:
        return by_low[low]
    for k in sorted(by_low, key=len, reverse=True):
        if low.startswith(k + " "):
            return by_low[k]
    starts = [v for k, v in by_low.items() if k.startswith(low + " ")]
    return starts[0] if len(starts) == 1 else None


def _known_spans(text: str, names: Iterable[str]) -> list[tuple[int, int, str]]:
    """Case-insensitive, word-bounded occurrences of known names, longest names first."""
    spans: list[tuple[int, int, str]] = []
    for name in sorted(names, key=lambda n: (-len(n), n)):
        rx = re.compile(r"\b" + r"\s+".join(re.escape(w) for w in name.split()) + r"\b", re.I)
        for m in rx.finditer(text):
            if not any(s < m.end() and m.start() < e for s, e, _ in spans):
                spans.append((m.start(), m.end(), name))
    return sorted(spans)


def _named(text: str, phrases: list[_Phrase], known: set[str], taken: list[tuple[int, int, str]],
           descriptive: bool = False) -> tuple[list[tuple[int, str]], list[str]]:
    """Names introduced by an explicit phrase ("<Name> field", "<Name> Dolomite").

    Returns the (position, known name) pairs and the names the workspace does
    not know. Phrases overlapping an occurrence of a known name are left to
    that name; a phrase of ordinary words is not a name.
    """
    found: list[tuple[int, str]] = []
    unknown: list[str] = []
    for rx, min_words, must_end in phrases:
        for m in rx.finditer(text):
            start, end = m.span("name")
            if any(s < end and start < e for s, e, _ in taken):
                continue
            name = _clean_name(m.group("name"), descriptive)
            if len(name.split()) < min_words:
                continue
            resolved = _resolve(name, known)
            if resolved is not None:
                found.append((start, resolved))
            elif _ordinary(name) or (must_end and not _NAME_END.match(text, end)):
                continue
            elif name.lower() not in {u.lower() for u in unknown}:
                unknown.append(name)
    return found, unknown


def _in_order(found: Iterable[tuple[int, str]]) -> list[str]:
    out: list[str] = []
    for _, name in sorted(found):
        if name not in out:
            out.append(name)
    return out


def _read_fields(text: str, fields: set[str], formations: set[str], unmatched: list[Unmatched]) -> list[str]:
    """Known field names anywhere, and names phrased as a field ("<Name> field", "field <Name>").

    A field name inside a formation name ("Orrindale Sand") is not a field, and
    "the Keldra Salt field interval" names a formation.
    """
    formation_spans = _known_spans(text, formations)
    spans = [(s, e, n) for s, e, n in _known_spans(text, fields)
             if not any(fs <= s and e <= fe for fs, fe, _ in formation_spans)]
    named, unknown = _named(text, _FIELD_PHRASES, fields, spans + formation_spans)
    unmatched.extend(Unmatched("field", n) for n in unknown if _resolve(n, formations) is None)
    return _in_order([(s, n) for s, _, n in spans] + named)


def _read_wells(text: str, wells: set[str], unmatched: list[Unmatched]) -> list[str]:
    """Well ids in any case, upper-cased.

    An id the workspace does not hold names a well (and is unmatched) when its
    prefix is one of the workspace's well prefixes ("ord-150") or it is written
    in capitals ("XYZ-101"). Otherwise it is running text ("top-10", "Covid-19")
    and is ignored.
    """
    prefixes = {w.split("-", 1)[0] for w in wells}
    out: list[str] = []
    for m in _WELL_ID.finditer(text):
        prefix, number = m.group(1), m.group(2)
        well = f"{prefix.upper()}-{number}"
        if well in out:
            continue
        if well in wells:
            out.append(well)
        elif prefix.upper() in prefixes or prefix.isupper():
            out.append(well)
            unmatched.append(Unmatched("well", well))
    return out


def _read_sections(text: str, sections: set[str], unmatched: list[Unmatched]) -> list[str]:
    """Hole sections in any common spelling, canonicalised, in the order the question names them.

    A size names a hole section when it is one the workspace holds, or when
    "section" or "hole" goes with it ("the 6\" section", "hole size 6\"").
    Only a size named that explicitly is unmatched when the workspace does
    not hold it, so a casing, liner or drill-pipe size ("9 5/8\" casing",
    "5\" drill pipe") does not make the answer abstain; a size followed by a
    casing, liner, shoe or drill-string word is not a section unless it is
    named explicitly. Without a unit, only a fraction or decimal followed by
    "section" or "hole" counts ("12 1/4 section"). After a whole number, a
    bare "in" is read as inches only before "section", "hole" or the end of
    a clause, so "2 in 10 wells" is not a size.
    """
    out: list[str] = []
    for m in _SIZE.finditer(text):
        fraction = m.group("whole") is not None
        exact = fraction or "." in (m.group("size") or "")
        unit = (m.group("unit") or "").lower()
        after, before = text[m.end():], text[:m.start()]
        explicit_after = bool(_SECTION_WORD_AFTER.match(after))
        explicit = explicit_after or bool(_SECTION_WORD_BEFORE.search(before))
        if not unit and not (exact and explicit_after):
            continue
        if unit == "in" and not exact and not (explicit_after or _CLAUSE_END.match(after)):
            continue
        if not explicit and _TUBULAR_AFTER.match(after):
            continue
        size = f'{m.group("whole")} {m.group("num")}/{m.group("den")}"' if fraction else f'{m.group("size")}"'
        section = canonical_section(size)
        if section not in sections:
            if not explicit:
                continue
            if Unmatched("section", section) not in unmatched:
                unmatched.append(Unmatched("section", section))
        if section not in out:
            out.append(section)
    return out


def _read_formations(text: str, formations: set[str], unmatched: list[Unmatched]) -> list[str]:
    spans = _known_spans(text, formations)
    named, unknown = _named(text, _FORMATION_PHRASES, formations, spans, descriptive=True)
    unmatched.extend(Unmatched("formation", n) for n in unknown)
    return _in_order([(s, n) for s, _, n in spans] + named)


def _read_depth(text: str) -> tuple[float | None, bool]:
    for m in _DEPTH.finditer(text):
        depth = float(m.group("num").replace(",", ""))
        if 0 < depth <= _MAX_DEPTH_M:
            return depth, (m.group("qual") or "").lower() == "at"
    return None, False


def plan_query(question: str, store: Store) -> QueryPlan:
    """Read the structured filters out of a natural-language question.

    Names are checked against what the workspace holds; any that it does not
    hold are listed in `unmatched`.
    """
    text = _prepare(question)
    catalog = store.catalog()
    plan = QueryPlan(text=question)
    plan.fields = _read_fields(text, catalog["fields"], catalog["formations"], plan.unmatched)
    plan.wells = _read_wells(text, catalog["wells"], plan.unmatched)
    plan.hole_sections = _read_sections(text, catalog["sections"], plan.unmatched)
    plan.formations = _read_formations(text, catalog["formations"], plan.unmatched)
    plan.codes = match_codes(text)
    plan.unmatched.extend(Unmatched("code", c) for c in plan.codes if c not in catalog["codes"])
    plan.doc_types = [t for t, rx in _DOC_TYPE_CUES.items() if rx.search(text)]
    plan.depth_m, plan.depth_exact = _read_depth(text)
    plan.npt = bool(_NPT_CUE.search(text))
    return plan


# ---------------------------------------------------------------------------
# Structured filters
# ---------------------------------------------------------------------------


def _in_band(depth: Any, band: tuple[float, float]) -> bool:
    return depth is not None and band[0] <= float(depth) <= band[1]


def _incident_matches(doc: Document, plan: QueryPlan) -> bool:
    """An incident report is kept when its classification, section, formation and depth fit the plan."""
    meta = doc.meta or {}
    if plan.codes and meta.get("code") not in plan.codes:
        return False
    for key, wanted in (("hole_section", plan.hole_sections), ("formation", plan.formations)):
        if wanted and meta.get(key) and meta[key] not in wanted:
            return False
    band = plan.depth_band
    if plan.depth_exact and band is not None and meta.get("depth_m") is not None:
        return _in_band(meta["depth_m"], band)
    return True


def _ledger_candidates(plan: QueryPlan, store: Store) -> set[str]:
    """The reports with a matching ledger entry, plus the end-of-well reports of
    those wells and their incident reports that fit the plan."""
    events = store.npt(**plan.ledger_filters())
    matched = {e.doc_id for e in events}
    wells = sorted({e.well for e in events})
    if wells:
        matched |= store.document_ids(doc_type="eowr", well=wells)
        matched |= {d.doc_id for d in store.documents(doc_type="incident", well=wells)
                    if _incident_matches(d, plan)}
    return matched


def _drilled_through(plan: QueryPlan, store: Store) -> set[str]:
    """For an "at" depth: the daily reports whose drilled interval meets the depth band,
    and the incident reports inside it."""
    band = plan.depth_band
    if band is None:
        return set()
    scope = {k: v for k, v in plan.document_filters().items() if k != "doc_type"}
    out = store.documents_drilled_through(band[0], band[1], **scope)
    out |= {d.doc_id for d in store.documents(doc_type="incident", **scope)
            if _in_band((d.meta or {}).get("depth_m"), band)}
    return out


def candidate_documents(plan: QueryPlan, store: Store) -> set[str] | None:
    """The documents a question may retrieve, chosen on parsed fields only (None: no filter).

    Field, wells and document type filter the documents table. Section,
    formation, code, an "at" depth, and a general question about NPT go
    through the NPT ledger: the daily reports with a matching entry, plus the
    end-of-well reports of those wells and their incident reports that fit
    the plan (that is where the lessons and root causes are written).

    A section, formation or code that the ledger does not match leaves an
    empty set. The general NPT cue and an "at" depth name nothing that could
    be missing, so when the ledger has no entry for them the question keeps
    the documents its other filters allow: for an "at" depth, the daily
    reports that drilled through the depth band (and incident reports in it);
    for the NPT cue, the documents of the named fields, wells and types. A
    well with reports but no NPT therefore gets an answer (no NPT recorded)
    instead of an abstention.
    """
    doc_filters = plan.document_filters()
    allowed = store.document_ids(**doc_filters) if doc_filters else None
    if not plan.uses_ledger:
        return allowed
    matched = _ledger_candidates(plan, store)
    if not matched and not plan.scoped:
        if not plan.depth_exact:
            return allowed
        matched = _drilled_through(plan, store)
    return matched if allowed is None else allowed & matched


def _depth_ranking(plan: QueryPlan, store: Store, allowed: set[str] | None) -> list[tuple[str, float]]:
    """Reports with an NPT entry (or an incident) in the plan's depth band, nearest first."""
    band = plan.depth_band
    if band is None or plan.depth_m is None:
        return []
    target = plan.depth_m
    nearest: dict[str, float] = {}
    for e in store.npt(**plan.ledger_filters(depth=False), depth_min=band[0], depth_max=band[1]):
        nearest[e.doc_id] = min(nearest.get(e.doc_id, abs(e.depth_m - target)), abs(e.depth_m - target))
    for doc in store.documents(doc_type="incident", **{k: v for k, v in plan.document_filters().items()
                                                       if k != "doc_type"}):
        depth = (doc.meta or {}).get("depth_m")
        if depth is not None and band[0] <= float(depth) <= band[1] and _incident_matches(doc, plan):
            nearest[doc.doc_id] = min(nearest.get(doc.doc_id, abs(depth - target)), abs(depth - target))
    ranked = sorted(nearest.items(), key=lambda kv: (kv[1], kv[0]))
    return [(doc_id, -dist) for doc_id, dist in ranked if allowed is None or doc_id in allowed]


def rrf_fuse(rankings: dict[str, list[tuple[str, float]]],
             k: int = RRF_K) -> list[tuple[str, float, dict[str, float]]]:
    """Reciprocal rank fusion. Scale free, so BM25, cosine and depth distance can be mixed."""
    fused: dict[str, float] = {}
    parts: dict[str, dict[str, float]] = {}
    for source, ranked in rankings.items():
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            contrib = 1.0 / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + contrib
            parts.setdefault(doc_id, {})[source] = round(score, 6)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc_id, round(score, 6), parts.get(doc_id, {})) for doc_id, score in out]


def expanded_query(question: str, plan: QueryPlan) -> str:
    """The question plus what the planner resolved, in the words the reports use.

    A synonym becomes its code ("pack-off" also searches STUCK_PIPE), a hole
    size its canonical spelling (`12.25"` also searches `12 1/4"`), and a
    lower-case well id its upper-case form.
    """
    return " ".join([question, *plan.codes, *plan.wells, *plan.hole_sections])


def quote_terms(question: str, plan: QueryPlan) -> list[str]:
    """The terms an evidence quote is scored on: the expanded query, its formations, and
    "npt" for a question about non-productive time in general (the reports write NPT)."""
    terms = tokenize(" ".join([expanded_query(question, plan), *plan.formations]))
    return [*terms, "npt"] if plan.npt else terms


def quoted_entries(plan: QueryPlan, store: Store, doc_ids: Iterable[str]) -> dict[str, list[NptEvent]]:
    """Per report, the ledger entries the plan's NPT filters select, in report order.

    These are the entries a retrieved daily report's evidence quote is taken
    from (see `quotes.evidence_quote`): the entries of the plan's codes, in its
    section, formation and "at" depth band when it names them, or every entry
    of the report for the general NPT cue. A plan that does not choose
    documents through the ledger (one that names only fields, wells or
    document types) selects no entry, so its hits are quoted by their best
    operations or remarks line.
    """
    ids = sorted(set(doc_ids))
    if not plan.uses_ledger or not ids:
        return {}
    out: dict[str, list[NptEvent]] = {}
    for e in store.npt(**plan.ledger_filters(), doc_id=ids):
        out.setdefault(e.doc_id, []).append(e)
    return out


def _reduce_best_chunk(ranking: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """The document ranking a chunk ranking implies: each document keeps its single
    best-ranked chunk's score. `ranking` is sorted best first, so the first chunk seen
    for a document is that document's best (only the best chunk per document counts,
    before top-k selection)."""
    best: dict[str, float] = {}
    order: list[str] = []
    for chunk_id, score in ranking:
        doc_id = chunk_document_id(chunk_id)
        if doc_id not in best:
            best[doc_id] = score
            order.append(doc_id)
    return [(d, best[d]) for d in order]


class Searcher:
    """Hybrid retrieval over one workspace's per-field chunk indexes.

    A question naming a field (`plan.fields`) is answered from that field's
    index alone; otherwise every field's ranking is computed independently
    (bm25, dense and depth fused within the field) and the per-field document
    rankings are then themselves fused with RRF, so one field's score scale
    never drowns out another's.
    """

    def __init__(self, store: Store, field_indexes: Mapping[str, FieldIndex], embedder: Embedder):
        self.store = store
        self.field_indexes = dict(field_indexes)
        self.embedder = embedder

    def _field_ranking(self, field_name: str, query: str, qvec: list[float] | None,
                       allowed: set[str] | None, depth: list[tuple[str, float]],
                       pool: int) -> list[tuple[str, float, dict[str, float]]]:
        index = self.field_indexes.get(field_name)
        if index is None:
            return []
        allowed_chunks: set[str] | None = None
        if allowed is not None:
            allowed_chunks = {c for c in index.bm25.doc_ids if chunk_document_id(c) in allowed}
            if not allowed_chunks:
                return []
        rankings: dict[str, list[tuple[str, float]]] = {
            "bm25": _reduce_best_chunk(index.bm25.search(query, top_k=pool, allowed=allowed_chunks)),
        }
        try:
            rankings["dense"] = (
                _reduce_best_chunk(index.vectors.search(qvec, top_k=pool, allowed=allowed_chunks))
                if qvec is not None else []
            )
        except Exception:  # noqa: BLE001 - the dense side is optional
            rankings["dense"] = []
        field_docs = index.doc_ids
        field_depth = [(d, s) for d, s in depth if d in field_docs][:pool]
        if field_depth:
            rankings["depth"] = field_depth
        return rrf_fuse(rankings)[:pool]

    def search(self, question: str, top_k: int = DEFAULT_TOP_K,
               plan: QueryPlan | None = None) -> tuple[list[SearchHit], QueryPlan]:
        """Rank the candidate documents; no hits when the plan names something unknown
        or its filters match no document."""
        plan = plan or plan_query(question, self.store)
        if plan.unmatched:
            return [], plan
        allowed = candidate_documents(plan, self.store)
        if allowed is not None and not allowed:
            return [], plan

        target_fields = plan.fields or sorted(self.field_indexes)
        pool = max(top_k * 5, 40)
        query = expanded_query(question, plan)
        try:
            qvec = self.embedder.embed(query)
        except Exception:  # noqa: BLE001 - the dense side is optional
            qvec = None
        # A depth in the question is a preference: reports near it get a third vote.
        depth = _depth_ranking(plan, self.store, allowed)

        per_field = {f: r for f in target_fields
                    if (r := self._field_ranking(f, query, qvec, allowed, depth, pool))}
        if len(per_field) <= 1:
            fused = next(iter(per_field.values()), [])[:top_k]
        else:
            cross = {f"field:{name}": [(d, s) for d, s, _ in ranked] for name, ranked in per_field.items()}
            fused = rrf_fuse(cross)[:top_k]

        terms = quote_terms(question, plan)
        entries = quoted_entries(plan, self.store, (doc_id for doc_id, _, _ in fused))
        hits: list[SearchHit] = []
        for doc_id, score, parts in fused:
            doc = self.store.get_document(doc_id)
            if not doc:
                continue
            quote = evidence_quote(doc, terms, entries.get(doc_id, ()), plan.depth_m)
            hits.append(SearchHit(doc_id=doc_id, score=score, document=doc, snippet=quote, components=parts))
        return hits, plan
