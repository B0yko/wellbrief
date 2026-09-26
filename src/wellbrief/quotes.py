"""Evidence quotes: which passage of a document a citation shows, copied verbatim.

A quote is a span of the document text exactly as it was ingested, never the
`normalise()`d copy the parsers read (that copy folds unicode fractions,
dashes and apostrophes, so it is not always in the source word for word).
Runs of whitespace inside the span are shown as one space, so a value that
wraps over several lines reads as one line. The citation checks compare
quotes the same whitespace-insensitive way, so every quote built here is
found in its source.

Which passage is quoted:

- A daily report with NPT entries that the question selects (the caller
  passes them in): the description of the best of those entries, its
  continuation lines joined. The nearest entry to a depth the question gives
  wins, then the one sharing the most terms with the question, then the one
  with the most hours, then the first.
- Otherwise the passage that shares the most terms with the question, from
  the content sections of the document type: the operations and remarks
  lines of a daily report, the lessons and recommendations of an end-of-well
  report, the sequence of events, root cause and corrective actions of an
  incident report. A question word that asks for a kind of passage ("what
  caused", "recommend") counts as a shared term of that kind and decides a
  tie in its favour; any other tie goes to the earlier passage.
- A document without such a passage falls back to its other content lines,
  and only a document with no content line at all to a label line; a
  section heading is quoted only when nothing else is left. The
  synthetic-document footer, and anything after it, is never quoted.

Label lines. Outside the content sections a line shaped `Label: value` (a
label of up to six words, then a colon: `Hole section : 17 1/2"`,
`Operator: ...    Well: ...`) is a label line. Inside them a line is
content even when it reads `Topic: text` ("Keldra Salt: hold at least
1.42 sg ..."); only the planning notes of the daily report template listed in
`CONTENT_LABELS` (`Next 24 h: ...`) count as label lines there, and a
numbered, bulleted or timed item never does.

A passage is one logical line: a line with the more indented lines that
continue it; a numbered or bulleted item without its marker, or an
operations line with its time range, together with every following line up
to the next item or blank line (as the parsers read items). A paragraph whose
lines all share one indentation (prose wrapped without a hanging indent) is
split into sentences, and also at a line break that is not a wrap: a line
without final punctuation that the next line's first word would still have
fitted on (the wrap width being the document's longest line, and at least
60 characters), followed by a line that starts with a capital or a digit.
"""

from __future__ import annotations

import math
import re
import unicodedata
from array import array
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from .config import SYNTHETIC_FOOTER
from .models import Document, NptEvent
from .text import normalise, tokenize

__all__ = [
    "CONTENT_LABELS",
    "CONTENT_SECTIONS",
    "MAX_QUOTE_CHARS",
    "NPT_DETAIL_HEADING",
    "Passage",
    "best_passage",
    "evidence_quote",
    "passages",
    "quote_page",
    "raw_span",
    "verbatim_quote",
]

# The sections whose lines say what happened, per document type, keyed by
# heading (numbering, a trailing colon and a trailing parenthesis dropped,
# upper case). The value names the kind of passage.
CONTENT_SECTIONS: dict[str, dict[str, str]] = {
    "ddr": {"OPERATIONS SUMMARY": "operations", "REMARKS": "remarks"},
    "eowr": {"LESSONS LEARNED": "lesson", "RECOMMENDATIONS FOR FUTURE WELLS": "recommendation"},
    "incident": {"SEQUENCE OF EVENTS": "sequence", "ROOT CAUSE": "root_cause",
                 "CORRECTIVE ACTIONS": "corrective_action"},
}
NPT_DETAIL_HEADING = "NPT DETAIL"

# Labels that open a planning note rather than a record of what happened,
# per passage kind: inside a content section, a line that starts with one of
# them and a colon is a label line.
CONTENT_LABELS: dict[str, tuple[str, ...]] = {
    "remarks": ("Casing programme for this section", "Next 24 h"),
}

# A quote longer than this is cut at the last word boundary before it.
MAX_QUOTE_CHARS = 400

# Words a question uses to ask for a kind of passage ("what caused ...",
# "what do the reports recommend ..."); a passage of that kind counts them as
# its own terms. Stored in the folded form `_fold` produces.
_KIND_CUES: dict[str, frozenset[str]] = {
    "lesson": frozenset({"lesson", "learned", "learn"}),
    "recommendation": frozenset({"recommend", "recommended", "recommendation"}),
    "sequence": frozenset({"sequence", "happened", "happen"}),
    "root_cause": frozenset({"root", "cause", "caused"}),
    "corrective_action": frozenset({"corrective", "action", "prevent", "mitigation", "mitigate"}),
}

# Words that name a document or where it sits rather than what happened: a
# question about "the end of well reports" does not favour a lesson that
# mentions "daily reports". Left out of the scoring on both sides.
_NOT_CONTENT = frozenset({"report", "daily", "end", "well", "eowr", "ddr", "document", "field"})

# "1. ", "2) ", "- ", "* ", a bullet: an item marker, not part of the quote.
_ITEM_MARKER = re.compile(r"(?:\d{1,3}[.)]|[-*\u2022])[ \t]+")
# "06:00-14:12": an operations line starts here and keeps its time range.
_TIME_RANGE = re.compile(r"\d{1,2}:\d{2}[ \t]*-[ \t]*\d{1,2}:\d{2}(?!\d)")
# "Hole section : 17 1/2"", "Operator: Quillfen Energy": a label of at most
# six words at the start of the line, then a colon.
_LABEL_CHAR = r"[A-Za-z0-9/()%&.,'#+-]"
_LABEL_LINE = re.compile(rf"[A-Za-z]{_LABEL_CHAR}*(?:[ \t]+{_LABEL_CHAR}+){{0,5}}[ \t]*:(?:\s|$)")
# The end of a sentence: . ! or ? (and any closing quote or bracket) before
# whitespace and a capital letter, a digit or an opening quote or bracket.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s+[A-Z0-9\"'(\[])")
# Abbreviations whose period does not end a sentence ("approx. 20 bbl").
_ABBREVIATIONS = frozenset({"approx", "ca", "e.g", "i.e", "incl", "est", "vs", "no", "nr"})
# A line ending in one of these goes on (or ends a sentence the splitter finds).
_OPEN_END = re.compile(r"[.!?,;:(&/\-\u2013\u2014]$")
# The narrowest wrap width assumed: a document whose lines are all shorter
# (a short table, a few remarks) is taken as not wrapped at all.
_MIN_WRAP_WIDTH = 60


@dataclass(frozen=True)
class Passage:
    """One quotable passage: its span in the raw text and where it sits."""

    start: int              # character positions in the raw text
    end: int
    text: str               # the span, whitespace runs shown as one space
    section: str            # heading key of the section it sits in ("" before the first heading)
    kind: str               # its content kind for the document type, "" outside the content sections
    role: str               # content | label | heading


@dataclass(frozen=True)
class _Line:
    start: int              # position of the first non-blank character
    end: int                # position after the last non-blank character
    indent: int
    text: str               # the line without its indentation and trailing blanks
    section: str
    heading: bool


# ---------------------------------------------------------------------------
# Verbatim spans
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _folded(text: str) -> tuple[str, array[int], array[int]]:
    """`text` as the parsers read it (normalise() and whitespace runs as one space), and for
    each character of that form the raw span [start, end) it came from.

    Cached: the miner quotes several sentences of one report in a row.
    """
    chars: list[str] = []
    starts: array[int] = array("q")
    ends: array[int] = array("q")
    plain = text.isascii()          # normalise() leaves ASCII text unchanged
    space = True                    # drop leading whitespace
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        if not plain:
            # A base character with its combining marks is normalised as one unit.
            while j < n and unicodedata.combining(text[j]):
                j += 1
        for ch in (text[i] if plain else normalise(text[i:j])):
            if ch.isspace():
                if space:
                    continue
                ch, space = " ", True
            else:
                space = False
            chars.append(ch)
            starts.append(i)
            ends.append(j)
        i = j
    if chars and chars[-1] == " ":
        chars.pop()
        starts.pop()
        ends.pop()
    return "".join(chars), starts, ends


def raw_span(text: str, needle: str, start: int = 0) -> tuple[int, int] | None:
    """Where `needle` occurs in `text` at or after position `start`, as raw positions.

    Both sides are compared the way the parsers read them, so a sentence the
    parser returned (normalised, continuation lines joined) is found in the
    raw document even where the source spells `12¼"` or wraps the line.
    """
    target = _folded(needle)[0]
    if not target:
        return None
    folded, starts, ends = _folded(text)
    k = folded.find(target, bisect_left(starts, start))
    if k < 0:
        return None
    return starts[k], ends[k + len(target) - 1]


def _clip(quote: str, limit: int = MAX_QUOTE_CHARS) -> str:
    """A long quote cut at the last word boundary before `limit` (a prefix is still verbatim)."""
    if len(quote) <= limit:
        return quote
    cut = quote.rfind(" ", 0, limit + 1)
    return quote[:cut] if cut > limit // 2 else quote[:limit]


def _span_text(text: str, start: int, end: int) -> str:
    return " ".join(text[start:end].split())


def verbatim_quote(text: str, needle: str, start: int = 0) -> str | None:
    """The raw span of `text` that `needle` was read from, whitespace runs as one space;
    None when `needle` is not in `text` (at or after position `start`)."""
    span = raw_span(text, needle, start)
    return _clip(_span_text(text, *span)) if span else None


# ---------------------------------------------------------------------------
# Layout: sections, logical lines, sentences
# ---------------------------------------------------------------------------


def _heading_key(line: str) -> str | None:
    """The key of a section heading: a line at the left margin in capitals
    ("NPT DETAIL", "4. LESSONS LEARNED", "OPERATIONS SUMMARY (24 h)", "REMARKS:").

    A line in capitals that reads as a sentence (it ends with . ! or ?, or
    holds a comma or a semicolon) is content, not a heading.
    """
    if not line or line[0].isspace():
        return None
    head = re.sub(r"^\d{1,2}\.[ \t]*", "", line.strip())
    head = re.sub(r"[ \t]*:$", "", head)
    head = re.sub(r"[ \t]*\([^)]*\)$", "", head)
    letters = [c for c in head if c.isalpha()]
    if len(letters) < 3 or any(c.islower() for c in letters) or re.search(r"[:,;]|[.!?]$", head):
        return None
    return " ".join(head.split()).upper()


def _layout(text: str) -> list[_Line]:
    """The non-blank lines before the footer, each with the heading of its section.

    A blank line is kept as a `_Line` with empty text, so paragraphs stay apart.
    """
    lines: list[_Line] = []
    section = ""
    pos = 0
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        at = pos
        pos += len(raw) + 1
        stripped = line.strip()
        if stripped == SYNTHETIC_FOOTER:
            break
        if not stripped:
            lines.append(_Line(at, at, 0, "", section, False))
            continue
        indent = len(line) - len(line.lstrip())
        key = _heading_key(line)
        if key is not None:
            section = key
        lines.append(_Line(at + indent, at + len(line.rstrip()), indent, stripped, section, key is not None))
    return lines


def _starts_item(line: _Line) -> bool:
    return bool(_ITEM_MARKER.match(line.text) or _TIME_RANGE.match(line.text))


def _is_label(text: str) -> bool:
    return bool(_LABEL_LINE.match(text))


def _is_content_label(text: str, kind: str) -> bool:
    """Whether a line of a content section of this kind opens with one of its `CONTENT_LABELS`."""
    low = " ".join(text.split()).lower()
    return any(re.match(re.escape(label.lower()) + r"[ \t]*:", low) for label in CONTENT_LABELS.get(kind, ()))


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split the raw span [start, end) into sentences, as raw spans."""
    out: list[tuple[int, int]] = []
    segment = text[start:end]
    begin = 0
    for m in _SENTENCE_END.finditer(segment):
        before = segment[begin:m.start()].split()
        if segment[m.start()] == "." and before and before[-1].lstrip("(\"'").lower() in _ABBREVIATIONS:
            continue
        out.append((start + begin, start + m.end()))
        rest = segment[m.end():]
        begin = m.end() + len(rest) - len(rest.lstrip())
    out.append((start + begin, end))
    return [(s, e) for s, e in out if e > s]


def _hard_break(line: _Line, nxt: _Line, width: int) -> bool:
    """Whether the break between two lines of one indentation ends a statement although
    `line` has no final punctuation: the next line starts with a capital or a digit and
    its first word would have fitted on `line`, so the break is not a wrap at `width`."""
    if _OPEN_END.search(line.text) or not (nxt.text[0].isupper() or nxt.text[0].isdigit()):
        return False
    return line.indent + len(line.text) + 1 + len(nxt.text.split()[0]) <= width


def _block_units(text: str, block: list[_Line], kind: str, width: int) -> list[tuple[int, int, bool]]:
    """The passages of one paragraph as (start, end, is a label line).

    `kind` is the content kind of the paragraph's section ("" outside the
    content sections) and `width` the wrap width: the longest line of the
    document, at least `_MIN_WRAP_WIDTH`.
    """
    def label(start: int, end: int, item: bool) -> bool:
        if item:
            return False
        line = _span_text(text, start, end)
        return _is_content_label(line, kind) if kind else _is_label(line)

    units: list[tuple[int, int, bool]] = []
    uniform = all(ln.indent == block[0].indent for ln in block) and not any(_starts_item(ln) for ln in block)
    if uniform:
        # Prose wrapped without a hanging indent: sentences, except that a
        # label line stays a unit of its own and a hard line break ends one.
        run: list[_Line] = []

        def flush() -> None:
            if run:
                units.extend((s, e, False) for s, e in _sentences(text, run[0].start, run[-1].end))
                run.clear()

        for i, ln in enumerate(block):
            if label(ln.start, ln.end, False):
                flush()
                units.append((ln.start, ln.end, True))
                continue
            run.append(ln)
            if i + 1 < len(block) and _hard_break(ln, block[i + 1], width):
                flush()
        flush()
        return units

    # Logical lines. An item marker or a time range opens a passage that runs
    # to the next one (or the end of the paragraph), as the parsers read
    # items; any other line opens one that its more indented lines continue.
    first_indent = -1
    item = False
    start = end = 0
    for ln in block:
        marker = _ITEM_MARKER.match(ln.text)
        opens = bool(marker or _TIME_RANGE.match(ln.text))
        if first_indent < 0 or opens or (not item and ln.indent <= first_indent):
            if first_indent >= 0:
                units.append((start, end, label(start, end, item)))
            first_indent, item = ln.indent, opens
            start = ln.start + (marker.end() if marker else 0)
        end = ln.end
    if first_indent >= 0:
        units.append((start, end, label(start, end, item)))
    return units


def passages(text: str, doc_type: str) -> list[Passage]:
    """Every quotable passage of a document, in document order (the footer and what follows excluded)."""
    content = CONTENT_SECTIONS.get(doc_type, {})
    layout = _layout(text)
    width = max((ln.indent + len(ln.text) for ln in layout), default=0)
    width = max(width, _MIN_WRAP_WIDTH)
    out: list[Passage] = []
    block: list[_Line] = []

    def flush() -> None:
        if not block:
            return
        section = block[0].section
        kind = content.get(section, "")
        for s, e, is_label in _block_units(text, block, kind, width):
            quote = _clip(_span_text(text, s, e))
            if quote:
                out.append(Passage(s, e, quote, section, kind, "label" if is_label else "content"))
        block.clear()

    for ln in layout:
        if ln.heading:
            flush()
            out.append(Passage(ln.start, ln.end, _clip(" ".join(ln.text.split())), ln.section, "", "heading"))
        elif not ln.text:
            flush()
        else:
            block.append(ln)
    flush()
    return out


def _section_start(text: str, key: str) -> int:
    """Position of the heading line of section `key`, or 0 when there is none."""
    return next((ln.start for ln in _layout(text) if ln.heading and ln.section == key), 0)


# ---------------------------------------------------------------------------
# Choosing the passage
# ---------------------------------------------------------------------------


def _fold(token: str) -> str:
    """A light plural fold, so "losses" meets "loss" and "failures" meets "failure"."""
    if not token.isalpha():
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _terms(tokens: Iterable[str]) -> frozenset[str]:
    """Tokens to score with: folded, without bare numbers (a depth, hole size or mud
    weight counts through its own domain token, a clock time never) and without
    the words that only name a document."""
    return frozenset(f for f in (_fold(t) for t in tokens if not t.isdigit()) if f not in _NOT_CONTENT)


def _overlap(wanted: frozenset[str], text: str) -> int:
    return len(wanted & _terms(tokenize(text)))


def _score(wanted: frozenset[str], p: Passage) -> tuple[int, bool]:
    """Terms shared with the question (a cue word for the passage's kind counts as one),
    then whether the question asked for this kind of passage."""
    cues = wanted & _KIND_CUES.get(p.kind, frozenset())
    return len(wanted & (_terms(tokenize(p.text)) | cues)), bool(cues)


def _best(candidates: list[Passage], wanted: frozenset[str]) -> Passage:
    """The candidate with the best `_score`; the earliest on a tie."""
    best, best_score = candidates[0], (-1, False)
    for p in candidates:
        score = _score(wanted, p)
        if score > best_score:
            best, best_score = p, score
    return best


def best_passage(text: str, doc_type: str, terms: Iterable[str]) -> str:
    """The quote for a document the question matched without an NPT entry of its own.

    `terms` are the question's search tokens (see `text.tokenize`). Content
    passages of the document type come first, then any other content line,
    then label lines, then headings; the footer never.
    """
    found = passages(text, doc_type)
    wanted = _terms(terms)
    tiers = (
        [p for p in found if p.kind and p.role == "content"],
        [p for p in found if not p.kind and p.role == "content"],
        [p for p in found if p.role == "label"],
        [p for p in found if p.role == "heading"],
    )
    for tier in tiers:
        if tier:
            return _best(tier, wanted).text
    return ""


def evidence_quote(doc: Document, terms: Iterable[str], entries: Sequence[NptEvent] = (),
                   depth_m: float | None = None) -> str:
    """The quote a citation of `doc` shows for a question.

    `entries` are the document's NPT ledger entries that the question
    selects; for a daily report the description of the best of them is
    quoted, as the raw span found from the NPT DETAIL heading on (from the
    start of the text when there is no such heading). Without such an entry,
    or when no description is found in the text, the best content passage is
    quoted instead (`best_passage`).
    """
    tokens = list(terms)
    wanted = _terms(tokens)
    if doc.doc_type == "ddr" and entries:
        def rank(item: tuple[int, NptEvent]) -> tuple[float, int, float, int]:
            i, e = item
            distance = 0.0
            if depth_m is not None:
                distance = abs(e.depth_m - depth_m) if e.depth_m else math.inf
            return distance, -_overlap(wanted, e.description), -e.hours, i

        start = _section_start(doc.text, NPT_DETAIL_HEADING)
        for _, entry in sorted(enumerate(entries), key=rank):
            quote = verbatim_quote(doc.text, entry.description, start) if entry.description else None
            if quote:
                return quote
    return best_passage(doc.text, doc.doc_type, tokens)


def quote_page(doc: Document, quote: str) -> int | None:
    """The 1-based page `quote` falls on, from `doc.page_map`; None when the
    document carries no page map (every document but a PDF, in this version)
    or the quote cannot be found in the raw text.

    `page_map` is a sorted list of the character offset in `doc.text` where
    each page ends (so a PDF reader that keeps page boundaries can build one
    without joining pages itself): the page of an offset is the position of
    the first boundary past it.
    """
    if not doc.page_map or not quote:
        return None
    span = raw_span(doc.text, quote)
    if span is None:
        return None
    offset = span[0]
    for page, end in enumerate(doc.page_map, start=1):
        if offset < end:
            return page
    return len(doc.page_map)
