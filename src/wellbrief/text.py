"""Tokenisation tuned for drilling paperwork.

A generic tokeniser destroys the parts of a daily report that carry the most
meaning. `12 1/4"` becomes three junk tokens, `1,940 m` splits into a number
and a letter, `ORD-118` loses the well identity. Every one of those is a term
an engineer would actually search on, so they get their own rules.
"""

from __future__ import annotations

import re
import unicodedata

from .config import SYNTHETIC_FOOTER

# 12 1/4"  /  8-1/2 in  /  17.5"
_SECTION_RE = re.compile(r'(\d{1,2})\s*[-\s]?\s*(\d)/(\d)\s*(?:"|in\b|inch\b)', re.I)
_SECTION_DEC_RE = re.compile(r'(\d{1,2}(?:\.\d+)?)\s*(?:"|in\b|inch\b)', re.I)
# ORD-118, VSS-201, well identifiers in general
_WELL_RE = re.compile(r"\b([A-Z]{2,5})-(\d{2,4})\b")
# 2,850 m / 2850m / 2 850 m
_DEPTH_RE = re.compile(r"\b(\d{1,3}(?:[,\s]\d{3})+|\d{3,5})\s*m\b", re.I)
# 1.42 sg / 1,42 sg
_MW_RE = re.compile(r"\b(\d\.\d{1,2})\s*(?:sg|s\.g\.)\b", re.I)
_WORD_RE = re.compile(r"[a-z0-9_]+", re.I)

_FRACTIONS = {"¼": " 1/4", "½": " 1/2", "¾": " 3/4", "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8", "⅞": " 7/8"}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "we", "our", "no", "not", "but", "if",
    "then", "than", "so", "up", "out", "into", "over", "per", "was", "had",
    "has", "have", "will", "would", "can", "could", "hrs", "hr", "h",
}


def normalise(text: str) -> str:
    """Fold unicode fractions and dashes so `12¼"` and `12 1/4"` match."""
    text = unicodedata.normalize("NFKC", text)
    for frac, repl in _FRACTIONS.items():
        text = text.replace(frac, repl)
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return text


def canonical_section(raw: str) -> str:
    """Map any spelling of a hole size onto one canonical string."""
    raw = normalise(raw)
    m = _SECTION_RE.search(raw)
    if m:
        whole, num, den = m.groups()
        return f'{int(whole)} {int(num)}/{int(den)}"'
    m = _SECTION_DEC_RE.search(raw)
    if m:
        val = float(m.group(1))
        whole = int(val)
        frac = round(val - whole, 3)
        table = {0.0: "", 0.125: " 1/8", 0.25: " 1/4", 0.5: " 1/2", 0.75: " 3/4", 0.625: " 5/8"}
        if frac in table:
            return f'{whole}{table[frac]}"'
        return f'{val}"'
    return raw.strip()


def tokenize(text: str) -> list[str]:
    """Produce searchable terms, keeping domain entities as single tokens."""
    text = normalise(text)
    tokens: list[str] = []

    for m in _SECTION_RE.finditer(text):
        whole, num, den = m.groups()
        tokens.append(f"sec:{int(whole)}_{int(num)}_{int(den)}")
    for m in _WELL_RE.finditer(text):
        tokens.append(f"well:{m.group(1).lower()}-{m.group(2)}")
    for m in _DEPTH_RE.finditer(text):
        digits = re.sub(r"[,\s]", "", m.group(1))
        tokens.append(f"depth:{digits}")
        # Bucket to 100 m so "losses around 2,700 m" retrieves 2,684 m too.
        tokens.append(f"dband:{int(int(digits) // 100) * 100}")
    for m in _MW_RE.finditer(text):
        tokens.append(f"mw:{m.group(1)}")

    for w in _WORD_RE.findall(text.lower()):
        if w in STOPWORDS or len(w) < 2:
            continue
        if w.isdigit() and len(w) > 5:
            continue
        tokens.append(w)

    return tokens


def snippet(text: str, terms: list[str], width: int = 240) -> str:
    """Pull the most term-dense line out of a document for display.

    The synthetic-document footer is never picked.
    """
    wanted = {t.lower() for t in terms if not t.startswith(("sec:", "well:", "depth:", "dband:", "mw:"))}
    best, best_score = "", -1
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < 12 or stripped == SYNTHETIC_FOOTER:
            continue
        low = stripped.lower()
        score = sum(1 for t in wanted if t in low)
        if score > best_score:
            best, best_score = stripped, score
    if not best:
        best = text.strip().splitlines()[0] if text.strip() else ""
    return best[:width]
