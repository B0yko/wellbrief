"""Scoring helpers: ranking metrics, tolerances and text matching."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence

# Pass tolerances per reported figure: 0.05 h, $1, 0.001 for shares.
TOLERANCES = {"total_hours": 0.05, "total_cost_usd": 1.0, "avoidable_share": 0.001}
# Float slack so a difference of exactly the tolerance passes.
_EPS = 1e-9


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """|top k ∩ relevant| / min(k, |relevant|): 1.0 is reachable even when fewer than k are relevant."""
    if not relevant:
        raise ValueError("precision is undefined for an empty relevant set")
    top = list(dict.fromkeys(ranking))[:k]
    return sum(1 for d in top if d in relevant) / min(k, len(relevant))


def mrr_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant document within the top k, else 0."""
    for rank, doc_id in enumerate(list(dict.fromkeys(ranking))[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def within(value: float, gold: float, tolerance: float) -> bool:
    return abs(value - gold) <= tolerance + _EPS


def match_counts(found: Iterable[Hashable], gold: Iterable[Hashable]) -> tuple[int, int, int]:
    """Multiset matching: (matched, found, gold) counts."""
    f, g = Counter(found), Counter(gold)
    return sum((f & g).values()), sum(f.values()), sum(g.values())


def squash(text: str) -> str:
    """Whitespace-insensitive form used for every verbatim comparison."""
    return re.sub(r"\s+", " ", text).strip()


def verbatim(quote: str, source: str) -> bool:
    """The quote appears in the source word for word (whitespace runs count as one space)."""
    q = squash(quote)
    return bool(q) and q in squash(source)
