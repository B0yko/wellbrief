"""Okapi BM25 over the well-file corpus, in plain Python.

No external index, no service to run and no model to download, so the
index builds and answers queries on a laptop or an isolated server with
nothing but the standard library. The postings persist as one JSON file.
"""

from __future__ import annotations

import math
from collections import Counter

from .config import BM25_B, BM25_K1
from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = {}
        self.avgdl: float = 0.0

    # -- build ------------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        counts = Counter(tokenize(text))
        self.doc_len.append(sum(counts.values()))
        for term, tf in counts.items():
            self.postings.setdefault(term, {})[idx] = tf

    def finalise(self) -> None:
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

    @classmethod
    def build(cls, items: list[tuple[str, str]]) -> BM25Index:
        idx = cls()
        for doc_id, text in items:
            idx.add(doc_id, text)
        idx.finalise()
        return idx

    # -- query ------------------------------------------------------------
    def idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.doc_ids:
            return []
        scores: dict[int, float] = {}
        allowed_idx = None
        if allowed is not None:
            allowed_idx = {i for i, d in enumerate(self.doc_ids) if d in allowed}
            if not allowed_idx:
                return []

        for term, qtf in Counter(terms).items():
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            if idf <= 0:
                continue
            for doc_idx, tf in posting.items():
                if allowed_idx is not None and doc_idx not in allowed_idx:
                    continue
                dl = self.doc_len[doc_idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * qtf * tf * (self.k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], round(s, 6)) for i, s in ranked[:top_k]]

    # -- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "postings": {t: {str(i): tf for i, tf in p.items()} for t, p in self.postings.items()},
        }

    @classmethod
    def from_json(cls, blob: dict) -> BM25Index:
        idx = cls(k1=blob.get("k1", BM25_K1), b=blob.get("b", BM25_B))
        idx.doc_ids = list(blob["doc_ids"])
        idx.doc_len = list(blob["doc_len"])
        idx.avgdl = blob["avgdl"]
        idx.postings = {t: {int(i): tf for i, tf in p.items()} for t, p in blob["postings"].items()}
        return idx
