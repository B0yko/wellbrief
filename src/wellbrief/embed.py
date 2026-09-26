"""Pluggable embeddings.

The dense side of hybrid retrieval sits behind the `Embedder` interface, so
another backend can be added without touching the search code. This version
ships one implementation:

  offline    Signed hashing over words and character n-grams. No network, no
             download, no weights on disk. It gives fuzzy lexical matching, so
             "losses", "lost returns" and "losing circulation" land near each
             other. It captures lexical similarity, not meaning.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .config import EMBED_DIM
from .text import STOPWORDS, normalise

_WORD = re.compile(r"[a-z0-9]+")


class Embedder:
    name = "base"
    dim = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbedder(Embedder):
    """Signed hashing over words and character n-grams."""

    name = "offline"

    def __init__(self, dim: int = EMBED_DIM, ngrams: tuple[int, ...] = (3, 4)):
        self.dim = dim
        self.ngrams = ngrams

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        for word in _WORD.findall(normalise(text).lower()):
            if word in STOPWORDS or len(word) < 2:
                continue
            feats[f"w:{word}"] += 1
            padded = f"^{word}$"
            for n in self.ngrams:
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    feats[f"g:{padded[i:i + n]}"] += 1
        return feats

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat, count in self._features(text).items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sublinear term frequency: the tenth mention of "losses" in one
            # report should not outweigh the first mention in another.
            vec[slot] += sign * (1.0 + math.log(count))
        return _l2(vec)


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Both sides are already L2 normalised, so this is a dot product."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def get_embedder(backend: str = "offline") -> Embedder:
    """Return the embedder for a backend name.

    `offline` (the hashing embedder) is the only backend in this version, so
    every name resolves to it.
    """
    return HashingEmbedder()
