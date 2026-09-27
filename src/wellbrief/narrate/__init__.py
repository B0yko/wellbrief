"""Narrators: the offline template writer (`narrate.offline`) and the OpenAI-compatible model
narrator behind an egress guard and a cost ledger (`narrate.llm`), and the verifier that checks
either one's output against its evidence (`narrate.verify`).

`get_narrator` is the one place that turns a narrator name into an instance; the CLI resolves the
`llm` narrator's configuration (`narrate.llm.narrator_from_env`) itself, since that needs the
workspace's egress log path.
"""

from __future__ import annotations

from .base import NO_MATCH, Narrator, extract_cited_ids
from .offline import OfflineNarrator

__all__ = ["NARRATORS", "NO_MATCH", "Narrator", "OfflineNarrator", "extract_cited_ids", "get_narrator"]

NARRATORS = ("offline", "llm")


def get_narrator(name: str = "offline") -> Narrator:
    """Return the narrator for a name that needs no further configuration.

    `offline` is the only one this resolves: the `llm` narrator carries an HTTP
    client, an egress log and a cost ledger, none of which has a parameter-free
    default, so a caller that wants it builds it with
    `narrate.llm.narrator_from_env` instead.

    Raises:
        ValueError: `name` is not `offline` (or `llm`, which this cannot build).
    """
    if name == "offline":
        return OfflineNarrator()
    if name == "llm":
        raise ValueError(
            "get_narrator cannot build the llm narrator (it needs an egress log path); "
            "call narrate.llm.narrator_from_env instead"
        )
    raise ValueError(f"unknown narrator {name!r}; expected one of {NARRATORS}")
