"""The narrator interface shared by `narrate.offline.OfflineNarrator` and
`narrate.llm.LlmNarrator`.

`qa.ask` and `riskbrief.build_brief` accept any `Narrator`. `last_rejection`
is how a narrator with somewhere to fall back to (the `llm` narrator) reports
that it replaced its own text with the offline narrator's: it is set on the
instance right after a call whose text failed the verifier
(`narrate.verify`) or whose request could not be sent at all, and read by the
caller immediately afterwards. `OfflineNarrator` never sets it, because it
has nothing to fall back to.
"""

from __future__ import annotations

import re
from typing import Any

#: The first line of an answer that abstains (see `qa.abstention_text`).
NO_MATCH = "No records in this workspace match that question."

_CITED_ID_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


class Narrator:
    name = "base"

    #: Set by a narrator that fell back to the offline answer on its most recent call:
    #: `{"text": <the rejected narrative>, "reasons": [<why the verifier rejected it>]}`.
    #: `None` otherwise, including for every `OfflineNarrator` call.
    last_rejection: dict[str, Any] | None = None

    def answer(self, question: str, evidence: list[dict[str, Any]], summary: dict[str, Any]) -> str:
        raise NotImplementedError

    def risk_brief(self, brief: Any) -> str:
        raise NotImplementedError


def extract_cited_ids(text: str) -> set[str]:
    """Every `[...]` bracketed token in `text`, the shape a narrator cites a document id in."""
    return set(_CITED_ID_RE.findall(text))
