"""The one place where the evaluation harness calls the product.

Everything the suites need from wellbrief goes through this module: write a
synthetic corpus, build a store from it, ask a question, build a brief, list
the detected patterns, read the parsed NPT ledger. The results come back as
plain records, so the checks in `suites.py` never touch product objects.

This module follows the product as it evolves; the case files do not. When a
case needs something the product cannot do yet, the call raises
`NotSupported` with the reason, and the runner records the case as a failure
with that reason.

Nothing derived from the ground truth reaches the product through here except
two inputs a user could type: field names, and the candidate sentence texts
handed to the classifier (never their labels).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..analytics import find_patterns, interval_patterns
from ..corpus import FORMATS, build_corpus, write_corpus
from ..embed import get_embedder
from ..ingest import ingest, load_corpus_dir
from ..llm import OfflineNarrator
from ..qa import ask as product_ask
from ..riskbrief import build_brief, verify_brief
from ..search import Searcher
from ..store import Store, load_indexes


class NotSupported(Exception):
    """The product does not offer what the case needs yet."""


@dataclass(frozen=True)
class Cited:
    doc_id: str
    quote: str


@dataclass
class AskResult:
    text: str
    ranking: list[str]                  # retrieval ranking, one entry per document, best first
    citations: list[Cited]
    figures: dict[str, float]           # total_hours, total_cost_usd, avoidable_share, event_count
    abstained: bool | None = None       # None: the product does not report it
    warnings: list[str] = field(default_factory=list)


@dataclass
class Mitigation:
    text: str
    doc_id: str


@dataclass
class Risk:
    code: str
    scope: str                          # interval | equipment
    section: str | None
    formation: str | None
    rig: str | None
    mwd: str | None
    driver: str
    driver_kind: str | None             # mud_weight | rig | tool | None
    driver_category: str | None         # the rig or tool the driver blames (rig and tool drivers)
    mitigations: list[Mitigation]
    citations: list[Cited]


@dataclass
class Brief:
    risks: list[Risk]
    verified: bool
    problems: list[str]


@dataclass(frozen=True)
class Pattern:
    code: str
    formation: str | None
    hours: float


@dataclass(frozen=True)
class LedgerRow:
    doc_id: str
    field: str | None
    well: str
    date: str
    code: str
    hours: float
    depth_m: float | None
    section: str | None
    formation: str | None


def version() -> str:
    return __version__


def generate_corpus(out_dir: Path, seed: int, formats: str = "txt", ledger_csv: bool = False) -> None:
    """Write the synthetic corpus of `seed` (and its ground truth) to `out_dir`."""
    if formats not in FORMATS:
        raise NotSupported(f"the corpus generator cannot write the {formats!r} format yet")
    if ledger_csv:
        raise NotSupported("the corpus generator cannot write an NPT ledger CSV yet")
    write_corpus(build_corpus(seed=seed), out_dir, formats)


def render(out_dir: Path, seed: int, fmt: str) -> Path:
    """Write one rendering (pdf, docx or csv) of the corpus of `seed`; return the folder to ingest.

    The csv rendering is the NPT ledger file alone, in a folder of its own.
    """
    corpus_dir = out_dir / "corpus"
    if fmt != "csv":
        generate_corpus(corpus_dir, seed, fmt)
        return corpus_dir
    generate_corpus(corpus_dir, seed, "txt", ledger_csv=True)
    ledger_dir = out_dir / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(corpus_dir / "npt-ledger.csv", ledger_dir / "npt-ledger.csv")
    return ledger_dir


# The product's driver attributes, mapped to the driver kinds the truth file uses.
_DRIVER_KINDS = {"mud_weight_sg": "mud_weight", "rig": "rig", "tool_string": "tool"}



class Workspace:
    """A product store built from one generated corpus, in a directory of its own."""

    def __init__(self, corpus_dir: Path, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(work_dir / "wellbrief.db")
        ingest(self.store, load_corpus_dir(corpus_dir))
        bm25, vectors = load_indexes(self.store)
        self.searcher = Searcher(self.store, bm25, vectors, get_embedder("offline"))
        self.narrator = OfflineNarrator()

    def close(self) -> None:
        self.store.close()

    def ask(self, question: str, top_k: int = 8, spread_rate: float | None = None) -> AskResult:
        kwargs = {} if spread_rate is None else {"spread_rate": spread_rate}
        answer = product_ask(question, self.store, self.searcher, narrator=self.narrator, top_k=top_k,
                             **kwargs)
        computed = answer.figures or {}
        figures = {k: float(computed[k])
                   for k in ("total_hours", "total_cost_usd", "avoidable_share", "event_count")
                   if computed.get(k) is not None}
        return AskResult(
            text=answer.text,
            ranking=[h.doc_id for h in answer.hits],
            citations=[Cited(c.doc_id, c.quote) for c in answer.citations],
            figures=figures,
            abstained=answer.abstained,
            warnings=list(answer.citation_warnings),
        )

    def brief(self, field_name: str, well: str, td_m: float, risk_filters: bool = True) -> Brief:
        brief = build_brief(self.store, well, field_name, td_m, narrator=self.narrator,
                            risk_filters=risk_filters)
        check = verify_brief(brief, self.store)
        find_kwargs: dict[str, Any] = {} if risk_filters else {"min_lift": 0.0, "avoidable_only": False}
        drivers = {}
        for p in find_patterns(self.store, field_name, **find_kwargs):
            key = (p.code, "interval", p.hole_section, p.formation) if p.scope == "interval" \
                else (p.code, "equipment", p.rig, p.mwd)
            drivers[key] = (_DRIVER_KINDS.get(p.driver_detail.get("attribute", "")),
                           p.driver_detail.get("category"))
        risks = []
        for r in brief.risks:
            # The brief lists each mitigation's source report as one of the
            # risk's last citations, in the same order as the mitigations.
            sources = r.citations[len(r.citations) - len(r.mitigations):] if r.mitigations else []
            key = (r.code, "interval", r.hole_section, r.formation) if r.scope == "interval" \
                else (r.code, "equipment", r.rig, r.mwd)
            kind, category = drivers.get(key, (None, None))
            risks.append(Risk(
                code=r.code, scope=r.scope, section=r.hole_section or None, formation=r.formation or None,
                rig=r.rig or None, mwd=r.mwd or None, driver=r.driver, driver_kind=kind,
                driver_category=category,
                mitigations=[Mitigation(text, src.doc_id)
                             for text, src in zip(r.mitigations, sources, strict=True)],
                citations=[Cited(c.doc_id, c.quote) for c in r.citations],
            ))
        return Brief(risks=risks, verified=bool(check["ok"]), problems=list(check["problems"]))

    def patterns(self, field_name: str) -> list[Pattern]:
        """Interval-scope patterns only: this feeds `discovery`, which asks
        for the code and formation a place in the well keeps costing, not a
        piece of equipment (which has no formation)."""
        return [Pattern(p.code, p.formation, p.hours_total)
                for p in interval_patterns(self.store, field_name)]

    def ledger(self, field_name: str | None) -> list[LedgerRow]:
        """The parsed NPT rows of one field, or of the whole workspace (None).

        Rows are returned whatever field they were filed under.
        """
        return [
            LedgerRow(e.doc_id, e.field_name or None, e.well, e.date, e.code, e.hours, e.depth_m,
                      e.hole_section or None, e.formation or None)
            for e in self.store.npt(field_name=field_name)
        ]

    def classify(self, sentences: list[str]) -> list[str]:
        """The product's practice / failure / neutral label for each sentence text (labels never go in)."""
        raise NotSupported("the product has no practice/failure classifier yet")
