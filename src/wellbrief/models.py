"""Domain objects.

Deliberately plain dataclasses. Everything that crosses a module boundary is
serialisable to JSON, so the same objects feed the CLI's JSON output and the
text renderers without a translation layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """One source file from the operator's archive."""

    doc_id: str
    doc_type: str            # ddr | eowr | incident
    well: str
    field_name: str
    date: str                # ISO-8601 date
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    source: str = ""              # path of the ingested file, relative to the ingested root
    page_map: list[int] | None = None   # PDF only: cumulative end-offset of each page in `text`

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """One retrievable slice of a document's text (see `store.chunk_document`).

    `start`/`end` are character offsets into the document's raw text;
    `chunk_id` is `<doc_id>#<n>`, which is also how a chunk id is mapped back
    to its document (`store.chunk_document_id`), so no separate mapping needs
    to be persisted alongside the per-field indexes.
    """

    chunk_id: str
    doc_id: str
    n: int
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileRecord:
    """One ingested file's bookkeeping row (the `files` table).

    `doc_ids` is a list because one file can yield more than one document (a
    CSV ledger row becomes its own citable one-line document); `reason` is
    set when `status` is not `"ingested"` (for example a skipped or unreadable
    file), for the ingest coverage report.
    """

    path: str
    sha256: str
    size: int
    mtime: float
    doc_ids: list[str]
    status: str
    reason: str | None = None
    field_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NptEvent:
    """A single chunk of non-productive time attributed to one report."""

    doc_id: str
    well: str
    field_name: str
    date: str
    code: str
    hours: float
    hole_section: str
    formation: str
    depth_m: float
    mud_weight_sg: float
    rig: str
    description: str
    mwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Well:
    name: str
    field_name: str
    rig: str
    spud_date: str
    td_m: float
    sections: list[str] = field(default_factory=list)
    formations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    """A pointer back to the exact line an answer came from.

    `quote` is copied verbatim out of the source document.
    `riskbrief.verify_brief` checks every brief citation against its source,
    and `qa.verify_citations` flags any document id an answer cites that is
    not in its evidence pack. `page` is the 1-based page the quote falls on
    when the source document carries a page map (PDF only, from
    `quotes.quote_page`); null for every other document type.
    """

    doc_id: str
    doc_type: str
    well: str
    date: str
    quote: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    doc_id: str
    score: float
    document: Document
    snippet: str = ""       # the evidence quote for this hit (see `quotes.evidence_quote`)
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Risk:
    """One entry in an offset-well risk register.

    `scope` is `interval` (keyed on hole section and formation) or
    `equipment` (keyed on `rig` or `mwd`, whichever is set; the other is
    empty). `applies` is one of `applies`, `does_not_apply` or
    `not_specified`: whether the planned well's `--rig`/`--mwd` names the
    same category as this risk (always `applies` for an interval risk, since
    it does not depend on rig or tool). A risk that does not apply is kept in
    `risks` so the register still shows it, but `RiskBrief`'s totals skip it.
    """

    risk_id: str
    title: str
    code: str
    scope: str
    hole_section: str
    formation: str
    rig: str
    mwd: str
    depth_window_m: tuple[float, float]
    wells_total: int
    wells_affected: int
    probability: float
    mean_npt_hours: float
    p90_npt_hours: float
    expected_npt_hours: float
    expected_cost_usd: float
    driver: str
    lift: float | None = None
    ratio: float | None = None
    applies: str = "applies"
    mitigations: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["depth_window_m"] = list(self.depth_window_m)
        return d


@dataclass
class RiskBrief:
    well_name: str
    field_name: str
    planned_td_m: float
    generated_from_wells: list[str]
    spread_rate_usd_per_day: float
    risks: list[Risk] = field(default_factory=list)
    unavoidable_hours: dict[str, float] = field(default_factory=dict)
    planned_rig: str | None = None
    planned_mwd: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""

    @property
    def counted_risks(self) -> list[Risk]:
        """Risks that count toward the totals: every interval risk, and an
        equipment risk that applies to the plan or that the plan did not
        speak to (`applies` or `not_specified`, never `does_not_apply`)."""
        return [r for r in self.risks if r.applies != "does_not_apply"]

    @property
    def total_expected_npt_hours(self) -> float:
        return sum(r.expected_npt_hours for r in self.counted_risks)

    @property
    def total_exposure_usd(self) -> float:
        return sum(r.expected_cost_usd for r in self.counted_risks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "well_name": self.well_name,
            "field_name": self.field_name,
            "planned_td_m": self.planned_td_m,
            "planned_rig": self.planned_rig,
            "planned_mwd": self.planned_mwd,
            "generated_from_wells": self.generated_from_wells,
            "spread_rate_usd_per_day": self.spread_rate_usd_per_day,
            "total_expected_npt_hours": round(self.total_expected_npt_hours, 1),
            "total_exposure_usd": round(self.total_exposure_usd, 0),
            "unavoidable_hours": {k: round(v, 1) for k, v in self.unavoidable_hours.items()},
            "risks": [r.to_dict() for r in self.risks],
            "provenance": self.provenance,
            "narrative": self.narrative,
        }


@dataclass
class Answer:
    """The result of `qa.ask`.

    `figures` are computed with SQL over the NPT ledger (None when the question
    scopes nothing countable or no entry matches); `figure_sources` lists every
    report that contributes to them, with its hours. An answer that abstains
    carries no figures, no citations and no mitigations, and says in
    `unmatched` what did not match.
    """

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    query_plan: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] | None = None
    figure_sources: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False
    unmatched: dict[str, Any] | None = None
    mitigations: list[dict[str, str]] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)
    narrator: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "abstained": self.abstained,
            "unmatched": self.unmatched,
            "query_plan": self.query_plan,
            "figures": self.figures,
            "figure_sources": self.figure_sources,
            "citations": [c.to_dict() for c in self.citations],
            "mitigations": self.mitigations,
            "citation_warnings": self.citation_warnings,
            "narrator": self.narrator,
        }
