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
    not in its evidence pack.
    """

    doc_id: str
    doc_type: str
    well: str
    date: str
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    doc_id: str
    score: float
    document: Document
    snippet: str = ""
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Risk:
    """One entry in an offset-well risk register."""

    risk_id: str
    title: str
    code: str
    hole_section: str
    formation: str
    depth_window_m: tuple[float, float]
    wells_total: int
    wells_affected: int
    probability: float
    mean_npt_hours: float
    p90_npt_hours: float
    expected_npt_hours: float
    expected_cost_usd: float
    driver: str
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
    narrative: str = ""

    @property
    def total_expected_npt_hours(self) -> float:
        return sum(r.expected_npt_hours for r in self.risks)

    @property
    def total_exposure_usd(self) -> float:
        return sum(r.expected_cost_usd for r in self.risks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "well_name": self.well_name,
            "field_name": self.field_name,
            "planned_td_m": self.planned_td_m,
            "generated_from_wells": self.generated_from_wells,
            "spread_rate_usd_per_day": self.spread_rate_usd_per_day,
            "total_expected_npt_hours": round(self.total_expected_npt_hours, 1),
            "total_exposure_usd": round(self.total_exposure_usd, 0),
            "risks": [r.to_dict() for r in self.risks],
            "narrative": self.narrative,
        }


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    structured: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "structured": self.structured,
        }
