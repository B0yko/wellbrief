"""Records the simulation fills in and the renderer turns into lines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..models import Document, Well
from .events import VARIANTS, Variant
from .fields import FieldSpec, Formation, SectionPlan


def fmt_m(depth: float) -> str:
    """Depth or length in whole metres with a thousands separator."""
    return f"{round(depth):,}"


@dataclass
class NptEvent:
    """One non-productive time event. It may run over several daily reports."""

    variant: str
    tenths: int                  # total duration in tenths of an hour
    depth_m: int
    formation: str
    params: dict[str, Any] = field(default_factory=dict)
    number: int = 0              # order of occurrence on the well, from 1
    parts: list[NptEntry] = field(default_factory=list)

    @property
    def spec(self) -> Variant:
        return VARIANTS[self.variant]

    @property
    def code(self) -> str:
        return self.spec.code

    @property
    def pattern(self) -> str | None:
        return self.spec.pattern

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def hours(self) -> float:
        return self.tenths / 10

    def text(self, which: str, tenths: int) -> str:
        template = getattr(self.spec, which)
        return str(template.format(d=fmt_m(self.depth_m), h=f"{tenths / 10:.1f}", **self.params))


@dataclass
class NptEntry:
    """One NPT DETAIL entry of a daily report: an event's share of that report."""

    event: NptEvent
    part: int                    # 1 for the report where the event starts
    tenths: int
    description: str
    report_id: str = ""

    @property
    def code(self) -> str:
        return self.event.code

    @property
    def pattern(self) -> str | None:
        return self.event.pattern

    @property
    def kind(self) -> str:
        return self.event.kind

    @property
    def variant(self) -> str:
        return self.event.variant

    @property
    def depth_m(self) -> int:
        return self.event.depth_m

    @property
    def formation(self) -> str:
        return self.event.formation

    @property
    def hours(self) -> float:
        return self.tenths / 10


@dataclass
class Activity:
    """One line of the operations log."""

    text: str
    tenths: int
    entry: NptEntry | None = None
    kind: str = ""               # the operation kind: drill, move, fixed, fill or npt
    tag: str = ""                # what the operation is, for the write-up (e.g. "casing_trip_out")


@dataclass
class DayReport:
    doc_id: str
    report_no: int
    section: SectionPlan
    depth_start: int
    depth_end: int
    formation_end: Formation
    mud_system: str
    mud_weight_sg: float
    bht_c: int
    mwd: str
    drilled: bool
    entries: list[NptEntry]
    activities: list[Activity]
    remarks: list[str]
    cosmetics: dict[str, Any]
    formations: list[str]
    day: date = date(1970, 1, 1)

    @property
    def npt_tenths(self) -> int:
        return sum(e.tenths for e in self.entries)


@dataclass
class Sentence:
    """A lesson, recommendation or corrective action with its true label."""

    kind: str                    # lesson | recommendation | corrective_action
    text: str
    label: str                   # practice | failure | neutral
    patterns: tuple[str, ...] = ()
    code: str | None = None


@dataclass
class Incident:
    doc_id: str
    number: int
    event: NptEvent
    severity: str
    root_cause: str
    actions: list[Sentence]

    @property
    def reports(self) -> list[str]:
        return [p.report_id for p in self.event.parts]


@dataclass
class WellRecord:
    spec: FieldSpec
    index: int
    name: str
    rig: str
    td_m: int
    mwd: str                     # the MWD generation run in the 12 1/4" section
    mwd_by_section: dict[str, str]
    salt_mw_sg: float
    salt_mw_ok: bool
    lcm_pretreat: bool
    move_days: int = 0
    spud: date = date(1970, 1, 1)
    days: list[DayReport] = field(default_factory=list)
    events: list[NptEvent] = field(default_factory=list)
    eowr_date: date = date(1970, 1, 1)
    patterns: dict[str, str] = field(default_factory=dict)
    lessons: list[Sentence] = field(default_factory=list)
    recommendations: list[Sentence] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    citations: dict[str, str] = field(default_factory=dict)

    @property
    def eowr_id(self) -> str:
        return f"EOWR-{self.name}"

    @property
    def release(self) -> date:
        """The date of the last daily report."""
        return self.days[-1].day

    @property
    def entries(self) -> list[NptEntry]:
        return [e for d in self.days for e in d.entries]

    def pattern_events(self, pattern: str) -> list[NptEvent]:
        return [e for e in self.events if e.pattern == pattern]

    def report(self, doc_id: str) -> DayReport:
        return next(d for d in self.days if d.doc_id == doc_id)


@dataclass
class RenderedDocument:
    """A document as a list of lines. Writers turn it into a file."""

    doc_id: str
    doc_type: str
    well: str
    field_name: str
    date: str
    title: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


@dataclass
class Corpus:
    seed: int
    scale: int
    fields: tuple[FieldSpec, ...]
    rigs: dict[str, tuple[str, ...]]
    wells: list[WellRecord]
    documents: list[RenderedDocument]

    def well_models(self) -> list[Well]:
        return [
            Well(
                name=w.name,
                field_name=w.spec.name,
                rig=w.rig,
                spud_date=w.spud.isoformat(),
                td_m=float(w.td_m),
                sections=[s.size for s in w.spec.sections],
                formations=[f.name for f in w.spec.formations],
            )
            for w in self.wells
        ]

    def document_models(self) -> list[Document]:
        return [
            Document(
                doc_id=d.doc_id, doc_type=d.doc_type, well=d.well, field_name=d.field_name,
                date=d.date, title=d.title, text=d.text, meta={},
            )
            for d in self.documents
        ]
