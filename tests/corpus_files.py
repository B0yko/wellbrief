"""Read a generated corpus back from its files, for the invariant tests.

Everything here works on the written text: the product parser for the
fields it extracts, and small independent readers for the layout the
parser does not cover (header labels, operations log, remarks, raw NPT
DETAIL labels, the end-of-well summary and the incident source line).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from wellbrief import parse
from wellbrief.corpus import build_corpus, write_corpus
from wellbrief.models import Document

OPS_START = re.compile(r"^  (\d\d):(\d\d)-(\d\d):(\d\d)  (.*)$")
NPT_OPS = re.compile(r"^NPT ([A-Z_]+) (\d+\.\d) h: (.*)$")
DEPTH_QUOTE = re.compile(r"\bat (\d{1,3}(?:,\d{3})*) m\b")
DURATION_QUOTE = re.compile(r"(\d+(?:\.\d+)?) h\b")
LABEL = re.compile(r"^  ([A-Za-z][A-Za-z /]*?)\s*: (.*)$")


@dataclass
class Op:
    start: int          # minutes after midnight
    end: int
    text: str

    @property
    def minutes(self) -> int:
        """Length of the line; an operation that fills the whole report runs 06:00-06:00."""
        return (self.end - self.start) % (24 * 60) or 24 * 60

    @property
    def npt(self) -> re.Match[str] | None:
        return NPT_OPS.match(self.text)


@dataclass
class Ddr:
    doc_id: str
    well: str
    field_name: str
    date: str
    rig: str
    mwd: str
    section: str
    depth_start: float
    depth_end: float
    formation_at_td: str
    mud_weight: float
    productive: float
    npt: float
    entries: list[dict[str, Any]]
    raw_npt_labels: list[str]
    ops: list[Op]
    labels: dict[str, str]
    remarks: list[str]
    text: str

    @property
    def drilled(self) -> bool:
        return any(o.text.startswith("Drilled ") and " hole from " in o.text for o in self.ops)


@dataclass
class Eowr:
    doc_id: str
    well: str
    field_name: str
    date: str
    spud: str
    td: float
    days: int
    summary: list[str]
    lessons: list[str]
    recommendations: list[str]
    npt_by_code: dict[str, float]
    text: str


@dataclass
class Incident:
    doc_id: str
    well: str
    field_name: str
    date: str
    code: str
    hours: float
    depth: float
    section: str
    formation: str
    reports: list[str]
    actions: list[str]
    text: str


@dataclass
class ParsedCorpus:
    root: Path
    truth: dict[str, Any]
    ddrs: list[Ddr] = field(default_factory=list)
    eowrs: dict[str, Eowr] = field(default_factory=dict)
    incidents: list[Incident] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)

    def ddrs_of(self, well: str) -> list[Ddr]:
        return [d for d in self.ddrs if d.well == well]

    def ddr(self, doc_id: str) -> Ddr:
        return next(d for d in self.ddrs if d.doc_id == doc_id)

    def field_spec(self, field_name: str) -> dict[str, Any]:
        return next(f for f in self.truth["fields"] if f["name"] == field_name)

    def formation_at(self, field_name: str, depth: float) -> str:
        formations = self.field_spec(field_name)["formations"]
        for f in formations:
            if f["top_m"] <= depth < f["base_m"]:
                return str(f["name"])
        return str(formations[-1]["name"])

    def formation_top(self, field_name: str, name: str) -> int:
        return int(next(f["top_m"] for f in self.field_spec(field_name)["formations"] if f["name"] == name))

    def section(self, field_name: str, size: str) -> dict[str, Any]:
        return next(s for s in self.field_spec(field_name)["sections"] if s["size"] == size)

    def section_top(self, field_name: str, size: str) -> int:
        return int(self.section(field_name, size)["top_m"])


def depth_value(text: str) -> float:
    """'1,850 m' -> 1850.0; 'surface' -> 0.0."""
    return 0.0 if text == "surface" else float(text.removesuffix(" m").replace(",", ""))


def _block(lines: list[str], heading: str) -> list[str]:
    i = lines.index(heading) + 1
    out = []
    while i < len(lines) and lines[i].strip():
        out.append(lines[i])
        i += 1
    return out


def _ops_log(lines: list[str]) -> list[Op]:
    out: list[Op] = []
    for line in _block(lines, "OPERATIONS SUMMARY (24 h)"):
        m = OPS_START.match(line)
        if m:
            start = int(m.group(1)) * 60 + int(m.group(2))
            end = int(m.group(3)) * 60 + int(m.group(4))
            out.append(Op(start, end, m.group(5).strip()))
        else:
            out[-1].text = f"{out[-1].text} {line.strip()}"
    return out


def _labels(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines[: lines.index("OPERATIONS SUMMARY (24 h)")]:
        m = LABEL.match(line)
        if m:
            out.setdefault(m.group(1).strip(), m.group(2).strip())
    return out


def _remarks(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in _block(lines, "REMARKS"):
        if line.startswith("    "):
            out[-1] = f"{out[-1]} {line.strip()}"
        else:
            out.append(line.strip())
    return out


def _raw_npt_labels(lines: list[str]) -> list[str]:
    i = lines.index("NPT DETAIL") + 1
    labels = []
    while i < len(lines) and lines[i] != "REMARKS":
        m = re.match(r"^  ([A-Za-z]+)\s*: ", lines[i])
        if m:
            labels.append(m.group(1))
        i += 1
    return labels


def _summary(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in _block(lines, "1. WELL SUMMARY"):
        if line.startswith("    "):
            out[-1] = f"{out[-1]} {line.strip()}"
        else:
            out.append(line.strip())
    return out


def load(root: Path) -> ParsedCorpus:
    truth = json.loads((root / "_truth.json").read_text(encoding="utf-8"))
    pc = ParsedCorpus(root=root, truth=truth)
    for path in sorted(root.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        pc.texts[path.stem] = text
        if path.name.startswith("DDR-"):
            p = parse.parse_ddr(text)
            pc.ddrs.append(Ddr(
                doc_id=path.stem, well=p["well"], field_name=p["field_name"], date=p["date"], rig=p["rig"],
                mwd=p["mwd"], section=p["hole_section"], depth_start=p["depth_start_m"],
                depth_end=p["depth_end_m"], formation_at_td=p["formation"], mud_weight=p["mud_weight_sg"],
                productive=p["productive_hours"], npt=p["npt_hours"], entries=p["npt"],
                raw_npt_labels=_raw_npt_labels(lines), ops=_ops_log(lines), labels=_labels(lines),
                remarks=_remarks(lines), text=text,
            ))
        elif path.name.startswith("EOWR-"):
            p = parse.parse_eowr(text)
            spud = re.search(r"Spud: (\d{4}-\d{2}-\d{2})", text)
            assert spud is not None
            pc.eowrs[p["well"]] = Eowr(
                doc_id=path.stem, well=p["well"], field_name=p["field_name"], date=p["date"],
                spud=spud.group(1), td=p["td_m"], days=int(p["days_on_well"]), summary=_summary(lines),
                lessons=p["lessons"], recommendations=p["recommendations"], npt_by_code=p["npt_by_code"],
                text=text,
            )
        elif path.name.startswith("INC-"):
            p = parse.parse_incident(text)
            source = re.search(r"^Daily reports?: (.*)$", text, re.M)
            assert source is not None
            pc.incidents.append(Incident(
                doc_id=path.stem, well=p["well"], field_name=p["field_name"], date=p["date"], code=p["code"],
                hours=p["hours"], depth=p["depth_m"], section=p["hole_section"], formation=p["formation"],
                reports=[r.strip() for r in source.group(1).split(",")], actions=p["corrective_actions"],
                text=text,
            ))
    pc.ddrs.sort(key=lambda d: d.doc_id)
    return pc


def ledger(pc: ParsedCorpus) -> list[Any]:
    """The NPT rows the product's parser extracts from the written DDRs, in document order."""
    rows = []
    for d in pc.ddrs:
        doc = Document(doc_id=d.doc_id, doc_type="ddr", well=d.well, field_name=d.field_name,
                       date=d.date, title="", text=d.text)
        rows.extend(parse.npt_events(doc))
    return rows


@cache
def generated(root: str, seed: int, scale: int = 1) -> ParsedCorpus:
    out = Path(root)
    write_corpus(build_corpus(seed=seed, scale=scale), out)
    return load(out)
