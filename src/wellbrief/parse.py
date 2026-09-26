"""Structured extraction from well paperwork.

The retrieval layer works on raw text. Everything numeric (hours, depths, mud
weights, NPT codes) is pulled out here and stored in a real table, because an
answer like "42 hours of stuck pipe across nine wells" has to come from
arithmetic over parsed records, not from a language model reading documents
and guessing at a total.

Extraction is deliberately rule based. Rules are auditable, cost nothing to
run and give the same result every time. The labels and headings they look
for are fixed in this module, so a report in another template needs its
patterns added here.
"""

from __future__ import annotations

import re
from typing import Any

from .config import NPT_CODES
from .models import Document, NptEvent
from .text import canonical_section, normalise

_LABEL = r"[ \t]*{}[ \t]*:[ \t]*(.+)"


def _label(text: str, label: str) -> str | None:
    m = re.search(_LABEL.format(label), text, re.I)
    return m.group(1).strip() if m else None


def _num(raw: str | None) -> float | None:
    if raw is None:
        return None
    m = re.search(r"-?\d[\d,\s]*(?:\.\d+)?", raw)
    if not m:
        return None
    try:
        return float(re.sub(r"[,\s]", "", m.group(0)))
    except ValueError:
        return None


def parse_header(text: str) -> dict[str, Any]:
    """Fields shared by every document family."""
    text = normalise(text)
    out: dict[str, Any] = {}
    m = re.search(r"Operator:\s*(.+?)\s{2,}Field:\s*(.+?)\s{2,}Well:\s*(\S+)", text, re.I)
    if m:
        out["operator"], out["field_name"], out["well"] = (g.strip() for g in m.groups())
    m = re.search(r"Rig:\s*(\S+)", text, re.I)
    if m:
        out["rig"] = m.group(1).strip()
    m = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text, re.I)
    if m:
        out["date"] = m.group(1)
    m = re.search(r"Report No:\s*(\d+)", text, re.I)
    if m:
        out["report_no"] = int(m.group(1))
    return out


def parse_ddr(text: str) -> dict[str, Any]:
    """Pull the numeric spine out of a daily drilling report."""
    text = normalise(text)
    out = parse_header(text)

    out["depth_start_m"] = _num(_label(text, "Depth at start"))
    out["depth_end_m"] = _num(_label(text, "Depth at end"))
    out["progress_m"] = _num(_label(text, "Progress"))

    raw_section = _label(text, "Hole section")
    out["hole_section"] = canonical_section(raw_section) if raw_section else None

    formation = _label(text, r"Formation at TD")
    if formation:
        out["formation"] = re.sub(r"\s*\(.*\)\s*$", "", formation).strip()

    out["mud_weight_sg"] = _num(_label(text, "Weight"))
    out["ecd_sg"] = _num(_label(text, "ECD at shoe"))
    out["bht_c"] = _num(_label(text, "BHT max circulating"))
    out["mwd"] = _label(text, "MWD")
    out["productive_hours"] = _num(_label(text, "Productive time"))
    out["npt_hours"] = _num(_label(text, "Non-productive time"))

    m = re.search(r"Flow\s*:\s*.*?/\s*(\d+)\s*gpm", text, re.I)
    if m:
        out["flow_gpm"] = float(m.group(1))

    out["npt"] = parse_npt_blocks(text)
    return out


def parse_npt_blocks(text: str) -> list[dict[str, Any]]:
    """Read the repeating Code / Hours / Description triples in NPT DETAIL."""
    text = normalise(text)
    start = re.search(r"^NPT DETAIL\s*$", text, re.M)
    if not start:
        return []
    tail = text[start.end():]
    stop = re.search(r"^REMARKS\s*$", tail, re.M)
    if stop:
        tail = tail[: stop.start()]

    events: list[dict[str, Any]] = []
    pattern = re.compile(
        r"Code\s*:\s*([A-Z_]+).*?Hours\s*:\s*([\d.]+).*?Description\s*:\s*(.+?)(?=\n\s*Code\s*:|\Z)",
        re.S,
    )
    for m in pattern.finditer(tail):
        code, hours, desc = m.group(1).strip(), float(m.group(2)), " ".join(m.group(3).split())
        if code not in NPT_CODES:
            code = "OTHER"
        events.append({"code": code, "hours": hours, "description": desc})
    return events


def parse_eowr(text: str) -> dict[str, Any]:
    """End of well report: totals plus the numbered lessons."""
    text = normalise(text)
    out = parse_header(text)
    out["days_on_well"] = _num(_label(text, "Days on well"))
    out["td_m"] = _num(_label(text, "Total depth"))
    out["total_npt_hours"] = _num(_label(text, "Non-productive time"))

    by_code: dict[str, float] = {}
    block = re.search(r"3\. NPT BREAKDOWN BY CODE(.*?)(?:\n4\.|\Z)", text, re.S)
    if block:
        for m in re.finditer(r"^\s*([A-Z_]{3,})\s+([\d.]+)\s*h\s*$", block.group(1), re.M):
            by_code[m.group(1)] = float(m.group(2))
    out["npt_by_code"] = by_code

    lessons: list[str] = []
    block = re.search(r"4\. LESSONS LEARNED(.*?)(?:\n5\.|\Z)", text, re.S)
    if block:
        for m in re.finditer(r"^\s*\d+\.\s+(.+?)(?=\n\s*\d+\.\s|\Z)", block.group(1), re.S | re.M):
            lessons.append(" ".join(m.group(1).split()))
    out["lessons"] = lessons

    recs: list[str] = []
    block = re.search(r"5\. RECOMMENDATIONS FOR FUTURE WELLS(.*?)\Z", text, re.S)
    if block:
        for m in re.finditer(r"^\s*-\s+(.+)$", block.group(1), re.M):
            recs.append(" ".join(m.group(1).split()))
    out["recommendations"] = recs
    return out


def parse_incident(text: str) -> dict[str, Any]:
    text = normalise(text)
    out = parse_header(text)
    m = re.search(r"Classification:\s*([A-Z_]+)\s+Lost time:\s*([\d.]+)", text, re.I)
    if m:
        out["code"], out["hours"] = m.group(1), float(m.group(2))
    m = re.search(r"Depth:\s*([\d,]+)\s*m MD\s+Hole section:\s*(\S+(?:\s+\d/\d\")?)\s+Formation:\s*(.+)", text)
    if m:
        out["depth_m"] = _num(m.group(1))
        out["hole_section"] = canonical_section(m.group(2))
        out["formation"] = m.group(3).strip()
    m = re.search(r"Severity:\s*(\w+)", text, re.I)
    if m:
        out["severity"] = m.group(1)
    block = re.search(r"4\. ROOT CAUSE\s*\n(.*?)(?:\n5\.|\Z)", text, re.S)
    if block:
        out["root_cause"] = " ".join(block.group(1).split())
    actions: list[str] = []
    block = re.search(r"5\. CORRECTIVE ACTIONS(.*?)\Z", text, re.S)
    if block:
        for m in re.finditer(r"^\s*-\s+(.+)$", block.group(1), re.M):
            actions.append(" ".join(m.group(1).split()))
    out["corrective_actions"] = actions
    return out


def parse(doc: Document) -> dict[str, Any]:
    if doc.doc_type == "ddr":
        return parse_ddr(doc.text)
    if doc.doc_type == "eowr":
        return parse_eowr(doc.text)
    if doc.doc_type == "incident":
        return parse_incident(doc.text)
    return parse_header(doc.text)


def npt_events(doc: Document) -> list[NptEvent]:
    """Turn one document into zero or more NPT rows."""
    if doc.doc_type != "ddr":
        return []
    p = parse_ddr(doc.text)
    events: list[NptEvent] = []
    for e in p.get("npt", []):
        events.append(NptEvent(
            doc_id=doc.doc_id,
            well=p.get("well") or doc.well,
            field_name=p.get("field_name") or doc.field_name,
            date=p.get("date") or doc.date,
            code=e["code"],
            hours=e["hours"],
            hole_section=p.get("hole_section") or "",
            formation=p.get("formation") or "",
            depth_m=p.get("depth_end_m") or 0.0,
            mud_weight_sg=p.get("mud_weight_sg") or 0.0,
            rig=p.get("rig") or "",
            description=e["description"],
        ))
    return events
