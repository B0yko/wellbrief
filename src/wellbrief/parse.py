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

Layout conventions the parsers rely on:

- a value that does not fit on its line continues on the following, more
  indented lines; NPT descriptions, lessons, recommendations and corrective
  actions are read as one logical line each;
- an NPT DETAIL entry may carry its own Depth and Formation lines; when it
  does not, the report header (depth at end, formation at TD) is used;
- every section ends at the synthetic-document footer line, which is never
  read as content.
"""

from __future__ import annotations

import re
from typing import Any

from .config import NPT_CODES, SYNTHETIC_FOOTER
from .models import Document, NptEvent
from .text import canonical_section, normalise

_LABEL = r"[ \t]*{}[ \t]*:[ \t]*(.+)"

# Labels of one NPT DETAIL entry in the canonical daily report template.
NPT_ENTRY_LABELS: dict[str, str] = {
    "code": "Code",
    "hours": "Hours",
    "depth": "Depth",
    "formation": "Formation",
    "description": "Description",
}

# A section heading: a line in capitals at the left margin.
_HEADING = re.compile(r"^[A-Z][A-Z0-9 ()/&.-]*$")
_NUMBERED_ITEM = re.compile(r"^[ \t]{0,3}\d+\.[ \t]+")
_BULLET_ITEM = re.compile(r"^[ \t]{0,3}[-*][ \t]+")


def strip_footer(text: str) -> str:
    """Cut the text at the synthetic-document footer line, if there is one."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == SYNTHETIC_FOOTER:
            return "\n".join(lines[:i])
    return text


def _items(block: str, marker: re.Pattern[str]) -> list[str]:
    """Numbered or bulleted items, each joined with its continuation lines.

    An item starts at a marker line; following non-blank lines that do not
    start a new item continue it; a blank line ends it.
    """
    items: list[list[str]] = []
    current: list[str] | None = None
    for line in block.splitlines():
        if not line.strip():
            current = None
            continue
        m = marker.match(line)
        if m:
            current = [line[m.end():].strip()]
            items.append(current)
        elif current is not None:
            current.append(line.strip())
    return [" ".join(" ".join(parts).split()) for parts in items]


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
    text = strip_footer(normalise(text))
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
    out["ecd_sg"] = _num(_label(text, "ECD"))
    out["bht_c"] = _num(_label(text, "Static BHT estimate"))
    out["mwd"] = _label(text, "MWD")
    out["productive_hours"] = _num(_label(text, "Productive time"))
    out["npt_hours"] = _num(_label(text, "Non-productive time"))

    m = re.search(r"Flow\s*:\s*.*?/\s*(\d+)\s*gpm", text, re.I)
    if m:
        out["flow_gpm"] = float(m.group(1))

    out["npt"] = parse_npt_blocks(text)
    return out


def parse_npt_blocks(text: str, labels: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Read the repeating entries of NPT DETAIL.

    Each entry starts at its Code line and needs an Hours line. Depth and
    Formation are optional (None when the entry has no such line).
    A line that is not a label line continues the value above it.
    """
    labels = labels or NPT_ENTRY_LABELS
    text = strip_footer(normalise(text))
    start = re.search(r"^NPT DETAIL\s*$", text, re.M)
    if not start:
        return []
    tail = text[start.end():]
    stop = next((m for m in re.finditer(r"^\S.*$", tail, re.M) if _HEADING.match(m.group(0))), None)
    if stop:
        tail = tail[: stop.start()]

    by_label = {v.lower(): k for k, v in labels.items()}
    label_re = re.compile(
        r"^[ \t]{0,4}(" + "|".join(re.escape(v) for v in labels.values()) + r")[ \t]*:[ \t]*(.*)$", re.I
    )
    raw_entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    last_key: str | None = None
    for line in tail.splitlines():
        if not line.strip():
            last_key = None
            continue
        m = label_re.match(line)
        if m:
            key = by_label[m.group(1).lower()]
            if key == "code":
                current = {}
                raw_entries.append(current)
            if current is None:
                continue
            current[key] = m.group(2).strip()
            last_key = key
        elif current is not None and last_key is not None:
            current[last_key] = f"{current[last_key]} {line.strip()}".strip()

    events: list[dict[str, Any]] = []
    for raw in raw_entries:
        code = raw.get("code", "").strip().upper()
        hours = _num(raw.get("hours"))
        if not code or hours is None:
            continue
        if code not in NPT_CODES:
            code = "OTHER"
        formation = raw.get("formation")
        if formation is not None:
            formation = re.sub(r"\s*\(.*\)\s*$", "", formation).strip() or None
        events.append({
            "code": code,
            "hours": hours,
            "depth_m": _num(raw.get("depth")),
            "formation": formation,
            "description": " ".join(raw.get("description", "").split()),
        })
    return events


def parse_eowr(text: str) -> dict[str, Any]:
    """End of well report: totals plus the numbered lessons."""
    text = strip_footer(normalise(text))
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

    block = re.search(r"4\. LESSONS LEARNED(.*?)(?:\n5\.|\Z)", text, re.S)
    out["lessons"] = _items(block.group(1), _NUMBERED_ITEM) if block else []

    block = re.search(r"5\. RECOMMENDATIONS FOR FUTURE WELLS(.*?)(?:\n\d+\.\s|\Z)", text, re.S)
    out["recommendations"] = _items(block.group(1), _BULLET_ITEM) if block else []
    return out


def parse_incident(text: str) -> dict[str, Any]:
    text = strip_footer(normalise(text))
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
    block = re.search(r"5\. CORRECTIVE ACTIONS(.*?)(?:\n\d+\.\s|\Z)", text, re.S)
    out["corrective_actions"] = _items(block.group(1), _BULLET_ITEM) if block else []
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
    """Turn one document into zero or more NPT rows.

    Depth and formation come from the entry itself when it carries them, and
    from the report header (depth at end, formation at TD) otherwise.
    """
    if doc.doc_type != "ddr":
        return []
    p = parse_ddr(doc.text)
    events: list[NptEvent] = []
    for e in p.get("npt", []):
        depth = e.get("depth_m")
        if depth is None:
            depth = p.get("depth_end_m") or 0.0
        events.append(NptEvent(
            doc_id=doc.doc_id,
            well=p.get("well") or doc.well,
            field_name=p.get("field_name") or doc.field_name,
            date=p.get("date") or doc.date,
            code=e["code"],
            hours=e["hours"],
            hole_section=p.get("hole_section") or "",
            formation=e.get("formation") or p.get("formation") or "",
            depth_m=depth,
            mud_weight_sg=p.get("mud_weight_sg") or 0.0,
            rig=p.get("rig") or "",
            description=e["description"],
        ))
    return events
