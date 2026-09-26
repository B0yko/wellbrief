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
from collections.abc import Mapping, Sequence
from typing import Any

from .config import (
    DEFAULT_DDR_LABELS,
    DEFAULT_EOWR_SECTIONS,
    DEFAULT_INCIDENT_SECTIONS,
    NPT_CODES,
    SYNTHETIC_FOOTER,
)
from .models import Document, NptEvent
from .text import canonical_section, normalise

_LABEL = r"[ \t]*{}[ \t]*:[ \t]*(.+)"

# The five fields of one NPT DETAIL entry, as keys of `DEFAULT_DDR_LABELS` (`npt_code` etc.);
# `_npt_entry_labels` turns the `[parse.ddr.labels]` shape into the flat shape
# `parse_npt_blocks` (and, before configuration, this module alone) works with.
_NPT_ENTRY_FIELDS = ("code", "hours", "depth", "formation", "description")

# A section heading: a line in capitals at the left margin.
_HEADING = re.compile(r"^[A-Z][A-Z0-9 ()/&.-]*$")
_NUMBERED_ITEM = re.compile(r"^[ \t]{0,3}\d+\.[ \t]+")
_BULLET_ITEM = re.compile(r"^[ \t]{0,3}[-*][ \t]+")


def _npt_entry_labels(ddr_labels: Mapping[str, str]) -> dict[str, str]:
    return {field: ddr_labels[f"npt_{field}"] for field in _NPT_ENTRY_FIELDS}


_CODE_CLEAN = re.compile(r"[^A-Za-z0-9]+")


def clean_code_key(raw: str) -> str:
    """A code (a parsed NPT entry's, a site's alias key, or a CSV ledger row's), folded to a
    canonical lookup key: non-alphanumeric runs become one underscore, upper-cased."""
    return _CODE_CLEAN.sub("_", raw.strip()).strip("_").upper()


def resolve_npt_code(raw: str, aliases: Mapping[str, str] | None = None) -> str:
    """A site's own spelling of an NPT code, resolved to the built-in taxonomy: the taxonomy
    code itself (however it is punctuated or cased), else a `[taxonomy.aliases]` entry (matched
    the same folded way, so `"DH-TOOL"` and `dh_tool` both find an alias keyed `"DH-TOOL"`), else
    `"OTHER"`, the catch-all every alias table still falls back to."""
    cleaned = clean_code_key(raw)
    if cleaned in NPT_CODES:
        return cleaned
    if aliases:
        by_key = {clean_code_key(k): v for k, v in aliases.items()}
        if cleaned in by_key:
            return by_key[cleaned]
    return "OTHER"


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
    m = re.search(_LABEL.format(re.escape(label)), text, re.I)
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


def parse_ddr(text: str, labels: Mapping[str, str] | None = None,
             aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Pull the numeric spine out of a daily drilling report.

    `labels` (`[parse.ddr.labels]`, default `DEFAULT_DDR_LABELS`) is the label text each header
    field and NPT entry field is read under; `aliases` (`[taxonomy.aliases]`) resolves a site's
    own NPT code spelling to the built-in taxonomy (`resolve_npt_code`).
    """
    labels = labels or DEFAULT_DDR_LABELS
    text = strip_footer(normalise(text))
    out = parse_header(text)

    out["depth_start_m"] = _num(_label(text, labels["depth_at_start"]))
    out["depth_end_m"] = _num(_label(text, labels["depth_at_end"]))
    out["progress_m"] = _num(_label(text, labels["progress"]))

    raw_section = _label(text, labels["hole_section"])
    out["hole_section"] = canonical_section(raw_section) if raw_section else None

    formation = _label(text, labels["formation_at_td"])
    if formation:
        out["formation"] = re.sub(r"\s*\(.*\)\s*$", "", formation).strip()

    out["mud_weight_sg"] = _num(_label(text, labels["weight"]))
    out["ecd_sg"] = _num(_label(text, labels["ecd"]))
    out["bht_c"] = _num(_label(text, labels["bht"]))
    out["mwd"] = _label(text, labels["mwd"])
    out["productive_hours"] = _num(_label(text, labels["productive_time"]))
    out["npt_hours"] = _num(_label(text, labels["non_productive_time"]))

    m = re.search(r"Flow\s*:\s*.*?/\s*(\d+)\s*gpm", text, re.I)
    if m:
        out["flow_gpm"] = float(m.group(1))

    out["npt"] = parse_npt_blocks(text, labels=_npt_entry_labels(labels), aliases=aliases)
    return out


def parse_npt_blocks(text: str, labels: Mapping[str, str] | None = None,
                     aliases: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Read the repeating entries of NPT DETAIL.

    Each entry starts at its Code line and needs an Hours line. Depth and
    Formation are optional (None when the entry has no such line).
    A line that is not a label line continues the value above it. `labels`
    maps the flat entry fields (`code`, `hours`, `depth`, `formation`,
    `description`) to their label text; `aliases` resolves a site's own NPT
    code spelling (`resolve_npt_code`).
    """
    labels = labels or _npt_entry_labels(DEFAULT_DDR_LABELS)
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
        raw_code = raw.get("code", "")
        hours = _num(raw.get("hours"))
        if not raw_code.strip() or hours is None:
            continue
        code = resolve_npt_code(raw_code, aliases)
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


def _section_blocks(text: str, headings: Sequence[str]) -> dict[str, str]:
    """Split `text` into the blocks that start at each of `headings`.

    A heading may carry its canonical-template leading counter ("3. NPT BREAKDOWN BY CODE")
    or be written bare ("NPT BREAKDOWN BY CODE"), on a line of its own. A block runs from just
    after its heading line to whichever other configured heading is found next in `text`
    (headings need not appear in the order `headings` lists them), or to the end of `text` for
    the last one. A heading not found in `text` has no entry in the result.
    """
    positions: list[tuple[int, int, str]] = []
    for heading in headings:
        m = re.search(rf"^(?:\d+\.\s*)?{re.escape(heading)}[ \t]*$", text, re.M | re.I)
        if m:
            positions.append((m.start(), m.end(), heading))
    positions.sort()
    blocks: dict[str, str] = {}
    for i, (_, end, heading) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        blocks[heading] = text[end:stop]
    return blocks


def parse_eowr(text: str, sections: Mapping[str, str] | None = None) -> dict[str, Any]:
    """End of well report: totals plus the numbered lessons.

    `sections` (`[parse.eowr.sections]`: `npt_by_code`, `lessons`, `recommendations`) is the
    heading text each block starts at, default `DEFAULT_EOWR_SECTIONS`.
    """
    sections = sections or DEFAULT_EOWR_SECTIONS
    text = strip_footer(normalise(text))
    out = parse_header(text)
    out["days_on_well"] = _num(_label(text, "Days on well"))
    out["td_m"] = _num(_label(text, "Total depth"))
    out["total_npt_hours"] = _num(_label(text, "Non-productive time"))

    blocks = _section_blocks(text, (sections["npt_by_code"], sections["lessons"],
                                    sections["recommendations"]))

    by_code: dict[str, float] = {}
    block = blocks.get(sections["npt_by_code"])
    if block:
        for m in re.finditer(r"^\s*([A-Z_]{3,})\s+([\d.]+)\s*h\s*$", block, re.M):
            by_code[m.group(1)] = float(m.group(2))
    out["npt_by_code"] = by_code

    lessons = blocks.get(sections["lessons"])
    out["lessons"] = _items(lessons, _NUMBERED_ITEM) if lessons else []

    recommendations = blocks.get(sections["recommendations"])
    out["recommendations"] = _items(recommendations, _BULLET_ITEM) if recommendations else []
    return out


def parse_incident(text: str, sections: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`sections` (`[parse.incident.sections]`: `root_cause`, `corrective_actions`), default
    `DEFAULT_INCIDENT_SECTIONS`."""
    sections = sections or DEFAULT_INCIDENT_SECTIONS
    text = strip_footer(normalise(text))
    out = parse_header(text)
    m = re.search(r"Classification:\s*([A-Z_]+)\s+Lost time:\s*([\d.]+)", text, re.I)
    if m:
        out["code"], out["hours"] = m.group(1), float(m.group(2))
    m = re.search(r"Depth:\s*([\d,]+)\s*m MD\s+"
                  r"Hole section:\s*(\S+(?:\s+\d/\d\")?)\s+Formation:\s*(.+)", text)
    if m:
        out["depth_m"] = _num(m.group(1))
        out["hole_section"] = canonical_section(m.group(2))
        out["formation"] = m.group(3).strip()
    m = re.search(r"Severity:\s*(\w+)", text, re.I)
    if m:
        out["severity"] = m.group(1)

    blocks = _section_blocks(text, (sections["root_cause"], sections["corrective_actions"]))
    root_cause = blocks.get(sections["root_cause"])
    if root_cause:
        out["root_cause"] = " ".join(root_cause.split())
    actions = blocks.get(sections["corrective_actions"])
    out["corrective_actions"] = _items(actions, _BULLET_ITEM) if actions else []
    return out


def parse(doc: Document, ddr_labels: Mapping[str, str] | None = None,
         eowr_sections: Mapping[str, str] | None = None,
         incident_sections: Mapping[str, str] | None = None,
         aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    if doc.doc_type == "ddr":
        return parse_ddr(doc.text, labels=ddr_labels, aliases=aliases)
    if doc.doc_type == "eowr":
        return parse_eowr(doc.text, sections=eowr_sections)
    if doc.doc_type == "incident":
        return parse_incident(doc.text, sections=incident_sections)
    return parse_header(doc.text)


def npt_events(doc: Document, ddr_labels: Mapping[str, str] | None = None,
              aliases: Mapping[str, str] | None = None) -> list[NptEvent]:
    """Turn one document into zero or more NPT rows.

    Depth and formation come from the entry itself when it carries them, and
    from the report header (depth at end, formation at TD) otherwise.
    """
    if doc.doc_type != "ddr":
        return []
    p = parse_ddr(doc.text, labels=ddr_labels, aliases=aliases)
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
            mwd=p.get("mwd") or "",
        ))
    return events
