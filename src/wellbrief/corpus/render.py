"""Turn generated records into the lines of a report.

Rendering stops at a list of lines. Writing them to a file is a separate
step (`writers.py`), so a plain-text, PDF or DOCX writer all receive exactly
the same lines.

Layout rules shared by every document:

- lines are at most 100 characters; longer prose is wrapped at word
  boundaries, and continuation lines are indented to the value column of a
  label line or to the text of a numbered or bulleted item, so a parser can
  join them back into one logical line;
- a number is never separated from its unit or from the fraction of a hole
  size (`2,660 m`, `1.45 sg`, `17.4 h`, `12 1/4"` stay on one line);
- the last line is the synthetic-document footer.
"""

from __future__ import annotations

import re

from ..config import SYNTHETIC_FOOTER
from .fields import MWD_VENDOR, OPERATOR
from .records import Activity, DayReport, Incident, WellRecord, fmt_m

WIDTH = 100
LABEL_WIDTH = 22
DAY_START_MINUTES = 6 * 60

_NUMBER = re.compile(r"^\(?\d[\d,.]*$")
# Units and hole-size fractions that must stay on the line of their number.
_UNIT = re.compile(r"^(?:m|MD|sg|C|h|bbl|bbl/h|gpm|t|rpm|cP|%|\d/\d\"?)[.,;:)]*$")
_KEEP = "\u00a0"   # a no-break space; never written to a document


def _tokens(text: str) -> list[str]:
    """Words, with a number and its unit (and "m MD") glued into one token."""
    out: list[str] = []
    for word in text.split():
        prev = out[-1] if out else ""
        last = prev.rsplit(_KEEP, 1)[-1]
        if _UNIT.match(word) and (_NUMBER.match(last) or (word.startswith("MD") and last == "m")):
            out[-1] = f"{prev}{_KEEP}{word}"
        else:
            out.append(word)
    return out


def wrap(prefix: str, text: str, indent: int) -> list[str]:
    """Wrap `prefix + text` at WIDTH; continuation lines start at column `indent`."""
    lines: list[str] = []
    current = prefix
    has_word = False
    for word in _tokens(text):
        candidate = f"{current} {word}" if has_word else current + word
        if len(candidate) <= WIDTH or not has_word:
            current = candidate
        else:
            lines.append(current)
            current = " " * indent + word
        has_word = True
    lines.append(current)
    return [line.replace(_KEEP, " ") for line in lines]


def label_line(label: str, value: str, indent: int = 2) -> list[str]:
    prefix = f"{' ' * indent}{label:<{LABEL_WIDTH}}: "
    return wrap(prefix, value, len(prefix))


def numbered(items: list[str], indent: int = 2) -> list[str]:
    out: list[str] = []
    for n, text in enumerate(items, start=1):
        prefix = f"{' ' * indent}{n}. "
        out += wrap(prefix, text, len(prefix))
    return out


def bullets(items: list[str], indent: int = 2) -> list[str]:
    out: list[str] = []
    for text in items:
        prefix = f"{' ' * indent}- "
        out += wrap(prefix, text, len(prefix))
    return out


def _clock(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def ops_log(activities: list[Activity]) -> list[str]:
    """Contiguous time ranges from 06:00 to 06:00; one tenth of an hour is 6 minutes."""
    out: list[str] = []
    start = DAY_START_MINUTES
    for act in activities:
        end = start + act.tenths * 6
        prefix = f"  {_clock(start)}-{_clock(end)}  "
        out += wrap(prefix, act.text, len(prefix))
        start = end
    return out


def render_ddr(well: WellRecord, day: DayReport) -> list[str]:
    spec = well.spec
    c = day.cosmetics
    npt = day.npt_tenths
    if day.drilled:
        ecd = f"{c['ecd']:.2f} sg"
        parameters = f"{c['wob']} t / {c['rpm']} rpm / {c['flow']} gpm"
    else:
        ecd = "-"
        parameters = "- / - / -"
    lines = [
        "DAILY DRILLING REPORT",
        f"Operator: {OPERATOR}    Field: {spec.name}    Well: {well.name}",
        f"Rig: {well.rig}    Report No: {day.report_no:03d}    Date: {day.day.isoformat()}",
        "Report period: 06:00 - 06:00",
        "",
        "DEPTH",
        *label_line("Depth at start", f"{fmt_m(day.depth_start)} m MD"),
        *label_line("Depth at end", f"{fmt_m(day.depth_end)} m MD"),
        *label_line("Progress", f"{fmt_m(day.depth_end - day.depth_start)} m"),
        *label_line("Hole section", day.section.size),
        *label_line("Formation at TD", f"{day.formation_end.name} ({day.formation_end.lithology})"),
        "",
        "MUD",
        *label_line("System", day.mud_system),
        *label_line("Weight", f"{day.mud_weight_sg:.2f} sg"),
        *label_line("PV / YP", f"{c['pv']} cP / {c['yp']} lb/100ft2"),
        *label_line("ECD", ecd),
        "",
        "BHA AND PARAMETERS",
        *label_line("Bit", str(c["bit"])),
        *label_line("MWD", f"{MWD_VENDOR} {day.mwd}"),
        *label_line("WOB / RPM / Flow", parameters),
        *label_line("Static BHT estimate", f"{day.bht_c} C"),
        "",
        "OPERATIONS SUMMARY (24 h)",
        *ops_log(day.activities),
        "",
        "TIME BREAKDOWN",
        *label_line("Productive time", f"{(240 - npt) / 10:.1f} h"),
        *label_line("Non-productive time", f"{npt / 10:.1f} h"),
        "",
        "NPT DETAIL",
    ]
    if day.entries:
        for e in day.entries:
            lines += label_line("Code", e.code)
            lines += label_line("Hours", f"{e.hours:.1f}")
            lines += label_line("Depth", f"{fmt_m(e.depth_m)} m MD")
            lines += label_line("Formation", e.formation)
            lines += label_line("Description", e.description)
            lines.append("")
    else:
        lines += ["  None reported this period.", ""]
    lines.append("REMARKS")
    for remark in day.remarks:
        lines += wrap("  ", remark, 4)
    lines += ["", SYNTHETIC_FOOTER]
    return lines


def _mwd_summary(well: WellRecord) -> str:
    tools = well.mwd_by_section
    sizes = list(tools)
    if len(set(tools.values())) == 1:
        return f"MWD: {MWD_VENDOR} {tools[sizes[0]]} in every section."
    parts = []
    for tool in dict.fromkeys(tools.values()):
        run = [s for s in sizes if tools[s] == tool]
        where = f"{run[0]} section" if len(run) == 1 else f"{run[0]} to {run[-1]} sections"
        parts.append(f"{tool} in the {where}")
    return f"MWD: {MWD_VENDOR} " + " and ".join(parts) + "."


def _mud_summary(well: WellRecord) -> str:
    by_system: dict[str, list[str]] = {}
    for s in well.spec.sections:
        by_system.setdefault(s.mud_system, []).append(s.size)
    parts = [f"{system.lower() if system[1].islower() else system} in {' and '.join(sizes)}"
             for system, sizes in by_system.items()]
    return "Mud systems: " + ", ".join(parts) + "."


def render_eowr(well: WellRecord) -> list[str]:
    spec = well.spec
    days = len(well.days)
    by_code: dict[str, int] = {}
    for e in well.entries:
        by_code[e.code] = by_code.get(e.code, 0) + e.tenths
    total = sum(by_code.values())
    hours_on_well = days * 24
    shoes = [s.base_m for s in spec.sections[:-1]] + [well.td_m]
    casing = ", ".join(f"{s.casing} at {fmt_m(d)} m" for s, d in zip(spec.sections, shoes, strict=True))
    lines = [
        "END OF WELL REPORT",
        f"Operator: {OPERATOR}    Field: {spec.name}    Well: {well.name}",
        f"Rig: {well.rig}    Spud: {well.spud.isoformat()}    Report date: {well.eowr_date.isoformat()}",
        f"Days on well: {days}    Total depth: {fmt_m(well.td_m)} m MD",
        "",
        "1. WELL SUMMARY",
        *wrap("  ", f"{well.name} was drilled as a development well on {spec.name} to {fmt_m(well.td_m)} m MD.", 4),
        *wrap("  ", "Four hole sections were drilled: " + ", ".join(s.size for s in spec.sections) + ".", 4),
        *wrap("  ", f"Casing: {casing}.", 4),
        *wrap("  ", _mud_summary(well), 4),
        *wrap("  ", _mwd_summary(well), 4),
        "",
        "2. TIME ANALYSIS",
        *label_line("Total time on well", f"{hours_on_well:.1f} h"),
        *label_line("Non-productive time", f"{total / 10:.1f} h ({total / 10 / hours_on_well * 100:.1f} %)"),
        "",
        "3. NPT BREAKDOWN BY CODE",
    ]
    if by_code:
        for code, tenths in sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {code:<24} {tenths / 10:>7.1f} h")
    else:
        lines.append("  None recorded.")
    lines += ["", "4. LESSONS LEARNED", *numbered([s.text for s in well.lessons])]
    lines += ["", "5. RECOMMENDATIONS FOR FUTURE WELLS", *bullets([s.text for s in well.recommendations])]
    lines += ["", SYNTHETIC_FOOTER]
    return lines


def render_incident(well: WellRecord, inc: Incident) -> list[str]:
    spec = well.spec
    event = inc.event
    start = well.report(inc.reports[0])
    reports = inc.reports
    source = (f"Daily report: {reports[0]}" if len(reports) == 1
              else f"Daily reports: {', '.join(reports)}")
    whole = event.text("whole", event.tenths)
    return [
        "WELL OPERATIONS INCIDENT REPORT",
        f"Operator: {OPERATOR}    Field: {spec.name}    Well: {well.name}",
        f"Rig: {well.rig}    Date: {start.day.isoformat()}    Severity: {inc.severity}",
        f"Classification: {event.code}    Lost time: {event.hours:.1f} h",
        *wrap("", source, 2),
        "",
        "1. LOCATION",
        f"  Depth: {fmt_m(event.depth_m)} m MD    Hole section: {start.section.size}    Formation: {event.formation}",
        f"  Formation temperature at depth (estimated): {spec.bht_c(event.depth_m)} C",
        "",
        "2. SEQUENCE OF EVENTS",
        *wrap("  ", whole, 2),
        *wrap("  ", f"Lost time {event.hours:.1f} h. No injuries and no environmental release.", 2),
        "",
        "3. IMMEDIATE CAUSE",
        *wrap("  ", whole.split(". ")[0].rstrip(".") + ".", 2),
        "",
        "4. ROOT CAUSE",
        *wrap("  ", inc.root_cause, 2),
        "",
        "5. CORRECTIVE ACTIONS",
        *bullets([a.text for a in inc.actions]),
        "",
        SYNTHETIC_FOOTER,
    ]
