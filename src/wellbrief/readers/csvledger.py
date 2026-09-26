"""CSV reader for NPT ledgers.

An NPT ledger has one row per non-productive-time event. The reader maps the file's headers to
logical columns, validates each row and keeps the exact text of the row so that it can later be
stored as a one-line citable document.

Logical columns:

* required: ``well``, ``date``, ``code``, ``hours``;
* optional: ``field``, ``depth``, ``section``, ``formation``, ``rig``, ``description``.

``columns`` maps a logical column to the header used in the file; a logical column that is not
mapped is looked up under its own name. Header matching ignores case and collapses whitespace. Two
logical columns may not resolve to the same header. An optional column that ``columns`` maps
explicitly but the file lacks is reported in :attr:`LedgerRead.warnings`.

Values are normalised as follows:

* ``date`` becomes ``YYYY-MM-DD``. Without ``date_format`` the value must be an ISO date,
  optionally followed by a time (``2026-03-04``, ``2026-03-04T06:00``). With ``date_format`` the
  value is parsed with :func:`time.strptime` using that format only; the format must give the year,
  the month and the day (or the year and the day of the year).
* ``hours`` and ``depth`` are numbers. ``12,5`` and ``0,750`` use a decimal comma; ``1.234,5`` is
  understood; a comma followed by groups of exactly three digits after a non-zero leading group is
  a thousands separator (``2,650``, ``12,345.5``). The one ambiguous form, a single comma followed
  by exactly three digits (``1,250``), is read as a thousands separator in ``depth``. In ``hours``
  it is read as a decimal comma when the delimiter is ``;`` (the list separator of locales that
  write decimal commas) and rejected as ambiguous when the delimiter is ``,``. ``hours`` may carry
  an ``h``/``hr``/``hrs``/``hours`` suffix; ``depth`` an ``m`` suffix optionally followed by a
  depth reference such as ``MD``, ``MDRT``, ``MDBRT``, ``TVD``, ``TVDSS`` or ``RT``. Depths in
  other units are not converted.
* Text values are stripped; an empty optional value becomes ``None``.

Each record is one physical line. A quoted value may continue onto the following lines, but only
when the quote is closed properly and none of the lines it would take in reads as a ledger row of
its own (values in every required column and a valid date). Otherwise the quote is treated as
unterminated: the first line is reported and reading resumes on the next line, so a stray quote
cannot swallow the rows after it.

A row that lacks a required value, has a required value that cannot be parsed, has malformed
quoting, or has more non-empty values than the header has columns is skipped and reported in
:attr:`LedgerRead.errors` as one message per skipped row (``"line 7: missing hours"``). A ``depth``
that cannot be read is left empty and reported in :attr:`LedgerRead.warnings`; the row is kept, so
that a unit typo does not remove NPT hours from the ledger. Blank rows are ignored. Problems with
the file as a whole (encoding, a header without the required columns) raise
:class:`~wellbrief.readers.ReaderError`.

The file must be UTF-8; a byte order mark is tolerated. The delimiter is ``,`` or ``;``: an Excel
``sep=`` first line decides it when present, otherwise the delimiter that occurs more often
outside quotes in the header line is tried first and the other one is used only if the first
does not yield all required columns. The header is a single physical line.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import ReaderError
from .txt import decode_text

__all__ = [
    "DELIMITERS",
    "LOGICAL_COLUMNS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "LedgerRead",
    "LedgerRow",
    "read_ledger",
]

REQUIRED_COLUMNS = ("well", "date", "code", "hours")
OPTIONAL_COLUMNS = ("field", "depth", "section", "formation", "rig", "description")
LOGICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
DELIMITERS = (",", ";")

_SEP_LINE = re.compile(r"sep=(.)", re.IGNORECASE)
_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")
_HOURS = re.compile(r"(?P<number>[+-]?[\d.,]+)\s*(?:h|hr|hrs|hours?)?", re.IGNORECASE)
_DEPTH = re.compile(
    r"(?P<number>[+-]?[\d.,]+)\s*(?:m|metres|meters)?"
    r"(?:\s*(?:MD|MDRT|MDBRT|MDDF|MDKB|TVD|TVDRT|TVDBRT|TVDSS|RT|BRT|DF|KB))?",
    re.IGNORECASE,
)
_ONE_THOUSANDS_GROUP = re.compile(r"[+-]?[1-9]\d{0,2},\d{3}")
_THOUSANDS_COMMA = re.compile(r"[+-]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?")
_THOUSANDS_DOT_DECIMAL_COMMA = re.compile(r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+,\d+")
_DECIMAL_COMMA = re.compile(r"[+-]?\d+,\d+")
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_STRPTIME_DIRECTIVE = re.compile(r"%(.)")


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One valid ledger row.

    Attributes:
        line_no: 1-based line number in the file where the row starts.
        raw_line: The exact text of the row as it appears in the file, without the line ending.
            A quoted value spanning several lines keeps its embedded newlines.
        well: Well name as written in the file (stripped).
        date: Event date as ``YYYY-MM-DD``.
        code: NPT code as written in the file (stripped); mapping to the taxonomy happens later.
        hours: Duration in hours.
        field: Field name, if given.
        depth_m: Depth in metres, if given and readable.
        section: Hole section, if given.
        formation: Formation, if given.
        rig: Rig, if given.
        description: Free-text description, if given.
    """

    line_no: int
    raw_line: str
    well: str
    date: str
    code: str
    hours: float
    field: str | None = None
    depth_m: float | None = None
    section: str | None = None
    formation: str | None = None
    rig: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerRead:
    """Result of reading a ledger.

    Attributes:
        rows: The valid rows, in file order.
        errors: One message per skipped row, in file order (``"line 7: missing hours"``).
        warnings: Problems that did not cost a row: an explicitly mapped optional column missing
            from the header, or an optional value that was left empty because it could not be read.
    """

    rows: list[LedgerRow]
    errors: list[str]
    warnings: list[str] = dataclasses.field(default_factory=list)


def read_ledger(
    path: str | os.PathLike[str],
    columns: Mapping[str, str] | None = None,
    date_format: str | None = None,
) -> LedgerRead:
    """Read an NPT ledger CSV file.

    Args:
        path: The ``.csv`` file.
        columns: Logical column name to header name in the file. Unmapped logical columns are
            looked up under their own name.
        date_format: ``strptime`` format of the date column; ISO dates are expected when omitted.

    Raises:
        ValueError: If ``columns`` names an unknown logical column, maps one to an empty header or
            maps two to the same header, or if ``date_format`` does not give a full date.
        ReaderError: If the file is not UTF-8, is empty, or its header is not valid CSV, lacks a
            required column or repeats a mapped one.
        OSError: If the file cannot be read.
    """
    explicit = dict(columns or {})
    mapping = _validate_mapping(explicit)
    _validate_date_format(date_format)
    lines = decode_text(Path(path).read_bytes()).split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    start = _first_content_line(lines, 0)
    if start is None:
        raise ReaderError("CSV file is empty")
    forced = _SEP_LINE.fullmatch(lines[start].strip())
    if forced:
        if forced.group(1) not in DELIMITERS:
            raise ReaderError(f"unsupported CSV delimiter {forced.group(1)!r} in the sep= line")
        header_at = _first_content_line(lines, start + 1)
        if header_at is None:
            raise ReaderError("CSV file has no header line")
        candidates: Sequence[str] = (forced.group(1),)
    else:
        header_at = start
        candidates = _delimiter_candidates(lines[start])

    delimiter, header_size, resolved = _choose_delimiter(lines, header_at, candidates, mapping)
    warnings = [
        f"column {logical!r} is mapped to header {mapping[logical]!r}, which the file does not have; "
        f"{logical} is left empty"
        for logical in OPTIONAL_COLUMNS
        if logical in explicit and logical not in resolved
    ]
    parser = _RowParser(resolved, header_size, delimiter, date_format)
    rows: list[LedgerRow] = []
    errors: list[str] = []
    index = header_at + 1
    while index < len(lines):
        record = _read_record(lines, index, delimiter, parser.reads_as_row)
        index = record.end
        line_no = record.start + 1
        if record.problem is not None:
            errors.append(f"line {line_no}: {record.problem}")
            continue
        if not any(cell.strip() for cell in record.cells):
            continue
        raw_line = "\n".join(lines[record.start : record.end])
        row, problems, notes = parser.build(record.cells, line_no, raw_line)
        warnings.extend(f"line {line_no}: {note}" for note in notes)
        if row is None:
            errors.append(f"line {line_no}: " + "; ".join(problems))
        else:
            rows.append(row)
    return LedgerRead(rows=rows, errors=errors, warnings=warnings)


@dataclass(frozen=True, slots=True)
class _Record:
    """One CSV record: lines ``start`` to ``end - 1`` (0-based), or the reason it was rejected."""

    start: int
    end: int
    cells: list[str]
    problem: str | None = None


class _LineFeed:
    """Feeds physical lines to :mod:`csv` and counts how many one record consumed."""

    def __init__(self, lines: Sequence[str], start: int) -> None:
        self._lines = lines
        self._next = start
        self.consumed = 0
        self.exhausted = False

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._next >= len(self._lines):
            self.exhausted = True
            raise StopIteration
        line = self._lines[self._next]
        self._next += 1
        self.consumed += 1
        return line + "\n"


def _read_record(
    lines: Sequence[str],
    start: int,
    delimiter: str,
    reads_as_row: Callable[[str], bool] | None,
    end_of_input: str = "the end of the file",
) -> _Record:
    """Read the record that starts on line ``start``.

    The :mod:`csv` module asks for another line only while a quoted value is open. Such a record is
    accepted only if the value closes properly (strict parsing) before the end of the file and no
    line it takes in reads as a ledger row (``reads_as_row`` may be ``None`` when ``lines`` holds a
    single line). A rejected record covers its first line only, so the caller resumes on the next
    line.
    """
    feed = _LineFeed(lines, start)
    failure: csv.Error | None = None
    cells: list[str] = []
    try:
        cells = next(csv.reader(feed, delimiter=delimiter, strict=True))
    except csv.Error as exc:
        failure = exc
    end = start + feed.consumed
    if end - start > 1 or feed.exhausted:
        for index in range(start + 1, end):
            if reads_as_row is not None and reads_as_row(lines[index]):
                return _Record(
                    start,
                    start + 1,
                    [],
                    f"unterminated quoted value: it would take in line {index + 1}, "
                    "which is a ledger row of its own",
                )
        if feed.exhausted:
            return _Record(
                start, start + 1, [], f"unterminated quoted value: it is still open at {end_of_input}"
            )
        if failure is not None:
            return _Record(start, start + 1, [], f"unterminated quoted value: {failure} on line {end}")
    elif failure is not None:
        return _Record(start, start + 1, [], f"invalid CSV: {failure}")
    return _Record(start, end, cells)


def _validate_mapping(columns: Mapping[str, str]) -> dict[str, str]:
    unknown = sorted(set(columns) - set(LOGICAL_COLUMNS))
    if unknown:
        raise ValueError(
            f"unknown ledger column(s): {', '.join(unknown)}; expected {', '.join(LOGICAL_COLUMNS)}"
        )
    empty = sorted(name for name, header in columns.items() if not header.strip())
    if empty:
        raise ValueError(f"ledger column(s) mapped to an empty header: {', '.join(empty)}")
    mapping = {name: columns.get(name, name) for name in LOGICAL_COLUMNS}
    by_header: dict[str, list[str]] = {}
    for name in LOGICAL_COLUMNS:
        by_header.setdefault(_normalise_header(mapping[name]), []).append(name)
    for names in by_header.values():
        if len(names) > 1:
            raise ValueError(
                f"ledger columns {' and '.join(names)} resolve to the same header {mapping[names[0]]!r}"
            )
    return mapping


def _validate_date_format(date_format: str | None) -> None:
    if not date_format:
        return
    directives = set(_STRPTIME_DIRECTIVE.findall(date_format.replace("%%", "")))
    has_year = bool(directives & {"Y", "y"})
    has_day = ("d" in directives and bool(directives & {"m", "b", "B"})) or "j" in directives
    if not (has_year and has_day):
        raise ValueError(
            f"date_format {date_format!r} must give the year (%Y or %y) and the day: the month "
            "(%m, %b or %B) with the day of the month (%d), or the day of the year (%j)"
        )


def _first_content_line(lines: Sequence[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _delimiter_candidates(header_line: str) -> list[str]:
    counts = {delimiter: _count_outside_quotes(header_line, delimiter) for delimiter in DELIMITERS}
    return sorted(DELIMITERS, key=lambda delimiter: -counts[delimiter])


def _count_outside_quotes(line: str, char: str) -> int:
    count = 0
    quoted = False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == char and not quoted:
            count += 1
    return count


def _choose_delimiter(
    lines: Sequence[str], header_at: int, candidates: Sequence[str], mapping: Mapping[str, str]
) -> tuple[str, int, dict[str, int]]:
    """Return the delimiter, the number of header cells and the resolved column positions."""
    header_line = [lines[header_at]]
    first_problems: list[str] = []
    for delimiter in candidates:
        record = _read_record(header_line, 0, delimiter, None, "the end of the header line")
        if record.problem is not None:
            problems = [record.problem]
        else:
            resolved, problems = _resolve_header(record.cells, mapping)
            if not problems:
                return delimiter, len(record.cells), resolved
        if not first_problems:
            first_problems = problems
    raise ReaderError(f"line {header_at + 1}: CSV header: " + "; ".join(first_problems))


def _normalise_header(name: str) -> str:
    return " ".join(name.split()).casefold()


def _resolve_header(header: Sequence[str], mapping: Mapping[str, str]) -> tuple[dict[str, int], list[str]]:
    positions: dict[str, list[int]] = {}
    for index, name in enumerate(header):
        positions.setdefault(_normalise_header(name), []).append(index)
    resolved: dict[str, int] = {}
    problems: list[str] = []
    for logical in LOGICAL_COLUMNS:
        header_name = mapping[logical]
        found = positions.get(_normalise_header(header_name), [])
        if len(found) > 1:
            problems.append(f"column {header_name!r} appears {len(found)} times")
        elif found:
            resolved[logical] = found[0]
        elif logical in REQUIRED_COLUMNS:
            problems.append(f"missing required column {logical!r} (header {header_name!r})")
    return resolved, problems


class _RowParser:
    """Turns the cells of one record into a :class:`LedgerRow` for a resolved header."""

    def __init__(
        self, resolved: Mapping[str, int], header_size: int, delimiter: str, date_format: str | None
    ) -> None:
        self._resolved = resolved
        self._header_size = header_size
        self._delimiter = delimiter
        self._date_format = date_format

    def _value(self, cells: Sequence[str], logical: str) -> str:
        index = self._resolved.get(logical)
        return cells[index].strip() if index is not None and index < len(cells) else ""

    def reads_as_row(self, line: str) -> bool:
        """Return whether ``line`` on its own has every required value and a valid date."""
        try:
            cells = next(csv.reader([line], delimiter=self._delimiter))
        except csv.Error:
            return False
        if not all(self._value(cells, logical) for logical in REQUIRED_COLUMNS):
            return False
        return _parse_date(self._value(cells, "date"), self._date_format) is not None

    def build(
        self, cells: Sequence[str], line_no: int, raw_line: str
    ) -> tuple[LedgerRow | None, list[str], list[str]]:
        """Return the row (or ``None``), the problems that skip it and the notes that do not."""

        def value(logical: str) -> str:
            return self._value(cells, logical)

        problems: list[str] = []
        notes: list[str] = []
        if len(cells) > self._header_size and any(cell.strip() for cell in cells[self._header_size :]):
            problems.append(f"{len(cells)} values but the header has {self._header_size} columns")
        for logical in REQUIRED_COLUMNS:
            if not value(logical):
                problems.append(f"missing {logical}")

        iso_date: str | None = None
        if value("date"):
            iso_date = _parse_date(value("date"), self._date_format)
            if iso_date is None:
                expected = (
                    f"in the format {self._date_format!r}"
                    if self._date_format
                    else "an ISO date (YYYY-MM-DD)"
                )
                problems.append(f"date {value('date')!r} is not {expected}")

        hours: float | None = None
        if value("hours"):
            hours, problem = self._parse_hours(value("hours"))
            if problem:
                problems.append(problem)

        depth: float | None = None
        if value("depth"):
            depth = _parse_measure(value("depth"), _DEPTH)
            if depth is None:
                notes.append(f"depth {value('depth')!r} is not a depth in metres; depth left empty")
            elif depth < 0:
                notes.append(f"depth {value('depth')!r} is negative; depth left empty")
                depth = None

        if problems or iso_date is None or hours is None:
            return None, problems, notes
        row = LedgerRow(
            line_no=line_no,
            raw_line=raw_line,
            well=value("well"),
            date=iso_date,
            code=value("code"),
            hours=hours,
            field=value("field") or None,
            depth_m=depth,
            section=value("section") or None,
            formation=value("formation") or None,
            rig=value("rig") or None,
            description=value("description") or None,
        )
        return row, [], notes

    def _parse_hours(self, text: str) -> tuple[float | None, str | None]:
        match = _HOURS.fullmatch(text)
        number = None if match is None else match.group("number")
        if number is not None and _ONE_THOUSANDS_GROUP.fullmatch(number):
            if self._delimiter != ";":
                return None, (
                    f"hours {text!r} is ambiguous (thousands separator or decimal comma); "
                    "write it without a thousands separator or with a decimal point"
                )
            hours: float | None = float(number.replace(",", "."))
        else:
            hours = None if number is None else _parse_number(number)
        if hours is None:
            return None, f"hours {text!r} is not a number of hours"
        if hours < 0:
            return None, f"hours {text!r} is negative"
        return hours, None


def _parse_date(text: str, date_format: str | None) -> str | None:
    if date_format:
        try:
            parsed = time.strptime(text, date_format)
        except ValueError:
            return None
        return date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday).isoformat()
    match = _ISO_DATE.fullmatch(text)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1)).isoformat()
    except ValueError:
        return None


def _parse_measure(text: str, pattern: re.Pattern[str]) -> float | None:
    match = pattern.fullmatch(text)
    return None if match is None else _parse_number(match.group("number"))


def _parse_number(text: str) -> float | None:
    if _THOUSANDS_COMMA.fullmatch(text):
        return float(text.replace(",", ""))
    if _THOUSANDS_DOT_DECIMAL_COMMA.fullmatch(text):
        return float(text.replace(".", "").replace(",", "."))
    if _DECIMAL_COMMA.fullmatch(text):
        return float(text.replace(",", "."))
    if _PLAIN_NUMBER.fullmatch(text):
        return float(text)
    return None
