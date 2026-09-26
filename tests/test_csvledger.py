"""Tests for the NPT ledger CSV reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief.readers import ReaderError
from wellbrief.readers.csvledger import LedgerRow, read_ledger

SITE_COLUMNS = {
    "well": "Well Name",
    "date": "Event Date",
    "code": "NPT Code",
    "hours": "Duration",
    "depth": "Depth (m)",
    "section": "Hole Section",
    "formation": "Formation",
    "description": "Comment",
    "rig": "Rig",
}

SITE_LEDGER_LINES = [
    "Well Name;Event Date;NPT Code;Duration;Depth (m);Hole Section;Formation;Comment;Rig",
    (
        'VSS-207;11.04.2026;LOST_CIRCULATION;31,5;2,662 m MD;"8 1/2""";Vessra Carbonate;'
        '"Total losses; LCM pill spotted";Vessra-3'
    ),
    'VSS-207;12.04.2026;LOST_CIRCULATION;;2,662;"8 1/2""";Vessra Carbonate;Partial losses;Vessra-3',
    "VSS-209;2026-04-30;RIG_REPAIR;6,0;;;;Mud pump fluid end changed;Vessra-3",
    "",
    'ORD-105;18.03.2026;STUCK_PIPE;7,5;1336 m;"17 1/2""";Keldra Salt;"String packed off',
    'while pulling out of hole";Orrin-1',
    (
        "  ORD-105 ; 18.03.2026 ; WELLBORE_INSTABILITY ; 1,5 h ; 1,360 ; 17 1/2 in ; Keldra Salt ; "
        "Tight hole ; Orrin-1"
    ),
    "ORD-106;19.03.2026;;4;;;;;",
    "ORD-106;20.03.2026;WEATHER;abc;;;;;",
    "ORD-106;21.03.2026;WEATHER;-2;;;;;",
    "ORD-106;22.03.2026;WEATHER;3;2650 ft;;;;",
    "ORD-106;23.03.2026;WEATHER;3;;;;;Orrin-2;extra",
    "ORD-106;24.03.2026;WEATHER;3;;;;;Orrin-2;;",
    ";;;;;;;;",
    ";31.02.2026;WEATHER;3;;;;;",
]


def _write(tmp_path: Path, lines: list[str], *, newline: str = "\n", bom: bool = False) -> Path:
    path = tmp_path / "npt-ledger.csv"
    data = newline.join(lines) + newline
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data.encode("utf-8"))
    return path


# The format required by the ledger integration: non-default layout, ";" delimiter --------------------


def test_site_layout_with_semicolons(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, SITE_LEDGER_LINES), columns=SITE_COLUMNS, date_format="%d.%m.%Y")

    assert result.rows == [
        LedgerRow(
            line_no=2,
            raw_line=SITE_LEDGER_LINES[1],
            well="VSS-207",
            date="2026-04-11",
            code="LOST_CIRCULATION",
            hours=31.5,
            field=None,
            depth_m=2662.0,
            section='8 1/2"',
            formation="Vessra Carbonate",
            rig="Vessra-3",
            description="Total losses; LCM pill spotted",
        ),
        LedgerRow(
            line_no=6,
            raw_line=SITE_LEDGER_LINES[5] + "\n" + SITE_LEDGER_LINES[6],
            well="ORD-105",
            date="2026-03-18",
            code="STUCK_PIPE",
            hours=7.5,
            depth_m=1336.0,
            section='17 1/2"',
            formation="Keldra Salt",
            rig="Orrin-1",
            description="String packed off\nwhile pulling out of hole",
        ),
        LedgerRow(
            line_no=8,
            raw_line=SITE_LEDGER_LINES[7],
            well="ORD-105",
            date="2026-03-18",
            code="WELLBORE_INSTABILITY",
            hours=1.5,
            depth_m=1360.0,
            section="17 1/2 in",
            formation="Keldra Salt",
            rig="Orrin-1",
            description="Tight hole",
        ),
        LedgerRow(
            line_no=12,
            raw_line=SITE_LEDGER_LINES[11],
            well="ORD-106",
            date="2026-03-22",
            code="WEATHER",
            hours=3.0,
            depth_m=None,
        ),
        LedgerRow(
            line_no=14,
            raw_line=SITE_LEDGER_LINES[13],
            well="ORD-106",
            date="2026-03-24",
            code="WEATHER",
            hours=3.0,
            rig="Orrin-2",
        ),
    ]
    assert result.errors == [
        "line 3: missing hours",
        "line 4: date '2026-04-30' is not in the format '%d.%m.%Y'",
        "line 9: missing code",
        "line 10: hours 'abc' is not a number of hours",
        "line 11: hours '-2' is negative",
        "line 13: 10 values but the header has 9 columns",
        "line 16: missing well; date '31.02.2026' is not in the format '%d.%m.%Y'",
    ]
    assert result.warnings == ["line 12: depth '2650 ft' is not a depth in metres; depth left empty"]


def test_raw_line_is_the_exact_source_text(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, SITE_LEDGER_LINES), columns=SITE_COLUMNS, date_format="%d.%m.%Y")
    source = (tmp_path / "npt-ledger.csv").read_text(encoding="utf-8")
    for row in result.rows:
        assert row.raw_line in source
        assert source.split("\n")[row.line_no - 1] == row.raw_line.split("\n")[0]
    assert result.rows[2].raw_line.startswith("  ORD-105 ; 18.03.2026 ;")


# Defaults, encodings and delimiter detection ------------------------------------------------------


def test_default_columns_match_headers_case_insensitively_with_bom_and_crlf(tmp_path: Path) -> None:
    lines = [
        "WELL,Date,  code ,Hours,Depth,Field,Description,Section,Formation,RIG",
        (
            'ORD-101,2026-01-05,RIG_REPAIR,3.5,"2,650",Orrindale,"Pump liner, washed out",'
            "12 1/4,Dovrin Shale,Orrin-1"
        ),
        "ORD-102,2026-01-06T06:00,MATERIALS,1,,Orrindale,,,,",
    ]
    result = read_ledger(_write(tmp_path, lines, newline="\r\n", bom=True))

    assert result.errors == []
    first, second = result.rows
    assert first == LedgerRow(
        line_no=2,
        raw_line=lines[1],
        well="ORD-101",
        date="2026-01-05",
        code="RIG_REPAIR",
        hours=3.5,
        field="Orrindale",
        depth_m=2650.0,
        section="12 1/4",
        formation="Dovrin Shale",
        rig="Orrin-1",
        description="Pump liner, washed out",
    )
    assert (second.date, second.hours, second.depth_m, second.description) == ("2026-01-06", 1.0, None, None)


def test_excel_sep_line_sets_the_delimiter(tmp_path: Path) -> None:
    lines = ["sep=;", "well;date;code;hours", "ORD-101;2026-01-05;RIG_REPAIR;2,5"]
    (row,) = read_ledger(_write(tmp_path, lines)).rows
    assert (row.line_no, row.hours) == (3, 2.5)


def test_other_delimiter_is_tried_when_the_first_does_not_fit(tmp_path: Path) -> None:
    lines = ["well,date,code,hours,note;a;b;c;d;e", "ORD-101,2026-01-05,RIG_REPAIR,3.5,x;y"]
    (row,) = read_ledger(_write(tmp_path, lines)).rows
    assert row.hours == 3.5


def test_delimiters_inside_quotes_are_not_counted(tmp_path: Path) -> None:
    lines = ['well;date;code;hours;"notes, comments, extra"', 'ORD-101;2026-01-05;RIG_REPAIR;3;"a, b"']
    (row,) = read_ledger(_write(tmp_path, lines)).rows
    assert row.raw_line == lines[1]


def test_blank_lines_are_ignored_and_line_numbers_stay_physical(tmp_path: Path) -> None:
    lines = [
        "",
        "well,date,code,hours",
        "",
        "ORD-101,2026-01-05,RIG_REPAIR,3",
        "   ",
        "ORD-101,2026-01-06,X,1",
    ]
    result = read_ledger(_write(tmp_path, lines))
    assert [row.line_no for row in result.rows] == [4, 6]
    assert result.errors == []


def test_header_only_file_has_no_rows(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, ["well,date,code,hours"]))
    assert (result.rows, result.errors) == ([], [])


# Value parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2,650", 2650.0),
        ("2650 m", 2650.0),
        ("2,650 m MD", 2650.0),
        ("2650m", 2650.0),
        ("1,402 m TVD", 1402.0),
        ("12,345.5", 12345.5),
        ("1.234,5", 1234.5),
        ("2650,5", 2650.5),
        ("2650.25 metres", 2650.25),
        ("  980  ", 980.0),
        ("1,402 m MDRT", 1402.0),
        ("1,402 m MDBRT", 1402.0),
        ("1402 m RT", 1402.0),
        ("1402mTVDSS", 1402.0),
        ("0,500", 0.5),
        ("0,5", 0.5),
        ("1,250", 1250.0),
    ],
)
def test_depth_values(text: str, expected: float, tmp_path: Path) -> None:
    lines = ["well;date;code;hours;depth", f'ORD-101;2026-01-05;X;1;"{text}"']
    (row,) = read_ledger(_write(tmp_path, lines)).rows
    assert row.depth_m == expected


def test_negative_depth_is_left_empty_with_a_warning(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, ["well,date,code,hours,depth", "W-1,2026-01-05,X,1,-5 m"]))
    (row,) = result.rows
    assert (row.hours, row.depth_m, result.errors) == (1.0, None, [])
    assert result.warnings == ["line 2: depth '-5 m' is negative; depth left empty"]


@pytest.mark.parametrize("text", ["2650 ft", "2,65,0", "about 2650", "m", "1e3", "1,402 m below RT"])
def test_unreadable_depth_keeps_the_row_and_its_hours(text: str, tmp_path: Path) -> None:
    lines = ["well;date;code;hours;depth", f"ORD-101;2026-01-05;X;7,5;{text}"]
    result = read_ledger(_write(tmp_path, lines))
    (row,) = result.rows
    assert (row.hours, row.depth_m, result.errors) == (7.5, None, [])
    assert result.warnings == [f"line 2: depth {text!r} is not a depth in metres; depth left empty"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7.5", 7.5),
        ("7,5", 7.5),
        ("7.5h", 7.5),
        ("7.5 hrs", 7.5),
        ("12 hours", 12.0),
        ("0", 0.0),
        (".5", 0.5),
        ("0,750", 0.75),
        ("1,250", 1.25),
        ("12,345", 12.345),
        ("1,250 h", 1.25),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("1,234,567", 1234567.0),
    ],
)
def test_hours_values(text: str, expected: float, tmp_path: Path) -> None:
    lines = ["well;date;code;hours", f"ORD-101;2026-01-05;X;{text}"]
    (row,) = read_ledger(_write(tmp_path, lines)).rows
    assert row.hours == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("7,5", 7.5), ("0,750", 0.75), ("12,345.5", 12345.5), ("1,234,567", 1234567.0), ("1.234,5", 1234.5)],
)
def test_hours_values_in_a_comma_delimited_file(text: str, expected: float, tmp_path: Path) -> None:
    (row,) = read_ledger(_write(tmp_path, ["well,date,code,hours", f'W-1,2026-01-05,X,"{text}"'])).rows
    assert row.hours == expected


@pytest.mark.parametrize("text", ["1,250", "12,345 h", "-1,250"])
def test_single_thousands_group_in_hours_is_ambiguous_in_a_comma_delimited_file(
    text: str, tmp_path: Path
) -> None:
    result = read_ledger(_write(tmp_path, ["well,date,code,hours", f'W-1,2026-01-05,X,"{text}"']))
    assert result.rows == []
    assert result.errors == [
        (
            f"line 2: hours {text!r} is ambiguous (thousands separator or decimal comma); "
            "write it without a thousands separator or with a decimal point"
        )
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2026-03-04", "2026-03-04"), ("2026-03-04 06:00", "2026-03-04"), ("2026-03-04T06:00:30", "2026-03-04")],
)
def test_iso_dates(text: str, expected: str, tmp_path: Path) -> None:
    (row,) = read_ledger(_write(tmp_path, ["well,date,code,hours", f"W-1,{text},X,1"])).rows
    assert row.date == expected


@pytest.mark.parametrize("text", ["04.03.2026", "2026-02-30", "20260304", "2026-W10-3", "yesterday"])
def test_invalid_iso_dates(text: str, tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, ["well,date,code,hours", f"W-1,{text},X,1"]))
    assert result.errors == [f"line 2: date {text!r} is not an ISO date (YYYY-MM-DD)"]


def test_custom_date_format(tmp_path: Path) -> None:
    lines = ["well,date,code,hours", "W-1,03/04/2026,X,1"]
    (row,) = read_ledger(_write(tmp_path, lines), date_format="%m/%d/%Y").rows
    assert row.date == "2026-03-04"


# File-level errors --------------------------------------------------------------------------------


def test_missing_required_column(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="missing required column 'hours' \\(header 'Duration'\\)"):
        read_ledger(_write(tmp_path, ["well,date,code,hours"]), columns={"hours": "Duration"})


def test_duplicate_header(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="'well' appears 2 times"):
        read_ledger(_write(tmp_path, ["well,date,code,hours,Well"]))


@pytest.mark.parametrize("lines", [[], ["", "  ", ""]])
def test_empty_file(lines: list[str], tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="empty"):
        read_ledger(_write(tmp_path, lines))


def test_sep_line_without_header(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="no header"):
        read_ledger(_write(tmp_path, ["sep=;", ""]))


def test_unsupported_sep_delimiter(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="unsupported CSV delimiter"):
        read_ledger(_write(tmp_path, ["sep=|", "well|date|code|hours"]))


def test_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    path.write_bytes(b"well,date,code,hours\nORD-101,2026-01-05,X,1,caf\xe9\n")
    with pytest.raises(ReaderError, match="not valid UTF-8"):
        read_ledger(path)


def test_header_with_an_unterminated_quote(tmp_path: Path) -> None:
    lines = ['well,date,code,hours,"description', "W-1,2026-01-05,X,1,ok"]
    with pytest.raises(ReaderError, match="line 1: CSV header: unterminated quoted value"):
        read_ledger(_write(tmp_path, lines))


def test_quoted_header_names(tmp_path: Path) -> None:
    lines = ['"Well";"Date";"Code";"Hours"', '"W-1";"2026-01-05";"X";"2,5"']
    (row,) = read_ledger(_write(tmp_path, lines)).rows
    assert (row.well, row.hours) == ("W-1", 2.5)


def test_unknown_logical_column_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown ledger column"):
        read_ledger(_write(tmp_path, ["well,date,code,hours"]), columns={"duration": "hours"})


def test_empty_header_mapping_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty header"):
        read_ledger(_write(tmp_path, ["well,date,code,hours"]), columns={"hours": "  "})


@pytest.mark.parametrize(
    "columns",
    [{"well": "Well", "field": "Well"}, {"well": "WELL NAME", "field": " well  name"}, {"well": "field"}],
)
def test_two_columns_mapped_to_one_header_is_a_configuration_error(
    columns: dict[str, str], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="resolve to the same header"):
        read_ledger(_write(tmp_path, ["Well,date,code,hours,field"]), columns=columns)


@pytest.mark.parametrize("date_format", ["%d.%m", "%m/%Y", "%Y-%m", "%d.%m.%%Y", "%x", "%H:%M"])
def test_date_format_without_a_full_date_is_a_configuration_error(date_format: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must give the year"):
        read_ledger(_write(tmp_path, ["well,date,code,hours"]), date_format=date_format)


@pytest.mark.parametrize(
    ("date_format", "text"),
    [
        ("%Y-%j", "2026-063"),
        ("%d %b %Y", "04 Mar 2026"),
        ("%B %d, %y", "March 04, 26"),
        ("%d%%%m%%%Y", "04%03%2026"),
    ],
)
def test_accepted_date_formats(date_format: str, text: str, tmp_path: Path) -> None:
    lines = ["well;date;code;hours", f"W-1;{text};X;1"]
    (row,) = read_ledger(_write(tmp_path, lines), date_format=date_format).rows
    assert row.date == "2026-03-04"


def test_explicitly_mapped_optional_column_missing_from_the_file_is_a_warning(tmp_path: Path) -> None:
    lines = ["well,date,code,hours,Depth [m]", "W-1,2026-01-05,X,1,2650"]
    result = read_ledger(_write(tmp_path, lines), columns={"depth": "Depth (m)", "rig": "Rig"})
    (row,) = result.rows
    assert (row.hours, row.depth_m, row.rig, result.errors) == (1.0, None, None, [])
    assert result.warnings == [
        "column 'depth' is mapped to header 'Depth (m)', which the file does not have; depth is left empty",
        "column 'rig' is mapped to header 'Rig', which the file does not have; rig is left empty",
    ]


def test_unmapped_optional_columns_may_be_absent_without_a_warning(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, ["well,date,code,hours", "W-1,2026-01-05,X,1"]))
    assert (len(result.rows), result.errors, result.warnings) == (1, [], [])


# Quoting: a stray quote must never swallow the rows after it ---------------------------------------

LEDGER_HEADER = "well;date;code;hours;description"
STUCK = 'ORD-105;2026-03-18;STUCK_PIPE;7,5;"String packed off at 1,336 m'
FOLLOWING_ROWS = [
    "ORD-105;2026-03-19;FISHING;12,0;Fished string",
    "ORD-106;2026-03-20;RIG_REPAIR;6,0;Top drive repair",
    "ORD-107;2026-03-21;WEATHER;4,0;Crane stop",
]


def test_unterminated_quote_does_not_swallow_the_following_rows(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, [LEDGER_HEADER, STUCK, *FOLLOWING_ROWS]))
    assert [(row.line_no, row.code, row.hours) for row in result.rows] == [
        (3, "FISHING", 12.0),
        (4, "RIG_REPAIR", 6.0),
        (5, "WEATHER", 4.0),
    ]
    assert [row.raw_line for row in result.rows] == FOLLOWING_ROWS
    assert result.errors == [
        "line 2: unterminated quoted value: it would take in line 3, which is a ledger row of its own"
    ]


def test_unterminated_quote_closed_by_a_later_quoted_value(tmp_path: Path) -> None:
    fishing = 'ORD-105;2026-03-19;FISHING;12,0;"Fished string"'
    result = read_ledger(_write(tmp_path, [LEDGER_HEADER, STUCK, fishing, *FOLLOWING_ROWS[1:]]))
    assert [(row.line_no, row.code) for row in result.rows] == [
        (3, "FISHING"),
        (4, "RIG_REPAIR"),
        (5, "WEATHER"),
    ]
    assert result.rows[0].description == "Fished string"
    assert result.errors == [
        "line 2: unterminated quoted value: it would take in line 3, which is a ledger row of its own"
    ]


def test_unterminated_quote_closed_cleanly_by_an_inch_mark_in_a_later_row(tmp_path: Path) -> None:
    # Strict CSV parsing alone accepts this record: the inch mark after 8 1/2 closes the stray quote and
    # is followed by the delimiter. Only the check on the lines taken in catches it.
    lines = [
        "well;date;code;hours;depth;section;formation;description;rig",
        'VSS-207;2026-04-11;LOST_CIRCULATION;31,5;2662;"8 1/2;Vessra Carbonate;Total losses;Vessra-3',
        'VSS-207;2026-04-12;LOST_CIRCULATION;4;2670;8 1/2";Vessra Carbonate;Partial losses;Vessra-3',
    ]
    result = read_ledger(_write(tmp_path, lines))
    assert [(row.line_no, row.hours, row.section) for row in result.rows] == [(3, 4.0, '8 1/2"')]
    assert result.errors == [
        "line 2: unterminated quoted value: it would take in line 3, which is a ledger row of its own"
    ]


def test_unterminated_quote_on_the_last_line(tmp_path: Path) -> None:
    result = read_ledger(_write(tmp_path, [LEDGER_HEADER, *FOLLOWING_ROWS, STUCK]))
    assert [row.line_no for row in result.rows] == [2, 3, 4]
    assert result.errors == ["line 5: unterminated quoted value: it is still open at the end of the file"]


def test_unterminated_quote_followed_by_free_text_lines(tmp_path: Path) -> None:
    lines = [LEDGER_HEADER, FOLLOWING_ROWS[0], STUCK, "while pulling out of hole", ""]
    result = read_ledger(_write(tmp_path, lines))
    assert [row.line_no for row in result.rows] == [2]
    assert result.errors == [
        "line 3: unterminated quoted value: it is still open at the end of the file",
        "line 4: missing date; missing code; missing hours",
    ]


def test_quoted_value_closed_with_trailing_text(tmp_path: Path) -> None:
    lines = [
        "well,date,code,hours,description",
        'W-1,2026-01-05,X,1,"first part',
        'second part" trailing',
        "W-2,2026-01-06,X,2,ok",
    ]
    result = read_ledger(_write(tmp_path, lines))
    assert [row.line_no for row in result.rows] == [4]
    assert result.errors == [
        "line 2: unterminated quoted value: ',' expected after '\"' on line 3",
        "line 3: missing date; missing code; missing hours",
    ]


def test_malformed_quoting_on_one_line_is_a_row_error(tmp_path: Path) -> None:
    lines = ["well,date,code,hours,description", 'W-1,2026-01-05,X,1,"quoted" tail', "W-2,2026-01-06,X,2,ok"]
    result = read_ledger(_write(tmp_path, lines))
    assert [row.line_no for row in result.rows] == [3]
    assert result.errors == ["line 2: invalid CSV: ',' expected after '\"'"]


def test_oversized_values_are_row_errors(tmp_path: Path) -> None:
    lines = [
        "well,date,code,hours,description",
        "W-1,2026-01-05,X,1,ok",
        'W-1,2026-01-05,X,1,"open',
        "y" * 200_000,
        "W-2,2026-01-06,X,2,ok",
    ]
    result = read_ledger(_write(tmp_path, lines))
    assert [row.line_no for row in result.rows] == [2, 5]
    assert result.errors == [
        "line 3: unterminated quoted value: field larger than field limit (131072) on line 4",
        "line 4: invalid CSV: field larger than field limit (131072)",
    ]


def test_well_formed_multi_line_values_are_kept(tmp_path: Path) -> None:
    lines = [
        "well,date,code,hours,description,rig",
        'W-1,2026-01-05,X,1,"Pump liner washed out,',
        "",
        'changed liner, tested to 5,000 psi",Orrin-1',
        "W-2,2026-01-06,X,2,ok,Orrin-1",
    ]
    result = read_ledger(_write(tmp_path, lines))
    assert result.errors == []
    first, second = result.rows
    assert (first.line_no, first.rig) == (2, "Orrin-1")
    assert first.description == "Pump liner washed out,\n\nchanged liner, tested to 5,000 psi"
    assert first.raw_line == "\n".join(lines[1:4])
    assert second.line_no == 5
