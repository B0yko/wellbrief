"""Document type detection: content headings first, the filename's prefix as a fallback, else "other"."""

from __future__ import annotations

from wellbrief.detect import detect_doc_type, detect_from_filename, detect_from_heading


def test_detects_a_daily_drilling_report_from_its_heading() -> None:
    text = "DAILY DRILLING REPORT\nOperator: Quillfen Energy    Field: Orrindale    Well: ORD-101\n"
    assert detect_from_heading(text) == "ddr"
    assert detect_doc_type(text, "whatever") == "ddr"


def test_detects_an_end_of_well_report_from_its_heading() -> None:
    text = "END OF WELL REPORT\nOperator: Quillfen Energy    Field: Orrindale    Well: ORD-101\n"
    assert detect_from_heading(text) == "eowr"


def test_detects_an_incident_report_from_its_heading() -> None:
    text = "WELL OPERATIONS INCIDENT REPORT\nOperator: Quillfen Energy    Field: Orrindale    Well: ORD-101\n"
    assert detect_from_heading(text) == "incident"


def test_a_heading_only_counts_within_the_first_few_lines() -> None:
    text = "\n".join(["filler"] * 10 + ["DAILY DRILLING REPORT"])
    assert detect_from_heading(text) is None


def test_falls_back_to_the_filename_prefix_when_no_heading_matches() -> None:
    text = "Some other document with no recognised heading.\n"
    assert detect_from_heading(text) is None
    assert detect_from_filename("DDR-ORD-101-001") == "ddr"
    assert detect_from_filename("EOWR-ORD-101") == "eowr"
    assert detect_from_filename("INC-ORD-101-01") == "incident"
    assert detect_doc_type(text, "DDR-ORD-101-001") == "ddr"


def test_the_filename_prefix_match_is_case_insensitive() -> None:
    assert detect_from_filename("ddr-ord-101-001") == "ddr"
    assert detect_from_filename("Inc-ord-101-01") == "incident"


def test_unrecognised_becomes_other() -> None:
    assert detect_from_filename("field-notes-2026") is None
    assert detect_doc_type("Just some prose.\n", "field-notes-2026") == "other"


def test_content_heading_wins_over_a_misleading_filename_prefix() -> None:
    text = "END OF WELL REPORT\nOperator: Quillfen Energy    Field: Orrindale    Well: ORD-101\n"
    assert detect_doc_type(text, "DDR-mislabelled") == "eowr"
