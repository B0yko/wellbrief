"""Document type detection: content headings first, the filename as a fallback.

Ingestion needs to know what kind of report a file is before it can be parsed. The
canonical templates all start with an all-caps heading line, so that decides it whenever the
heading is present and recognised; a file whose heading is missing or unrecognised (a scan, a
report in an unlisted template) falls back to its filename's prefix (`DDR-`, `EOWR-`, `INC-`,
case-insensitive); anything else becomes `"other"`: searchable and citable, but never parsed for
NPT. `HEADINGS` (document type -> heading text) is the built-in default a site's own
`[detect.headings]` (`settings.DetectSettings`) overrides.
"""

from __future__ import annotations

from .config import DEFAULT_DETECT_HEADINGS as HEADINGS

# A filename prefix (before the first "-"), case-insensitive. Not configurable: a fallback for a
# heading that is missing or unrecognised, not a template convention a site would restate.
PREFIXES: dict[str, str] = {"DDR": "ddr", "EOWR": "eowr", "INC": "incident"}

# A heading only ever appears in the first few lines of a report (after that, blank lines and
# label lines start); scanning further would risk matching a quoted heading inside prose.
_HEADING_SCAN_LINES = 8


def detect_from_heading(text: str, headings: dict[str, str] | None = None) -> str | None:
    """The document type whose heading (`headings`: document type -> heading text, default
    `HEADINGS`) is one of the first `_HEADING_SCAN_LINES` lines of `text`; `None` when none
    of them matches."""
    by_heading = {v: k for k, v in (headings or HEADINGS).items()}
    for line in text.splitlines()[:_HEADING_SCAN_LINES]:
        doc_type = by_heading.get(line.strip())
        if doc_type:
            return doc_type
    return None


def detect_from_filename(stem: str) -> str | None:
    """The document type named by `stem`'s prefix (`DDR-ORD-101-001` -> `ddr`); `None` when the
    prefix is not one of `PREFIXES`."""
    prefix = stem.split("-", 1)[0].upper()
    return PREFIXES.get(prefix)


def detect_doc_type(text: str, stem: str, headings: dict[str, str] | None = None) -> str:
    """The document type of a file: its content heading, else its filename's prefix, else `"other"`."""
    return detect_from_heading(text, headings) or detect_from_filename(stem) or "other"
