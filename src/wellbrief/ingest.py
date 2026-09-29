"""Loading documents into the store.

`load_corpus_dir` / `ingest` read a folder of plain-text well files in the template `parse.py`
understands (the generator's default "txt" rendering writes exactly that template) and are kept
for the tests and tools that only ever need that one, simple case.

`ingest_folder` is the general entry point: a folder of `.txt`, `.md`, `.pdf`,
`.docx` and `.csv` files, read incrementally by sha256 (unchanged files are skipped; a changed
file replaces its document(s), chunks and events), with content-heading type detection, a
coverage report, `--prune` and `--dry-run`. It is what `wellbrief ingest` and the evaluation
adapter's `Workspace` use.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any

from . import parse
from .detect import detect_doc_type
from .models import Document, FileRecord, NptEvent, Well
from .readers import ReaderError
from .readers.csvledger import read_ledger
from .readers.docx import read_docx
from .readers.pdf import read_pdf
from .readers.txt import read_text
from .settings import Settings, resolve_settings
from .store import Store, chunk_document

DOC_TYPE_BY_PREFIX = {"DDR": "ddr", "EOWR": "eowr", "INC": "incident"}


def infer_doc_type(name: str) -> str:
    prefix = name.split("-", 1)[0].upper()
    return DOC_TYPE_BY_PREFIX.get(prefix, "ddr")


def load_corpus_dir(path: Path | str) -> list[Document]:
    """Read a folder of plain-text well files.

    Files whose name starts with an underscore are not documents (the corpus
    generator's ground-truth sidecar is one) and are never ingested.
    """
    path = Path(path)
    docs: list[Document] = []
    for file in sorted(path.glob("*.txt")):
        if file.name.startswith("_"):
            continue
        text = file.read_text(encoding="utf-8")
        header = parse.parse_header(text)
        docs.append(Document(
            doc_id=file.stem,
            doc_type=infer_doc_type(file.stem),
            well=header.get("well", "UNKNOWN"),
            field_name=header.get("field_name", "UNKNOWN"),
            date=header.get("date", "1970-01-01"),
            title=text.splitlines()[0].strip() if text.strip() else file.stem,
            text=text,
            meta={},
            source=file.name,
        ))
    return docs


def ingest(store: Store, docs: list[Document], wells: list[Well] | None = None) -> dict[str, int]:
    """Parse, store (which also chunks -- see `Store.put_documents`) and record.

    Metadata that arrives with a document is not trusted: it is recomputed
    from the text, because metadata exported from another system can be
    missing or disagree with the report it describes. Indexing (per-field
    BM25 and hashing vectors) is a separate, workspace-level step; see
    `workspace.ensure_field_indexes` and the CLI's `index` command.
    """
    parsed_docs: list[Document] = []
    all_events = []
    for doc in docs:
        meta = parse.parse(doc)
        doc.meta = {k: v for k, v in meta.items() if v is not None}
        if not doc.well or doc.well == "UNKNOWN":
            doc.well = meta.get("well", doc.well)
        if not doc.field_name or doc.field_name == "UNKNOWN":
            doc.field_name = meta.get("field_name", doc.field_name)
        parsed_docs.append(doc)
        all_events.extend(parse.npt_events(doc))

    store.put_documents(parsed_docs)
    if wells:
        store.put_wells(wells)
    else:
        store.put_wells(_wells_from_docs(parsed_docs))
    store.put_npt(all_events)
    store.put_files(_file_records(parsed_docs))
    chunks = sum(len(chunk_document(d.doc_id, d.text)) for d in parsed_docs)
    return {"documents": len(parsed_docs), "npt_events": len(all_events), "chunks": chunks}


def _file_records(docs: list[Document]) -> list[FileRecord]:
    """One `files` row per ingested document, keyed by its source path.

    `sha256`/`size` are computed from the document's (already decoded) text
    rather than re-reading the file, which is exact for the UTF-8, BOM-less
    text this simple loader produces. `ingest_folder` computes both from the
    raw file bytes instead, and that hash is what its incremental skip keys on.
    `mtime` is left at 0.0 here because nothing reads it.
    """
    return [
        FileRecord(
            path=d.source or d.doc_id, sha256=hashlib.sha256(d.text.encode("utf-8")).hexdigest(),
            size=len(d.text.encode("utf-8")), mtime=0.0, doc_ids=[d.doc_id], status="ingested",
            reason=None, field_name=d.field_name,
        )
        for d in docs
    ]


def _wells_from_docs(docs: list[Document]) -> list[Well]:
    """Rebuild a well register from the documents when no master well list is supplied."""
    by_well: dict[str, dict[str, Any]] = {}
    for d in docs:
        entry = by_well.setdefault(d.well, {
            "field": d.field_name, "rig": "", "spud": d.date, "td": 0.0,
            "sections": set(), "formations": set(),
        })
        entry["spud"] = min(entry["spud"], d.date)
        entry["rig"] = entry["rig"] or (d.meta or {}).get("rig", "")
        meta = d.meta or {}
        if meta.get("hole_section"):
            entry["sections"].add(meta["hole_section"])
        if meta.get("formation"):
            entry["formations"].add(meta["formation"])
        for key in ("depth_end_m", "td_m", "depth_m"):
            if meta.get(key):
                entry["td"] = max(entry["td"], float(meta[key]))
    order = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']
    return [
        Well(
            name=name,
            field_name=e["field"],
            rig=e["rig"],
            spud_date=e["spud"],
            td_m=e["td"],
            sections=sorted(e["sections"], key=lambda s: order.index(s) if s in order else 99),
            formations=sorted(e["formations"]),
        )
        for name, e in sorted(by_well.items())
    ]


# ==========================================================================
# General ingestion: `.txt`/`.md`/`.pdf`/`.docx`/`.csv`, incremental by sha256,
# type and field detection, a coverage report, `--prune` and `--dry-run`.
# ==========================================================================

UNASSIGNED_FIELD = "unassigned"
CSV_DOC_TYPE = "csv"

# The DDR/EOWR/incident readers handled by `_read_document_file`; a CSV ledger is handled on its own
# (`_ingest_csv_files`), since one file becomes many one-line documents, not one.
_READ_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
_CSV_EXTENSION = ".csv"

# Two NPT rows (a CSV ledger row and a DDR NPT entry) describe the same event when their well,
# date and code agree and their hours agree within this many hours.
_DEDUP_HOURS_TOLERANCE = 0.05

# The `[Depth, Formation, Weight, ...]` attributes the coverage table's per-DDR extraction share
# is reported for.
_EXTRACTION_ATTRIBUTES = ("depth", "section", "formation", "mud_weight", "npt_blocks")


@dataclass(frozen=True)
class SkippedFile:
    """One line of the coverage table's "skipped" column."""

    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason}


@dataclass
class Coverage:
    """What one `ingest_folder` run did or (`dry_run=True`) would do.

    `ddr_extraction` is the share (0.0-1.0) of the daily reports *ingested this run* whose
    depth, hole section, formation, mud weight and NPT DETAIL block were each found by the
    parser; `None` for an attribute when no DDR was processed this run (nothing to share).
    """

    files_seen: int = 0
    ingested: int = 0
    unchanged: int = 0
    pruned: int = 0
    duplicates_merged: int = 0
    documents: int = 0
    npt_events: int = 0
    chunks: int = 0
    skipped: list[SkippedFile] = _field(default_factory=list)
    csv_row_errors: list[str] = _field(default_factory=list)
    csv_row_warnings: list[str] = _field(default_factory=list)
    fields: set[str] = _field(default_factory=set)
    ddr_extraction: dict[str, float | None] = _field(
        default_factory=lambda: dict.fromkeys(_EXTRACTION_ATTRIBUTES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen, "ingested": self.ingested, "unchanged": self.unchanged,
            "pruned": self.pruned, "duplicates_merged": self.duplicates_merged,
            "documents": self.documents, "npt_events": self.npt_events, "chunks": self.chunks,
            "skipped": [s.to_dict() for s in self.skipped],
            "csv_row_errors": list(self.csv_row_errors), "csv_row_warnings": list(self.csv_row_warnings),
            "fields": sorted(self.fields), "ddr_extraction": dict(self.ddr_extraction),
        }


def _read_document_file(path: Path) -> tuple[str, list[int] | None]:
    """`(text, page_map)` for one `.txt`/`.md`/`.pdf`/`.docx` file; `page_map` is set only for
    a PDF (cumulative end-offset of each page's own text in the joined `text`, so `quotes.quote_page`
    can map a quote back to the page it came from)."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return read_text(path), None
    if suffix == ".pdf":
        pages = read_pdf(path)
        texts = [p.text for p in pages]
        page_map: list[int] = []
        offset = 0
        for t in texts:
            offset += len(t)
            page_map.append(offset)
            offset += 1  # the "\n" the pages are joined with
        return "\n".join(texts), page_map
    if suffix == ".docx":
        return read_docx(path), None
    raise ValueError(f"_read_document_file does not handle {suffix!r} files")  # pragma: no cover


def _doc_id_for(rel_path: str, stem: str, *, collides: bool) -> str:
    """The document id for a whole file: its stem, plus a short hash of its path when another
    file in the same ingest shares that stem."""
    if not collides:
        return stem
    digest = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def _build_document(text: str, page_map: list[int] | None, doc_id: str, rel_path: str,
                    field: str | None, settings: Settings) -> Document:
    """One non-CSV document: type from its content heading (else its filename, else `"other"`),
    field from its header (else `--field`, else `"unassigned"`), and its full parsed metadata."""
    doc_type = detect_doc_type(text, Path(rel_path).stem, settings.detect.headings)
    header = parse.parse_header(text)
    title = next((s.strip() for s in text.splitlines() if s.strip()), doc_id)
    doc = Document(
        doc_id=doc_id, doc_type=doc_type, well=header.get("well") or "UNKNOWN",
        field_name=header.get("field_name") or field or UNASSIGNED_FIELD,
        date=header.get("date") or "1970-01-01", title=title, text=text, meta={},
        source=rel_path, page_map=page_map,
    )
    meta = parse.parse(doc, ddr_labels=settings.parse.ddr_labels, eowr_sections=settings.parse.eowr_sections,
                       incident_sections=settings.parse.incident_sections, aliases=settings.taxonomy.aliases)
    doc.meta = {k: v for k, v in meta.items() if v is not None}
    return doc


def _ddr_extraction_share(parsed: list[dict[str, Any]], texts: list[str]) -> dict[str, float | None]:
    if not parsed:
        return dict.fromkeys(_EXTRACTION_ATTRIBUTES)
    n = len(parsed)
    return {
        "depth": sum(1 for p in parsed if p.get("depth_end_m") is not None) / n,
        "section": sum(1 for p in parsed if p.get("hole_section")) / n,
        "formation": sum(1 for p in parsed if p.get("formation")) / n,
        "mud_weight": sum(1 for p in parsed if p.get("mud_weight_sg") is not None) / n,
        "npt_blocks": sum(1 for t in texts if re.search(r"^NPT DETAIL\s*$", t, re.M)) / n,
    }


def _hours_match(candidates: list[float], target: float) -> bool:
    return any(abs(h - target) <= _DEDUP_HOURS_TOLERANCE for h in candidates)


def _replace_stale_doc_ids(store: Store, existing: FileRecord | None, new_ids: set[str]) -> None:
    """A changed file's previous document id(s) that are no longer produced (a CSV ledger that
    lost rows, or a stem-collision hash that started or stopped applying) do not get cleaned up
    by `put_documents`/`put_npt`/`put_chunks` alone, since those only ever replace a document that
    is still produced under the same id."""
    if existing is None:
        return
    stale = set(existing.doc_ids) - new_ids
    if stale:
        store.delete_documents(stale)


def _ingest_regular_files(
    store: Store, changed: list[tuple[Path, str, str, str, int]], field: str | None, dry_run: bool,
    existing_files: dict[str, FileRecord], coverage: Coverage, settings: Settings,
    parse_config: dict[str, Any],
) -> list[NptEvent]:
    """Read, detect, parse and (unless `dry_run`) store every changed `.txt`/`.md`/`.pdf`/`.docx`
    file. Returns the DDR events built this run (also used by the CSV pass's duplicate check)."""
    docs: list[Document] = []
    ddr_parsed: list[dict[str, Any]] = []
    ddr_texts: list[str] = []
    for path, rel, sha, doc_id, size in changed:
        try:
            text, page_map = _read_document_file(path)
        except ReaderError as exc:
            coverage.skipped.append(SkippedFile(rel, str(exc)))
            if not dry_run:
                # A file that used to parse and no longer does (rather than one that never
                # has) must not leave its previous document sitting in the store unnoticed.
                _replace_stale_doc_ids(store, existing_files.get(rel), set())
                store.put_files([FileRecord(
                    path=rel, sha256=sha, size=size, mtime=0.0, doc_ids=[], status="skipped",
                    reason=str(exc), field_name="", parse_config=parse_config,
                )])
            continue
        doc = _build_document(text, page_map, doc_id, rel, field, settings)
        docs.append(doc)
        coverage.fields.add(doc.field_name)
        if doc.doc_type == "ddr":
            ddr_parsed.append(parse.parse_ddr(text, labels=settings.parse.ddr_labels,
                                              aliases=settings.taxonomy.aliases))
            ddr_texts.append(text)
        if not dry_run:
            _replace_stale_doc_ids(store, existing_files.get(rel), {doc_id})
            store.put_files([FileRecord(
                path=rel, sha256=sha, size=size, mtime=0.0,
                doc_ids=[doc_id], status="ingested", reason=None, field_name=doc.field_name,
                parse_config=parse_config,
            )])

    events = [e for doc in docs
             for e in parse.npt_events(doc, ddr_labels=settings.parse.ddr_labels,
                                       aliases=settings.taxonomy.aliases)]
    if not dry_run:
        if docs:
            store.put_documents(docs)
        if events:
            store.put_npt(events)
    coverage.ingested += len(docs)
    coverage.documents += len(docs)
    coverage.npt_events += len(events)
    coverage.chunks += sum(len(chunk_document(d.doc_id, d.text)) for d in docs)
    coverage.ddr_extraction = _ddr_extraction_share(ddr_parsed, ddr_texts)
    return events


def _ingest_csv_files(
    store: Store, changed: list[tuple[Path, str, str, str, int]], field: str | None, dry_run: bool,
    existing_files: dict[str, FileRecord], ddr_events: list[NptEvent], coverage: Coverage,
    settings: Settings, parse_config: dict[str, Any],
) -> None:
    """Read every changed CSV ledger. Each row becomes one NPT event and one citable one-line
    document (its raw row text); a row that describes the same event as a DDR NPT entry ingested
    this run, or already on record, is stored once: the coverage table counts it.

    The "already on record" half of that check is restricted to events that actually came from a
    DDR: `store.npt(...)` alone would also return other CSV rows' events, including this same row's
    own event from a prior run, which would wrongly count a re-ingested (or merely re-ingested
    unedited) CSV row, or two independent CSV ledgers describing the same event, as a CSV/DDR
    duplicate and silently drop a row that is not a DDR duplicate at all. Dedup is defined only
    between a CSV row and a DDR entry, never between two CSV rows.
    """
    ddr_hours: dict[tuple[str, str, str], list[float]] = {}
    for e in ddr_events:
        ddr_hours.setdefault((e.well, e.date, e.code), []).append(e.hours)
    ddr_doc_ids = store.document_ids(doc_type="ddr")

    for path, rel, sha, file_key, size in changed:
        try:
            result = read_ledger(path, columns=settings.csv.columns, date_format=settings.csv.date_format)
        except (ReaderError, ValueError) as exc:
            coverage.skipped.append(SkippedFile(rel, str(exc)))
            if not dry_run:
                _replace_stale_doc_ids(store, existing_files.get(rel), set())
                store.put_files([FileRecord(
                    path=rel, sha256=sha, size=size, mtime=0.0, doc_ids=[], status="skipped",
                    reason=str(exc), field_name="", parse_config=parse_config,
                )])
            continue
        coverage.csv_row_errors += [f"{rel}: {msg}" for msg in result.errors]
        coverage.csv_row_warnings += [f"{rel}: {msg}" for msg in result.warnings]

        docs: list[Document] = []
        events: list[NptEvent] = []
        for row in result.rows:
            doc_id = f"{file_key}-r{row.line_no}"
            row_field = row.field or field or UNASSIGNED_FIELD
            code = parse.resolve_npt_code(row.code, settings.taxonomy.aliases)
            docs.append(Document(
                doc_id=doc_id, doc_type=CSV_DOC_TYPE, well=row.well, field_name=row_field,
                date=row.date, title=row.raw_line, text=row.raw_line, meta={}, source=rel,
            ))
            coverage.fields.add(row_field)
            candidates = list(ddr_hours.get((row.well, row.date, code), []))
            candidates += [
                ev.hours for ev in store.npt(well=row.well, date=row.date, code=code)
                if ev.doc_id in ddr_doc_ids
            ]
            if _hours_match(candidates, row.hours):
                coverage.duplicates_merged += 1
                continue
            events.append(NptEvent(
                doc_id=doc_id, well=row.well, field_name=row_field, date=row.date, code=code,
                hours=row.hours, hole_section=row.section or "", formation=row.formation or "",
                depth_m=row.depth_m or 0.0, mud_weight_sg=0.0, rig=row.rig or "",
                description=row.description or row.raw_line,
            ))

        if not dry_run:
            new_ids = {d.doc_id for d in docs}
            _replace_stale_doc_ids(store, existing_files.get(rel), new_ids)
            if docs:
                store.put_documents(docs)
            if events:
                store.put_npt(events)
            store.put_files([FileRecord(
                path=rel, sha256=sha, size=size, mtime=0.0,
                doc_ids=sorted(d.doc_id for d in docs), status="ingested", reason=None, field_name="",
            )])
        coverage.ingested += 1
        coverage.documents += len(docs)
        coverage.npt_events += len(events)
        coverage.chunks += sum(len(chunk_document(d.doc_id, d.text)) for d in docs)


def ingest_folder(store: Store, folder: Path | str, *, field: str | None = None,
                  config: Path | str | None = None, workspace_root: Path | str | None = None,
                  prune: bool = False, dry_run: bool = False) -> Coverage:
    """Ingest a folder of `.txt`, `.md`, `.pdf`, `.docx` and `.csv` well files.

    Unchanged files (same sha256 as the store's last-recorded `files` row) are skipped; a
    changed file replaces its document(s), chunks and events. Files whose name starts with `_`
    and the folder's own `wellbrief.toml` are never read as documents (the generator's
    ground-truth sidecar is one such file), but *are* read as configuration: the labels, section
    headings, detection headings, taxonomy aliases and CSV column mapping this run parses with
    are resolved once, via `settings.resolve_settings`, from `config` (`ingest --config`, highest
    precedence), else `folder`'s own `wellbrief.toml`, else `workspace_root`'s (the workspace's
    own, when this ingest has one), else the built-in defaults; the resolved parse configuration
    is recorded on every `FileRecord` this run writes, so a later re-parse is reproducible even if
    the workspace's own `wellbrief.toml` has since changed. `--prune` removes documents whose
    files are gone from `folder`; both are reported in the returned `Coverage`, and neither writes
    anything when `dry_run` is set.

    The incremental skip above is keyed only on a file's sha256: fixing a `wellbrief.toml` entry
    (an alias, a label, a heading) and re-ingesting does not by itself re-parse a file whose
    content did not change, since that file is still recognised as unchanged and skipped. Getting
    the new settings applied needs the file itself touched (even a no-op rewrite changes its
    hash), or a fresh store.

    `field` names the field a document with no detected field (its header, or its ledger row's
    own `field` column) is filed under; a document with neither goes to `"unassigned"`.
    """
    root = Path(folder)
    settings = resolve_settings(workspace_root=workspace_root, ingest_config_path=config,
                                ingest_folder=root)
    parse_config = settings.ingest_parse_config()
    coverage = Coverage()
    existing_files = {f.path: f for f in store.files()}

    all_files = sorted(
        p for p in root.iterdir()
        if p.is_file() and not p.name.startswith("_") and p.name != "wellbrief.toml"
    )
    coverage.files_seen = len(all_files)

    stems = [p.stem for p in all_files]
    stem_counts = {stem: stems.count(stem) for stem in set(stems)}

    seen_paths: set[str] = set()
    changed_regular: list[tuple[Path, str, str, str, int]] = []
    changed_csv: list[tuple[Path, str, str, str, int]] = []
    for p in all_files:
        rel = p.relative_to(root).as_posix()
        seen_paths.add(rel)
        suffix = p.suffix.lower()
        if suffix not in _READ_EXTENSIONS and suffix != _CSV_EXTENSION:
            coverage.skipped.append(SkippedFile(rel, "unsupported extension"))
            continue
        try:
            raw = p.read_bytes()
        except OSError as exc:
            coverage.skipped.append(SkippedFile(rel, f"cannot read file: {exc}"))
            continue
        sha = hashlib.sha256(raw).hexdigest()
        prior = existing_files.get(rel)
        if prior is not None and prior.sha256 == sha and prior.status == "ingested":
            coverage.unchanged += 1
            continue
        doc_id = _doc_id_for(rel, p.stem, collides=stem_counts[p.stem] > 1)
        entry = (p, rel, sha, doc_id, len(raw))
        (changed_csv if suffix == _CSV_EXTENSION else changed_regular).append(entry)

    ddr_events = _ingest_regular_files(store, changed_regular, field, dry_run, existing_files, coverage,
                                      settings, parse_config)
    _ingest_csv_files(store, changed_csv, field, dry_run, existing_files, ddr_events, coverage,
                      settings, parse_config)

    if prune:
        for rel in sorted(existing_files):
            if rel not in seen_paths:
                coverage.pruned += 1
                if not dry_run:
                    store.delete_file(rel)

    if not dry_run and (coverage.ingested or coverage.pruned):
        store.put_wells(_wells_from_docs(store.documents(doc_type="ddr")))

    return coverage
