"""Loading documents into the store.

One entry point reads a folder of plain-text well files in the template
`parse.py` understands (the generator writes the same template); another
parses, chunks, files and indexes-worth of NPT it into the store. Readers for
`.pdf`, `.docx` and `.csv` are a later phase; this one is
deliberately `.txt`/`.md` only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import parse
from .models import Document, FileRecord, Well
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
    text this phase's reader produces; a byte-accurate reader (a later
    phase) will recompute both from the raw file bytes instead, which
    is also the basis for the incremental, sha256-skip ingest that comes with
    it. `mtime` is left at 0.0 here for the same reason: it is not used until
    that incremental check exists.
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
    by_well: dict[str, dict] = {}
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
