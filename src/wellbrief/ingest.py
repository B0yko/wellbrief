"""Loading documents into the store.

Two entry points share one parser: the generated synthetic corpus, and a
folder of plain-text well files in the same template. Both yield the same
NPT ledger. The folder path takes each title from the first line of the file
and rebuilds the well register from the documents, so titles and well
metadata can differ from those of the generated corpus.
"""

from __future__ import annotations

from pathlib import Path

from . import corpus, parse
from .models import Document, Well
from .store import Store, build_indexes

DOC_TYPE_BY_PREFIX = {"DDR": "ddr", "EOWR": "eowr", "INC": "incident"}


def infer_doc_type(name: str) -> str:
    prefix = name.split("-", 1)[0].upper()
    return DOC_TYPE_BY_PREFIX.get(prefix, "ddr")


def load_corpus_dir(path: Path | str) -> list[Document]:
    """Read a folder of plain-text well files."""
    path = Path(path)
    docs: list[Document] = []
    for file in sorted(path.glob("*.txt")):
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
        ))
    return docs


def ingest(store: Store, docs: list[Document], wells: list[Well] | None = None) -> dict[str, int]:
    """Parse, store and index. Metadata is always recomputed from the text.

    Metadata that arrives with a document is not trusted: it is recomputed
    from the text, because metadata exported from another system can be
    missing or disagree with the report it describes.
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
    build_indexes(store)
    return {"documents": len(parsed_docs), "npt_events": len(all_events)}


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


def bootstrap(store: Store, regenerate: bool = True) -> dict[str, int]:
    """Generate the demo field history and load it end to end."""
    store.reset()
    wells, docs = corpus.generate(write_files=regenerate)
    # Drop the generator's own metadata so ingest has to earn it from the text.
    for d in docs:
        d.meta = {}
    return ingest(store, docs, wells)
