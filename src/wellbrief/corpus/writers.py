"""Write rendered documents to disk, one writer per file format.

A writer receives the rendered lines and a target path and produces one
file. `WRITERS` maps a writer name to its file suffix and function, and
`FORMATS` maps a corpus format (the `--formats` option) to the writer used
for each document type: `txt` (the default), `pdf` and `docx` write every
document type in that one format, and `mixed` writes daily reports as
`.txt`, end-of-well reports as `.pdf` and incident reports as `.docx`, so
that a `demo` or a `corpus generate --formats mixed` corpus exercises every
reader ingestion has. A writer for another format is added with
`register_writer()` and a `FORMATS` entry.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .docxwriter import write_docx
from .fields import FIELDS
from .pdfwriter import write_pdf
from .records import Corpus
from .truth import TRUTH_FILENAME, build_truth, write_truth

WriteFn = Callable[[Sequence[str], Path], None]


@dataclass(frozen=True)
class Writer:
    suffix: str
    write: WriteFn


def write_txt(lines: Sequence[str], path: Path) -> None:
    """UTF-8, LF line endings, a final newline: byte-identical on every platform."""
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


WRITERS: dict[str, Writer] = {
    "txt": Writer(".txt", write_txt),
    "pdf": Writer(".pdf", write_pdf),
    "docx": Writer(".docx", write_docx),
}

FORMATS: dict[str, dict[str, str]] = {
    "txt": {"ddr": "txt", "eowr": "txt", "incident": "txt"},
    "pdf": {"ddr": "pdf", "eowr": "pdf", "incident": "pdf"},
    "docx": {"ddr": "docx", "eowr": "docx", "incident": "docx"},
    "mixed": {"ddr": "txt", "eowr": "pdf", "incident": "docx"},
}

LEDGER_FILENAME = "npt-ledger.csv"

# The default column layout `readers.csvledger.read_ledger` expects with no `[csv.columns]`
# mapping, so the generated ledger can be read back (and ingested) with no configuration.
LEDGER_COLUMNS = ("well", "date", "code", "hours", "field", "depth", "section", "formation", "rig",
                 "description")

# Names the generator writes: documents of its own fields, the sidecar and the
# NPT ledger. A directory is treated as an earlier corpus only when it holds
# the sidecar and nothing but these names; anything else belongs to someone
# else and is never touched.
_PREFIXES = "|".join(sorted(f.prefix for f in FIELDS))
_GENERATED_NAME = re.compile(
    rf"^(?:(?:DDR|EOWR|INC)-(?:{_PREFIXES})-\d{{2,4}}(?:-\d{{2,3}})?\.(?:txt|pdf|docx)"
    rf"|{re.escape(TRUTH_FILENAME)}|npt-ledger\.csv)$"
)


def register_writer(name: str, suffix: str, write: WriteFn) -> None:
    WRITERS[name] = Writer(suffix, write)


class OutputDirError(Exception):
    """The output directory is neither empty nor an earlier generated corpus."""


def _prepare(out_dir: Path) -> None:
    """Make sure `out_dir` is empty, clearing it only if it holds an earlier corpus.

    Hidden files (a desktop's folder metadata) are ignored and kept.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted((p for p in out_dir.iterdir() if not p.name.startswith(".")), key=lambda p: p.name)
    if not entries:
        return
    if not (out_dir / TRUTH_FILENAME).is_file():
        raise OutputDirError(
            f"{out_dir} is not empty and holds no earlier corpus (for example {entries[0].name}); "
            f"use an empty directory"
        )
    foreign = [p.name for p in entries if not (p.is_file() and _GENERATED_NAME.match(p.name))]
    if foreign:
        raise OutputDirError(
            f"{out_dir} contains files the corpus generator did not write (for example {foreign[0]}); "
            f"use an empty directory"
        )
    for p in entries:
        p.unlink()


def _ledger_rows(corpus: Corpus) -> list[list[str]]:
    """Every DDR NPT row of the corpus, well by well and day by day (deterministic:
    both orders come straight from the generator's own, seed-deterministic output)."""
    return [
        [w.name, d.day.isoformat(), e.code, f"{e.hours:.1f}", w.spec.name, str(e.depth_m),
         d.section.size, e.formation, w.rig, e.description]
        for w in corpus.wells for d in w.days for e in d.entries
    ]


def write_ledger_csv(corpus: Corpus, path: Path) -> None:
    """Write every DDR NPT row of `corpus` as an NPT ledger CSV (`corpus generate --ledger-csv`),
    in `LEDGER_COLUMNS` order so it reads back with `readers.csvledger.read_ledger`'s defaults."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LEDGER_COLUMNS)
    writer.writerows(_ledger_rows(corpus))
    path.write_bytes(buffer.getvalue().encode("utf-8"))


def write_corpus(corpus: Corpus, out_dir: Path | str, formats: str = "txt",
                 ledger_csv: bool = False) -> list[Path]:
    """Write every document and the ground-truth sidecar. Returns the written paths.

    `ledger_csv` also writes `LEDGER_FILENAME` with every DDR NPT row:
    the format-parity evaluation cases and the fresh-clone CSV check ingest it on its own.
    """
    if formats not in FORMATS:
        raise ValueError(f"unknown corpus format {formats!r}; choose from {', '.join(sorted(FORMATS))}")
    plan = FORMATS[formats]
    out = Path(out_dir)
    _prepare(out)
    written: list[Path] = []
    files: dict[str, str] = {}
    for doc in corpus.documents:
        writer = WRITERS[plan[doc.doc_type]]
        path = out / f"{doc.doc_id}{writer.suffix}"
        writer.write(doc.lines, path)
        files[doc.doc_id] = path.name
        written.append(path)
    if ledger_csv:
        ledger_path = out / LEDGER_FILENAME
        write_ledger_csv(corpus, ledger_path)
        written.append(ledger_path)
    written.append(write_truth(out, build_truth(corpus, files, formats)))
    return written


def manifest_hash(out_dir: Path | str, files: Sequence[Path] | None = None) -> str:
    """sha256 over the sorted (relative path, sha256 of the bytes) of the corpus files.

    Without `files`, every file under `out_dir` is included (the sidecar too),
    except hidden files such as a desktop's folder metadata.
    """
    root = Path(out_dir)
    if files is None:
        paths = [p for p in root.rglob("*")
                 if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    else:
        paths = list(files)
    entries = sorted(
        (p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths
    )
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(f"{rel}\t{digest}\n".encode())
    return h.hexdigest()


__all__ = [
    "FORMATS", "LEDGER_COLUMNS", "LEDGER_FILENAME", "WRITERS", "OutputDirError", "Writer",
    "manifest_hash", "register_writer", "write_corpus", "write_ledger_csv", "write_txt",
]
