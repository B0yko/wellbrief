"""Write rendered documents to disk, one writer per file format.

A writer receives the rendered lines and a target path and produces one
file. `WRITERS` maps a writer name to its file suffix and function, and
`FORMATS` maps a corpus format (the `--formats` option) to the writer used
for each document type. Plain text is the only writer in this version; a PDF
or DOCX writer is added with `register_writer()` and a `FORMATS` entry.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .fields import FIELDS
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


WRITERS: dict[str, Writer] = {"txt": Writer(".txt", write_txt)}

FORMATS: dict[str, dict[str, str]] = {
    "txt": {"ddr": "txt", "eowr": "txt", "incident": "txt"},
}

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


def write_corpus(corpus: Corpus, out_dir: Path | str, formats: str = "txt") -> list[Path]:
    """Write every document and the ground-truth sidecar. Returns the written paths."""
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
    "FORMATS", "WRITERS", "OutputDirError", "Writer", "manifest_hash",
    "register_writer", "write_corpus", "write_txt",
]
