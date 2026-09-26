"""The ground-truth sidecar stays out of the product's code path.

Gold answers for the evaluation suites come from `_truth.json`, so the
product must never see it: ingestion skips it, the corpus package only
writes it, and no module outside `wellbrief.evals` reads it.

The static check looks for anything in a module's code that could name the
file: a string containing "truth.json" (also as a piece of a concatenated
name), a glob pattern that would match it, or an import or attribute of the
sidecar module. The runtime check makes the file unreadable and ingests the
folder anyway.
"""

from __future__ import annotations

import ast
import os
import stat
from fnmatch import fnmatch
from pathlib import Path

import pytest

from wellbrief import ingest
from wellbrief.corpus import build_corpus, write_corpus

SRC = Path(__file__).resolve().parents[1] / "src" / "wellbrief"
CORPUS_PACKAGE = SRC / "corpus"
EVALS_PACKAGE = SRC / "evals"
SIDECAR = "_truth.json"


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if EVALS_PACKAGE not in p.parents)


def _docstrings(tree: ast.AST) -> set[int]:
    """ids of the string constants that are docstrings."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


def _mentions_truth(tree: ast.AST) -> list[str]:
    """Strings, glob patterns, imported names and attributes that could refer to the sidecar."""
    hits = []
    docstrings = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            value = node.value
            if "truth.json" in value or value == "_truth":
                hits.append(f"string {value!r}")
            elif any(c in value for c in "*?[") and fnmatch(SIDECAR, value):
                hits.append(f"glob {value!r}")
        elif isinstance(node, ast.ImportFrom):
            hits += [f"import {a.name}" for a in node.names if "truth" in a.name.lower()]
            if node.module and "truth" in node.module.lower():
                hits.append(f"from {node.module}")
        elif isinstance(node, ast.Attribute) and "truth" in node.attr.lower():
            hits.append(f"attribute {node.attr}")
    return hits


def test_no_module_outside_corpus_and_evals_refers_to_the_sidecar() -> None:
    """The corpus package writes the sidecar and the evaluation package reads it; nothing else names it."""
    offenders = {}
    for path in _modules():
        if CORPUS_PACKAGE in path.parents:
            continue
        hits = _mentions_truth(ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            offenders[str(path.relative_to(SRC))] = hits
    assert offenders == {}


def _names_the_file(tree: ast.AST) -> bool:
    return any("truth.json" in h or h.startswith("glob") for h in _mentions_truth(tree))


def test_inside_the_evaluation_package_only_the_truth_module_names_the_sidecar() -> None:
    naming = [p.name for p in sorted(EVALS_PACKAGE.rglob("*.py"))
              if _names_the_file(ast.parse(p.read_text(encoding="utf-8")))]
    assert naming == ["truth.py"]


def _product_imports(tree: ast.AST) -> list[str]:
    """Imports of wellbrief modules outside the evaluation package."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 2 or (node.level == 0 and (node.module or "").startswith("wellbrief")
                                   and not (node.module or "").startswith("wellbrief.evals")):
                found.append(f"{'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names
                      if a.name.startswith("wellbrief") and not a.name.startswith("wellbrief.evals")]
    return found


def test_only_the_adapter_calls_the_product() -> None:
    """Gold answers never go through product code: every other evaluation module stays inside the package."""
    offenders = {p.name: hits for p in sorted(EVALS_PACKAGE.rglob("*.py"))
                 if p.name != "adapter.py" and (hits := _product_imports(ast.parse(p.read_text(encoding="utf-8"))))}
    assert offenders == {}
    assert _product_imports(ast.parse((EVALS_PACKAGE / "adapter.py").read_text(encoding="utf-8")))


@pytest.mark.parametrize("snippet", [
    "NAME = '_truth.json'",
    "NAME = '_' + 'truth.json'",
    "from pathlib import Path\nFILES = Path('.').glob('_*.json')",
    "from .corpus.truth import TRUTH_FILENAME",
    "from .corpus import truth\nX = truth.TRUTH_FILENAME",
])
def test_the_static_check_catches_ways_of_naming_the_sidecar(snippet: str) -> None:
    assert _mentions_truth(ast.parse(snippet))


def test_the_corpus_package_only_writes_the_sidecar() -> None:
    """No module of the generator parses JSON or reads a file back as text (the manifest only hashes bytes)."""
    for path in sorted(CORPUS_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in {"load", "loads", "read_text", "open"}:
                    raise AssertionError(f"{path.name}:{node.lineno} calls {name}()")
    # Only the truth module names the file; the others import its constant.
    naming = [p.name for p in sorted(CORPUS_PACKAGE.rglob("*.py"))
              if any("truth.json" in h for h in _mentions_truth(ast.parse(p.read_text(encoding="utf-8"))))]
    assert naming == ["truth.py"]


def test_ingestion_never_reads_underscore_files(tmp_path: Path) -> None:
    written = write_corpus(build_corpus(), tmp_path)
    (tmp_path / "_notes.txt").write_text("DAILY DRILLING REPORT\nnot a document\n", encoding="utf-8")
    docs = ingest.load_corpus_dir(tmp_path)
    assert len(docs) == len(written) - 1
    assert not any(d.doc_id.startswith("_") for d in docs)
    assert (tmp_path / SIDECAR).exists()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs file permissions that bind the user")
def test_ingestion_works_when_the_sidecar_cannot_be_read(tmp_path: Path) -> None:
    written = write_corpus(build_corpus(), tmp_path)
    sidecar = tmp_path / SIDECAR
    sidecar.chmod(0)
    try:
        docs = ingest.load_corpus_dir(tmp_path)
    finally:
        sidecar.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert len(docs) == len(written) - 1


def test_ingestion_works_when_the_sidecar_is_a_directory(tmp_path: Path) -> None:
    written = write_corpus(build_corpus(), tmp_path)
    (tmp_path / SIDECAR).unlink()
    (tmp_path / SIDECAR).mkdir()
    assert len(ingest.load_corpus_dir(tmp_path)) == len(written) - 1
