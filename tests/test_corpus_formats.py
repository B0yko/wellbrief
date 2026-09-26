"""The corpus generator's non-txt renderings: `pdf`, `docx`, `mixed`, and `--ledger-csv`
(`corpus generate --formats txt|mixed|pdf|docx [--ledger-csv]`).

`test_corpus_invariants.py` covers the `txt` rendering and the truth sidecar; this file covers
the writers this phase adds on top of it: manifest-hash determinism for every format, that the
real generated documents (not just the writers' own hand-picked samples in `test_pdfwriter.py`)
round-trip through `pypdf` with their spaces intact, and the ledger CSV against the truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wellbrief.corpus import SEED, build_corpus, manifest_hash, write_corpus
from wellbrief.readers.csvledger import read_ledger
from wellbrief.readers.docx import read_docx
from wellbrief.readers.pdf import read_pdf


@pytest.mark.parametrize("formats", ["txt", "mixed", "pdf", "docx"])
def test_manifest_hash_is_deterministic_for_every_format(tmp_path: Path, formats: str) -> None:
    hashes = []
    for name in ("a", "b"):
        written = write_corpus(build_corpus(seed=SEED), tmp_path / f"{formats}-{name}", formats,
                               ledger_csv=True)
        hashes.append(manifest_hash(tmp_path / f"{formats}-{name}", written))
    assert hashes[0] == hashes[1]


def test_mixed_writes_ddr_txt_eowr_pdf_incident_docx(tmp_path: Path) -> None:
    corpus = build_corpus(seed=SEED)
    write_corpus(corpus, tmp_path, "mixed")
    by_type = {d.doc_type: d.doc_id for d in corpus.documents}
    assert (tmp_path / f"{by_type['ddr']}.txt").exists()
    assert (tmp_path / f"{by_type['eowr']}.pdf").exists()
    assert (tmp_path / f"{by_type['incident']}.docx").exists()


@pytest.mark.parametrize("doc_type", ["ddr", "eowr", "incident"])
def test_pdf_rendering_of_the_real_corpus_keeps_every_line_and_its_spaces_intact(
        tmp_path: Path, doc_type: str) -> None:
    corpus = build_corpus(seed=SEED)
    write_corpus(corpus, tmp_path, "pdf")
    doc = next(d for d in corpus.documents if d.doc_type == doc_type)
    pages = read_pdf(tmp_path / f"{doc.doc_id}.pdf")
    assert "\n".join(page.text for page in pages) == "\n".join(doc.lines)


@pytest.mark.parametrize("doc_type", ["ddr", "eowr", "incident"])
def test_docx_rendering_of_the_real_corpus_round_trips_exactly(tmp_path: Path, doc_type: str) -> None:
    corpus = build_corpus(seed=SEED)
    write_corpus(corpus, tmp_path, "docx")
    doc = next(d for d in corpus.documents if d.doc_type == doc_type)
    text = read_docx(tmp_path / f"{doc.doc_id}.docx")
    assert text == "\n".join(doc.lines)


def test_ledger_csv_matches_the_truth_events(tmp_path: Path) -> None:
    corpus = build_corpus(seed=SEED)
    write_corpus(corpus, tmp_path, "txt", ledger_csv=True)
    truth = json.loads((tmp_path / "_truth.json").read_text(encoding="utf-8"))

    result = read_ledger(tmp_path / "npt-ledger.csv")
    assert not result.errors
    assert not result.warnings
    assert len(result.rows) == len(truth["npt_events"])

    by_key = {(r.well, r.date, r.code, round(r.hours, 1)): r for r in result.rows}
    for e in truth["npt_events"]:
        row = by_key[(e["well"], e["date"], e["code"], round(e["hours"], 1))]
        assert row.field == e["field"]
        assert row.depth_m == pytest.approx(e["depth_m"])
        assert row.section == e["section"]
        assert row.formation == e["formation"]
        assert row.rig == e["rig"]


def test_ledger_csv_is_only_written_when_asked_for(tmp_path: Path) -> None:
    write_corpus(build_corpus(seed=SEED), tmp_path)
    assert not (tmp_path / "npt-ledger.csv").exists()
