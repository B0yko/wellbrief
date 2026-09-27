"""Retrieval ablation: `wellbrief eval --ablation` (search.ABLATION_MODES over the extended
suite's retrieval-precision cases), on the default corpus seed."""

from __future__ import annotations

from pathlib import Path

from wellbrief.corpus import SEED
from wellbrief.evals import ablation
from wellbrief.search import ABLATION_MODES, MODE_LABELS

# The same case the shipped `extended.toml` runs (its `relevant_sql` is known non-empty on
# the default seed's corpus: `eval --suite extended` passes it every run).
_STUCK_PIPE_CASE = (
    'suite = "extended"\n'
    '[[case]]\n'
    'id = "rp-orrindale-stuck"\n'
    'category = "retrieval-precision"\n'
    'question = "What caused stuck pipe on Orrindale wells in the 17 1/2\\" section?"\n'
    'field = "Orrindale"\n'
    'relevant_sql = """\n'
    "SELECT doc_id FROM truth_events WHERE field = 'Orrindale' AND code = 'STUCK_PIPE' "
    "AND section = '17 1/2\"'\n"
    "UNION SELECT doc_id FROM truth_docs WHERE type = 'incident' AND field = 'Orrindale' "
    "AND code = 'STUCK_PIPE'\n"
    "UNION SELECT doc_id FROM truth_sentences WHERE field = 'Orrindale' AND code = 'STUCK_PIPE' "
    "AND label = 'failure'\n"
    '"""\n'
)


def _cases_dir(tmp_path: Path, text: str) -> Path:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "extended.toml").write_text(text, encoding="utf-8")
    return cases_dir


def test_run_scores_every_mode_over_the_one_case(tmp_path: Path) -> None:
    payload = ablation.run([SEED], ["eval", "--ablation"], cases_dir=_cases_dir(tmp_path, _STUCK_PIPE_CASE))
    modes = {m["mode"]: m for m in payload["modes"]}
    assert set(modes) == set(ABLATION_MODES)
    for mode in ABLATION_MODES:
        assert modes[mode]["label"] == MODE_LABELS[mode]
        assert modes[mode]["cases"] == 1
        assert 0.0 <= modes[mode]["mean_p_at_8"] <= 1.0
        assert 0.0 <= modes[mode]["mean_mrr_at_8"] <= 1.0
    # The product's own retrieval (planner filters on) is at least as precise on this case
    # as the same fused ranking searched without them.
    assert modes["hybrid_filtered"]["mean_p_at_8"] >= modes["hybrid_unfiltered"]["mean_p_at_8"]
    assert payload["metadata"]["suites"] == ["extended"]
    assert payload["metadata"]["seeds"] == [SEED]


def test_table_lists_every_mode_by_label(tmp_path: Path) -> None:
    payload = ablation.run([SEED], ["eval", "--ablation"], cases_dir=_cases_dir(tmp_path, _STUCK_PIPE_CASE))
    text = ablation.table(payload)
    for mode in ABLATION_MODES:
        assert MODE_LABELS[mode] in text


def test_a_case_whose_relevant_set_is_empty_on_this_corpus_is_skipped(tmp_path: Path) -> None:
    empty_case = (
        'suite = "extended"\n'
        '[[case]]\n'
        'id = "rp-nothing"\n'
        'category = "retrieval-precision"\n'
        'question = "Show wellbore instability in Orrindale"\n'
        'field = "Orrindale"\n'
        "relevant_sql = \"SELECT doc_id FROM truth_docs WHERE field = 'No Such Field'\"\n"
    )
    payload = ablation.run([SEED], ["eval", "--ablation"], cases_dir=_cases_dir(tmp_path, empty_case))
    assert all(m["cases"] == 0 for m in payload["modes"])
    assert all(m["mean_p_at_8"] == 0.0 and m["mean_mrr_at_8"] == 0.0 for m in payload["modes"])
