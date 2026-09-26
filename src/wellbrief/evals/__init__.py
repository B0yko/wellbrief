"""Evaluation harness: case files, ground truth, checks and result files.

`truth` is the only reader of a generated corpus's `_truth.json`, `adapter`
the only caller of the product, `cases` reads `evals/cases/*.toml`, `suites`
holds one check per case category, `runner` runs them per seed, `metrics`
scores and `results` writes the result files. `wellbrief eval` is the command
line entry point.
"""
