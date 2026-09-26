from __future__ import annotations

from pathlib import Path

import pytest
from corpus_files import ParsedCorpus, generated

from wellbrief.corpus import SEED

SEEDS = (SEED, 7, 42)


@pytest.fixture(scope="session")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("corpora")


@pytest.fixture(scope="session", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def corpus(request: pytest.FixtureRequest, corpus_root: Path) -> ParsedCorpus:
    """A corpus written to disk and read back from its files, for each seed."""
    seed = request.param
    return generated(str(corpus_root / f"seed-{seed}"), seed)
