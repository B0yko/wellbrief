"""`narrate.get_narrator`: the only narrator it can build without further configuration."""

from __future__ import annotations

import pytest

from wellbrief import narrate


def test_offline_resolves_to_the_offline_narrator() -> None:
    narrator = narrate.get_narrator("offline")
    assert isinstance(narrator, narrate.OfflineNarrator)
    assert narrate.get_narrator() is not narrator  # a fresh instance each call


def test_llm_names_narrator_from_env_instead() -> None:
    with pytest.raises(ValueError, match="narrator_from_env"):
        narrate.get_narrator("llm")


def test_an_unknown_name_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown narrator 'remote'"):
        narrate.get_narrator("remote")
