from pathlib import Path

from locium.config import (
    COLLECTION_NAME,
    DEFAULT_INDEX,
    DEFAULT_PALACE,
    TUNING,
)


def test_tuning_matches_spec_defaults():
    assert TUNING.arc_max_distance == 0.45
    assert TUNING.arcs_per_drawer == 3
    assert TUNING.arc_global_cap == 20000
    assert TUNING.preview_chars == 200
    assert TUNING.seed == 42
    assert TUNING.dot_cap == 0
    assert TUNING.pad_hall == 8.5
    assert TUNING.pad_chamber == 5.0


def test_tuning_is_immutable():
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        TUNING.seed = 7


def test_default_paths():
    assert DEFAULT_PALACE == Path.home() / ".mempalace" / "palace"
    assert DEFAULT_INDEX == Path.home() / ".locium" / "index"
    assert COLLECTION_NAME == "mempalace_drawers"
