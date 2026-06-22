"""Contract: the q99 relative-stats loader has a bundled-path seam and a
diagnosable error when no data is available (no fabricated statistics).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cybernetics.lib.dreamzero import relative_actions
from cybernetics.lib.dreamzero.relative_actions import (
    RelativeStatsUnavailable,
    default_relative_stats_path,
    load_relative_stats,
)

_STATS = {
    "joint_position": {
        "q01": [-1.0, -1.0],
        "q99": [1.0, 1.0],
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
        "min": [-2.0, -2.0],
        "max": [2.0, 2.0],
    }
}


def test_load_from_explicit_path(tmp_path: Path) -> None:
    p = tmp_path / "relative_stats_dreamzero.json"
    p.write_text(json.dumps(_STATS))
    loaded = load_relative_stats(str(p))
    assert set(loaded) == {"joint_position"}
    assert loaded["joint_position"].q99.tolist() == [1.0, 1.0]


def test_missing_data_raises_diagnosable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force "not bundled" regardless of the actual install.
    monkeypatch.setattr(relative_actions, "default_relative_stats_path", lambda: None)
    with pytest.raises(RelativeStatsUnavailable, match="relative_stats_dreamzero.json"):
        load_relative_stats()


def test_default_path_returns_none_or_existing_file() -> None:
    # The data file is not committed; the seam must return None (not crash) when absent.
    result = default_relative_stats_path()
    assert result is None or result.is_file()
