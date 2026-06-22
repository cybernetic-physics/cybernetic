from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_dreamzero_sft_smoke_example_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/dreamzero_sft_smoke.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "built_datum=true" in result.stdout
    assert "remote_run=false" in result.stdout
    assert "action" in result.stdout


def test_dreamzero_rl_smoke_example_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/dreamzero_rl_smoke.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "built_rl_datum=true" in result.stdout
    assert "remote_run=false" in result.stdout
    assert "advantages" in result.stdout
    assert "rwr_weights" in result.stdout
