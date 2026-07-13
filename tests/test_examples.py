from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


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


def test_pi0_droid_sampling_example_validates_observation_without_network(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    observation = tmp_path / "observation.npz"
    np.savez(
        observation,
        exterior_image_0_left=np.zeros((2, 3, 3), dtype=np.uint8),
        exterior_image_1_left=np.ones((2, 3, 3), dtype=np.uint8),
        wrist_image_left=np.full((2, 3, 3), 2, dtype=np.uint8),
        joint_position=np.arange(7, dtype=np.float32),
        gripper_position=np.array([0.25], dtype=np.float32),
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/pi0_droid_sampling.py",
            str(observation),
            "--instruction",
            "pick up the cube",
            "--validate-only",
        ],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "observation_valid=true" in result.stdout
    assert "remote_run=false" in result.stdout


def test_pi0_droid_sampling_example_does_not_offer_unsupported_seed() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/pi0_droid_sampling.py", "--help"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--seed" not in result.stdout
