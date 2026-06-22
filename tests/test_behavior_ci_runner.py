"""End-to-end fixture-mode run: writes a valid behavior-ci/v1 bundle, red for
v18 and green for v19, with honest provenance and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from test_behavior_ci_evaluator import EVAL  # noqa: E402

from cybernetics.behavior_ci.artifacts import validate_bundle  # noqa: E402
from cybernetics.behavior_ci.runner import BehaviorCiRunner  # noqa: E402

CONFIG = {
    "schema_version": "cybernetic-behavior-ci-config/v1",
    "project": "unitree-g1-vla-policies",
    "robot": "Unitree G1-compatible humanoid proxy",
    "artifacts": {"out": "artifacts/behavior-ci"},
    "simulator": {
        "adapter": "fixture",
        "session": {
            "scene_env": "behavior-ci-tabletop-welding",
            "camera": "/World/Cameras/BehaviorCI_PassFailCamera",
        },
    },
    "evals": {"obstacle_shift": "evals/g1_weld_obstacle_shift.yaml"},
}


def _policy(policy_id: str, margin: float) -> dict:
    return {
        "schema_version": "behavior-ci-policy/v1",
        "policy_id": policy_id,
        "display_filename": f"{policy_id}.pt",
        "behavior": "g1_weld_approach",
        "robot": "Unitree G1-compatible humanoid proxy",
        "backend": "scripted-vla-shim",
        "controller": {"type": "scripted_trajectory", "clearance_margin_cm": margin},
    }


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "evals").mkdir()
    (tmp_path / "policies").mkdir()
    (tmp_path / "cybernetic-behavior-ci.yaml").write_text(yaml.safe_dump(CONFIG))
    (tmp_path / "evals" / "g1_weld_obstacle_shift.yaml").write_text(yaml.safe_dump(EVAL))
    (tmp_path / "policies" / "g1_weld_approach_v18.pt").write_text(
        json.dumps(_policy("g1_weld_approach_v18", 6.0))
    )
    (tmp_path / "policies" / "g1_weld_approach_v19.pt").write_text(
        json.dumps(_policy("g1_weld_approach_v19", 14.0))
    )
    return tmp_path


def test_v18_run_writes_failed_bundle(repo: Path) -> None:
    runner = BehaviorCiRunner.from_config(repo / "cybernetic-behavior-ci.yaml")
    out = repo / "art18"
    result = runner.run_policy("policies/g1_weld_approach_v18.pt", "obstacle_shift", out_dir=out)

    assert not result.passed
    assert result.summary["passed_runs"] == 5
    assert result.honesty.simulator_adapter == "fixture"
    assert result.honesty.policy_backend_real_vla is False
    assert validate_bundle(out, result) == []

    for rel in (
        "result.json",
        "metrics.json",
        "comment.md",
        "report/index.html",
        "manifest.normalized.json",
        "provenance.json",
    ):
        assert (out / rel).exists(), rel
    assert (out / "replays" / "replay-failed.mp4").exists()
    assert (out / "replays" / "replay-passed.mp4").exists()

    result_doc = json.loads((out / "result.json").read_text())
    assert result_doc["schema_version"] == "behavior-ci/v1"
    assert result_doc["status"] == "failed"
    assert "FAIL" in (out / "comment.md").read_text()
    # fixture-mode placeholder replay must be loudly disclosed in the report.
    assert "Placeholder clip" in (out / "report" / "index.html").read_text()


def test_v19_run_writes_passing_bundle(repo: Path) -> None:
    runner = BehaviorCiRunner.from_config(repo / "cybernetic-behavior-ci.yaml")
    out = repo / "art19"
    result = runner.run_policy("policies/g1_weld_approach_v19.pt", "obstacle_shift", out_dir=out)

    assert result.passed
    assert result.summary["passed_runs"] == 8
    assert validate_bundle(out, result) == []
    assert not (out / "replays" / "replay-failed.mp4").exists()
    assert (out / "replays" / "replay-passed.mp4").exists()
    assert "PASS" in (out / "comment.md").read_text()
