"""End-to-end runner test for the pinned v2 Task Pack path (fixture adapter, no network).

Proves the whole anti-gaming pipeline offline: a v2 policy emits a trajectory, the runner
grades it against the pinned task's VISIBLE + HELD-OUT scenarios with independent geometric
measurement, and an under-detour regresses while a tuned policy passes 16/16.
"""

from __future__ import annotations

import json
from pathlib import Path

from cybernetics.behavior_ci.runner import BehaviorCiRunner

CONFIG = """
schema_version: cybernetic-behavior-ci-config/v1
project: unitree-g1-vla-policies
robot: Unitree G1-compatible humanoid proxy
artifacts:
  out: artifacts/behavior-ci
simulator:
  adapter: fixture
  session:
    scene_env: behavior-ci-tabletop-welding
    camera: /World/Cameras/BehaviorCI_PassFailCamera
policy_backends:
  scripted-vla-shim:
    type: scripted
evals:
  obstacle_shift:
    path: evals/unused.yaml
"""


def _policy(checkpoint, pid="g1_weld_approach_vX"):
    return {
        "schema_version": "behavior-ci-policy/v2",
        "policy_id": pid,
        "display_filename": f"{pid}.pt",
        "behavior": "g1_weld_approach",
        "robot": "Unitree G1-compatible humanoid proxy",
        "backend": "scripted-vla-shim",
        "task": "g1_weld_approach",
        "checkpoint": checkpoint,
    }


def _setup(tmp_path: Path):
    (tmp_path / "cybernetic-behavior-ci.yaml").write_text(CONFIG)
    (tmp_path / "policies").mkdir()
    return BehaviorCiRunner.from_config(tmp_path / "cybernetic-behavior-ci.yaml")


def _run(runner, tmp_path, name, checkpoint):
    p = tmp_path / "policies" / f"{name}.pt"
    p.write_text(json.dumps(_policy(checkpoint, name)))
    return runner.run_policy(
        policy_ref=f"policies/{name}.pt",
        eval_ref="obstacle_shift",
        out_dir=tmp_path / "out" / name,
    )


def test_under_detour_regresses_good_passes(tmp_path: Path):
    runner = _setup(tmp_path)

    under = _run(
        runner,
        tmp_path,
        "v18",
        {
            "detour_mode": "relative",
            "detour_gain": 0.85,
            "clearance_margin_cm": 6.5,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": 0.075,
        },
    )
    assert under.status == "failed"
    # 16 scenarios graded (8 visible + 8 held-out), not just the published 8.
    assert under.summary["total_runs"] == 16
    assert under.honesty.pins_verified is True
    assert under.honesty.task_id == "g1_weld_approach"
    assert under.honesty.production_eval_path_used is True
    codes = {f["code"] for f in under.failures}
    assert {"SAFETY_ZONE_INTRUSION", "OBSTACLE_COLLISION", "TARGET_TIMEOUT"} & codes

    good = _run(
        runner,
        tmp_path,
        "v19",
        {
            "detour_mode": "relative",
            "detour_gain": 1.0,
            "clearance_margin_cm": 12.0,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": 0.12,
        },
    )
    assert good.status == "passed"
    assert good.summary["passed_runs"] == 16


def test_geometry_blind_constant_apex_is_rejected(tmp_path: Path):
    """A fixed (geometry-blind) apex that clears the visible set still fails on the held-out
    bank -> the runner returns failed. Overfitting to the published scenarios cannot win."""
    runner = _setup(tmp_path)
    res = _run(
        runner,
        tmp_path,
        "overfit",
        {
            "detour_mode": "absolute",
            "absolute_apex_cm": 55.0,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": 0.12,
        },
    )
    assert res.status == "failed"
