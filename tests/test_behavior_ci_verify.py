"""verify-task integrity gate + candidate-copy pin enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cybernetics.behavior_ci.schemas import ContractError
from cybernetics.behavior_ci.tasks import TaskLock, sha256_bytes, verify_candidate_copies
from cybernetics.cli.commands.behavior_ci import cli

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


def _v2(extra=None):
    d = {
        "schema_version": "behavior-ci-policy/v2",
        "policy_id": "g1_weld_approach_v21",
        "display_filename": "g1_weld_approach_v21.pt",
        "behavior": "g1_weld_approach",
        "robot": "Unitree G1-compatible humanoid proxy",
        "backend": "scripted-vla-shim",
        "task": "g1_weld_approach",
        "checkpoint": {"detour_mode": "relative", "detour_gain": 1.0},
    }
    if extra:
        d.update(extra)
    return d


def _setup(tmp_path: Path, policy: dict) -> str:
    (tmp_path / "cybernetic-behavior-ci.yaml").write_text(CONFIG)
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "p.pt").write_text(json.dumps(policy))
    return str(tmp_path / "cybernetic-behavior-ci.yaml")


def test_verify_task_ok(tmp_path: Path):
    cfg = _setup(tmp_path, _v2())
    r = CliRunner().invoke(cli, ["verify-task", "--config", cfg, "--policy-ref", "policies/p.pt"])
    assert r.exit_code == 0, r.output
    assert "task=g1_weld_approach" in r.output and "pins_verified=true" in r.output


def test_verify_task_rejects_session_entrypoint(tmp_path: Path):
    cfg = _setup(tmp_path, _v2({"session_entrypoint": "evil"}))
    r = CliRunner().invoke(cli, ["verify-task", "--config", cfg, "--policy-ref", "policies/p.pt"])
    assert r.exit_code == 2  # closed schema -> input error


def test_verify_task_rejects_unknown_task(tmp_path: Path):
    cfg = _setup(tmp_path, _v2({"task": "no_such_task"}))
    r = CliRunner().invoke(cli, ["verify-task", "--config", cfg, "--policy-ref", "policies/p.pt"])
    assert r.exit_code == 2


def test_candidate_copy_pin_enforcement(tmp_path: Path):
    (tmp_path / "evals").mkdir()
    f = tmp_path / "evals" / "copy.yaml"
    f.write_text("authoritative bytes")
    good = sha256_bytes(b"authoritative bytes")

    # identical copy -> OK
    verify_candidate_copies(
        tmp_path, TaskLock.from_dict({"candidate_copies": {"evals/copy.yaml": good}})
    )

    # tampered copy -> ContractError (exit 4 at the CLI)
    with pytest.raises(ContractError):
        verify_candidate_copies(
            tmp_path, TaskLock.from_dict({"candidate_copies": {"evals/copy.yaml": "0" * 64}})
        )

    # absent copy -> OK (the candidate may delete the readability copy entirely)
    verify_candidate_copies(
        tmp_path, TaskLock.from_dict({"candidate_copies": {"evals/missing.yaml": good}})
    )
