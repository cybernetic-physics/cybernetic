from __future__ import annotations

import pytest

from cybernetics.robotics import (
    ROBOT_POLICY_SCHEMA_VERSION,
    ROBOT_TASK_SCHEMA_VERSION,
    PolicyArtifact,
    RobotContractError,
    RobotTaskSpec,
)


def task_dict() -> dict:
    return {
        "schema_version": ROBOT_TASK_SCHEMA_VERSION,
        "task_id": "fixture_walk",
        "robot_id": "fixture_bot",
        "simulator_backend": "fixture",
        "backend_config": {"image": "fixture"},
        "asset_refs": [{"kind": "fixture", "uri": "fixture://bot"}],
        "joint_map": {"left_hip": "L_HIP", "right_hip": "R_HIP"},
        "actuator_model": {"kind": "position"},
        "observation_space": {"position": {"dtype": "float32", "shape": []}},
        "action_space": {"delta": {"dtype": "float32", "shape": []}},
        "sim_dt": 0.01,
        "control_dt": 0.02,
        "reset_spec": {"position": 0.0},
        "reward_spec": {"kind": "fixture_position"},
        "success_metric": {"metric": "position", "operator": ">=", "value": 3.0},
        "randomization": {},
        "termination": {"max_steps": 8},
        "eval_protocol": {"episodes": 1, "max_steps": 8},
    }


def policy_dict(task: RobotTaskSpec, *, checkpoint_uri: str | None = None) -> dict:
    return {
        "schema_version": ROBOT_POLICY_SCHEMA_VERSION,
        "artifact_id": "pol_fixture",
        "task_spec_uri": f"task://{task.task_id}",
        "task_spec_hash": task.task_hash(),
        "checkpoint_uri": checkpoint_uri,
        "policy_format": "worldlines" if checkpoint_uri else "custom",
        "observation_schema": task.observation_space,
        "action_schema": task.action_space,
        "robot_id": task.robot_id,
        "simulator_backend": task.simulator_backend,
        "backend_version": "fixture",
        "eval_metrics": {"success_rate": 1.0},
        "rollout_artifacts": ["rollout.json"],
        "created_by_run_id": "rrun_fixture",
    }


def test_robot_task_spec_round_trips_and_hashes_joint_map() -> None:
    spec = RobotTaskSpec.from_dict(task_dict())

    assert spec.to_dict() == task_dict()
    assert spec.joint_map == {"left_hip": "L_HIP", "right_hip": "R_HIP"}
    assert len(spec.task_hash()) == 64
    assert RobotTaskSpec.from_dict(spec.to_dict()).task_hash() == spec.task_hash()


def test_robot_task_spec_requires_schema_version() -> None:
    data = task_dict()
    data.pop("schema_version")

    with pytest.raises(RobotContractError, match="schema_version"):
        RobotTaskSpec.from_dict(data)


def test_robot_task_spec_rejects_invalid_backend() -> None:
    data = task_dict()
    data["simulator_backend"] = "rlmesh"

    with pytest.raises(RobotContractError, match="simulator_backend"):
        RobotTaskSpec.from_dict(data)


def test_robot_task_spec_requires_control_dt_multiple_of_sim_dt() -> None:
    data = task_dict()
    data["control_dt"] = 0.015

    with pytest.raises(RobotContractError, match="control_dt"):
        RobotTaskSpec.from_dict(data)


def test_policy_artifact_accepts_worldlines_checkpoint_uri() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    artifact = PolicyArtifact.from_dict(
        policy_dict(task, checkpoint_uri="worldlines://model_fixture/weights/step-1")
    )

    assert artifact.policy_format == "worldlines"
    assert artifact.checkpoint_uri == "worldlines://model_fixture/weights/step-1"
    assert artifact.task_spec_hash == task.task_hash()


def test_policy_artifact_checkpoint_uri_is_optional_but_task_hash_is_not() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    artifact = PolicyArtifact.from_dict(policy_dict(task))
    assert artifact.checkpoint_uri is None

    data = policy_dict(task)
    data["task_spec_hash"] = ""
    with pytest.raises(RobotContractError, match="task_spec_hash"):
        PolicyArtifact.from_dict(data)
