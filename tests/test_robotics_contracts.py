from __future__ import annotations

import pytest

from cybernetics.robotics import (
    ROBOT_DATASET_SCHEMA_VERSION,
    ROBOT_POLICY_SCHEMA_VERSION,
    ROBOT_TASK_SCHEMA_VERSION,
    PolicyArtifact,
    RobotContractError,
    RobotTaskSpec,
    TrajectoryDatasetArtifact,
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


def dataset_dict(task: RobotTaskSpec) -> dict:
    return {
        "schema_version": ROBOT_DATASET_SCHEMA_VERSION,
        "artifact_id": "tds_fixture",
        "task_spec_uri": f"task://{task.task_id}",
        "task_spec_hash": task.task_hash(),
        "source_backend": task.simulator_backend,
        "source_runs": ["rrun_fixture"],
        "observation_schema": task.observation_space,
        "action_schema": task.action_space,
        "episode_count": 1,
        "frame_count": 3,
        "storage_uri": "artifact://datasets/tds_fixture",
        "data_provenance": "sim",
        "artifact_refs": [
            {"kind": "rollout", "run_id": "rrun_fixture", "uri": "artifact://rollout.json"}
        ],
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


def test_trajectory_dataset_artifact_round_trips_with_sim_provenance() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    artifact = TrajectoryDatasetArtifact.from_dict(dataset_dict(task))

    assert artifact.to_dict() == dataset_dict(task)
    assert artifact.source_backend == "fixture"
    assert artifact.data_provenance == "sim"
    assert artifact.source_runs == ["rrun_fixture"]
    assert artifact.observation_schema == task.observation_space
    assert artifact.action_schema == task.action_space
    assert artifact.task_spec_hash == task.task_hash()


def test_trajectory_dataset_artifact_rejects_invalid_backend_and_provenance() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    bad_backend = dataset_dict(task)
    bad_backend["source_backend"] = "worldlines"

    with pytest.raises(RobotContractError, match="source_backend"):
        TrajectoryDatasetArtifact.from_dict(bad_backend)

    bad_provenance = dataset_dict(task)
    bad_provenance["data_provenance"] = "teleop"
    with pytest.raises(RobotContractError, match="data_provenance"):
        TrajectoryDatasetArtifact.from_dict(bad_provenance)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_runs", [], "source_runs"),
        ("storage_uri", "", "storage_uri"),
        ("episode_count", 0, "episode_count"),
        ("frame_count", 0, "frame_count"),
        ("task_spec_hash", "", "task_spec_hash"),
    ],
)
def test_trajectory_dataset_artifact_requires_owned_source_and_positive_counts(
    field: str, value, match: str
) -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    data = dataset_dict(task)
    data[field] = value

    with pytest.raises(RobotContractError, match=match):
        TrajectoryDatasetArtifact.from_dict(data)
