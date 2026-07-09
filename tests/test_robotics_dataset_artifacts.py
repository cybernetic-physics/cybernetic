from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_robotics_contracts import task_dict

from cybernetics.robotics import (
    FixtureRobotEnv,
    RobotContractError,
    RobotTaskSpec,
    TrajectoryDatasetArtifact,
    create_trajectory_dataset_from_runs,
    run_robot_episode,
    write_trajectory_dataset_artifact,
)


def test_fixture_rollout_dataset_is_emitted_from_runner_artifacts(tmp_path) -> None:
    spec = RobotTaskSpec.from_dict(task_dict())
    record = run_robot_episode(spec, FixtureRobotEnv(), tmp_path / "run", seed=42)

    dataset = create_trajectory_dataset_from_runs(spec, [record], tmp_path / "dataset")

    assert dataset.source_runs == [record.run_id]
    assert dataset.task_spec_uri == f"task://{spec.task_id}"
    assert dataset.task_spec_hash == spec.task_hash()
    assert dataset.source_backend == spec.simulator_backend
    assert dataset.episode_count == 1
    assert dataset.storage_uri == str(tmp_path / "dataset")

    rollout = json.loads((tmp_path / "run" / "rollout.json").read_text())
    assert dataset.frame_count == len(rollout["steps"])
    assert "action" in rollout["steps"][0]

    saved = TrajectoryDatasetArtifact.from_dict(
        json.loads((tmp_path / "dataset" / "trajectory_dataset.json").read_text())
    )
    assert saved == dataset

    explicit_path = write_trajectory_dataset_artifact(tmp_path / "copy.json", dataset)
    assert explicit_path.exists()


def test_dataset_emission_rejects_mismatched_task_hashes(tmp_path) -> None:
    spec = RobotTaskSpec.from_dict(task_dict())
    record = run_robot_episode(spec, FixtureRobotEnv(), tmp_path / "run", seed=1)
    mismatched = replace(record, task_spec_hash="other")

    with pytest.raises(RobotContractError, match="task_spec_hash"):
        create_trajectory_dataset_from_runs(spec, [mismatched], tmp_path / "dataset")


def test_dataset_emission_works_after_source_env_is_closed(tmp_path) -> None:
    spec = RobotTaskSpec.from_dict(task_dict())
    env = FixtureRobotEnv()
    record = run_robot_episode(spec, env, tmp_path / "run", seed=2)

    assert env.closed is True
    dataset = create_trajectory_dataset_from_runs(spec, [record], tmp_path / "dataset")

    assert dataset.source_runs == [record.run_id]
    assert dataset.frame_count > 0


def test_dataset_artifact_contains_only_serialized_references(tmp_path) -> None:
    spec = RobotTaskSpec.from_dict(task_dict())
    record = run_robot_episode(spec, FixtureRobotEnv(), tmp_path / "run", seed=3)
    dataset = create_trajectory_dataset_from_runs(spec, [record], tmp_path / "dataset")

    encoded = json.dumps(dataset.to_dict(), sort_keys=True)

    assert record.run_id in encoded
    assert str(tmp_path / "dataset") in encoded
    assert "FixtureRobotEnv" not in encoded
    assert "RobotEnv" not in encoded
    assert "LocoMuJoCoRobotEnv" not in encoded
    assert "callback" not in encoded
    assert "client" not in encoded
