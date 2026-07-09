from __future__ import annotations

import json

import pytest
from test_robotics_contracts import task_dict

from cybernetics import Client
from cybernetics.robotics import (
    PolicyArtifact,
    RobotContractError,
    RobotRunRecord,
    RobotTasksClient,
    RobotTaskSpec,
)


def test_client_exposes_robot_tasks_without_sim_import() -> None:
    client = Client(api_key="cp_live_test", base_url="https://api.test")

    assert isinstance(client.robot_tasks, RobotTasksClient)
    assert client.robot_tasks is client.robot_tasks


def test_robot_tasks_load_save_validate_and_run_fixture(tmp_path) -> None:
    client = RobotTasksClient()
    spec = client.validate(task_dict())
    spec_path = tmp_path / "task.json"

    client.save(spec, spec_path)
    loaded = client.load(spec_path)
    assert loaded.task_hash() == spec.task_hash()

    result = client.run_fixture(loaded, tmp_path / "run", seed=42)
    assert result.run_record.status == "succeeded"
    assert result.run_record.task_spec_hash == loaded.task_hash()
    assert result.metrics_path.exists()
    assert result.rollout_path.exists()

    saved = RobotRunRecord.from_dict(json.loads(result.run_record_path.read_text()))
    assert saved.run_id == result.run_record.run_id


def test_robot_tasks_policy_artifact_uses_task_lineage(tmp_path) -> None:
    client = RobotTasksClient()
    spec = RobotTaskSpec.from_dict(task_dict())
    run = client.run_fixture(spec, tmp_path / "run", seed=1)

    artifact = client.policy_artifact(
        spec,
        artifact_id="pol_fixture",
        created_by_run_id=run.run_record.run_id,
        checkpoint_uri="worldlines://fixture/ckpt",
        policy_format="worldlines",
        eval_metrics={"success_rate": 1.0},
        rollout_artifacts=[str(run.rollout_path)],
        control_dt=spec.control_dt,
    )

    assert isinstance(artifact, PolicyArtifact)
    assert artifact.task_spec_hash == spec.task_hash()
    assert artifact.observation_schema == spec.observation_space
    assert artifact.action_schema == spec.action_space
    assert artifact.created_by_run_id == run.run_record.run_id

    output = client.write_policy_artifact(artifact, tmp_path / "policy.json")
    reloaded = PolicyArtifact.from_dict(json.loads(output.read_text()))
    assert reloaded.task_spec_hash == spec.task_hash()


def test_robot_tasks_validate_rejects_bad_specs() -> None:
    data = task_dict()
    data["simulator_backend"] = "not-a-backend"

    with pytest.raises(RobotContractError, match="simulator_backend"):
        RobotTasksClient().validate(data)
