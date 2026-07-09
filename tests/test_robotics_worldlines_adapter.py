from __future__ import annotations

import builtins
import importlib
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from cybernetics.robotics.contracts import (
    ROBOT_DATASET_SCHEMA_VERSION,
    PolicyArtifact,
    TrajectoryDatasetArtifact,
)


def _worldlines_module():
    return importlib.import_module("cybernetics.robotics.worldlines")


def _dataset(*, source_backend: str = "fixture") -> TrajectoryDatasetArtifact:
    return TrajectoryDatasetArtifact.from_dict(
        {
            "schema_version": ROBOT_DATASET_SCHEMA_VERSION,
            "artifact_id": "tds_goal5_fixture",
            "task_spec_uri": "task://fixture_walk",
            "task_spec_hash": "dataset-task-hash",
            "source_backend": source_backend,
            "source_runs": ["rrun_a", "rrun_b"],
            "observation_schema": {
                "base_velocity": {"dtype": "float32", "shape": [3]},
                "joints": {"dtype": "float32", "shape": [12]},
            },
            "action_schema": {"joint_targets": {"dtype": "float32", "shape": [12]}},
            "episode_count": 2,
            "frame_count": 128,
            "storage_uri": "artifact://datasets/goal5/fixture_walk.jsonl",
            "data_provenance": "sim",
            "artifact_refs": [{"kind": "rollout", "uri": "artifact://rollouts/rrun_a.json"}],
        }
    )


@dataclass
class RecordingWorldlinesClient:
    response: Mapping[str, Any]
    payloads: list[Mapping[str, Any]] = field(default_factory=list)

    def train(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.payloads.append(payload)
        return self.response


def test_mock_client_consumes_dataset_and_emits_policy_artifact() -> None:
    from cybernetics.robotics import WorldlinesModelPlaneAdapter

    dataset = _dataset()
    client = RecordingWorldlinesClient(
        {
            "job_id": "wtrain_goal5_001",
            "checkpoint_uri": "worldlines://checkpoints/wtrain_goal5_001/final",
            "backend_version": "worldlines-mock/test",
            "eval_metrics": {"train_loss": 0.125},
        }
    )

    policy = WorldlinesModelPlaneAdapter(client).train_policy(
        dataset,
        task_spec_hash="caller-task-hash",
        robot_id="fixture_bot",
        model_args={"family": "tiny-world-action-model", "horizon": 16},
    )

    assert isinstance(policy, PolicyArtifact)
    assert policy.policy_format == "worldlines"
    assert policy.checkpoint_uri == "worldlines://checkpoints/wtrain_goal5_001/final"
    assert policy.observation_schema == dataset.observation_schema
    assert policy.action_schema == dataset.action_schema
    assert policy.observation_schema is not dataset.observation_schema
    assert policy.action_schema is not dataset.action_schema
    assert policy.rollout_artifacts == [dataset.storage_uri]
    assert policy.created_by_run_id == "wtrain_goal5_001"


def test_checkpoint_uri_does_not_become_task_semantics() -> None:
    module = _worldlines_module()
    dataset = _dataset(source_backend="mujoco")
    client = RecordingWorldlinesClient(
        {
            "job_id": "wtrain_goal5_isaac_named_checkpoint",
            "checkpoint_uri": "worldlines://isaaclab/unitree/g1/not-task-semantics",
        }
    )

    policy = module.train_worldlines_policy(
        client,
        dataset,
        task_spec_hash="hash-provided-by-caller",
        robot_id="robot-provided-by-caller",
        model_args={"family": "mock"},
    )

    assert policy.simulator_backend == "mujoco"
    assert policy.task_spec_hash == "hash-provided-by-caller"
    assert policy.robot_id == "robot-provided-by-caller"
    assert policy.task_spec_uri == dataset.task_spec_uri
    assert policy.checkpoint_uri == "worldlines://isaaclab/unitree/g1/not-task-semantics"


def test_adapter_import_stays_runtime_dependency_free(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked_roots = {
        "cosmos",
        "isaac",
        "isaaclab",
        "isaacsim",
        "locomujoco",
        "mujoco",
        "rclpy",
        "rlmesh",
        "ros2",
        "rospy",
        "unitree",
        "worldlines",
        "worldlines_backend",
    }
    attempted: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if level == 0 and root in blocked_roots:
            attempted.append(name)
            raise AssertionError(f"runtime package import attempted: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("cybernetics.robotics.worldlines", None)

    module = importlib.import_module("cybernetics.robotics.worldlines")

    assert module.WorldlinesModelPlaneAdapter is not None
    assert attempted == []


def test_payload_contains_dataset_metadata_and_model_args_only() -> None:
    module = _worldlines_module()
    dataset = _dataset(source_backend="isaaclab")
    client = RecordingWorldlinesClient(
        {
            "job_id": "wtrain_payload_001",
            "checkpoint_uri": "worldlines://checkpoints/wtrain_payload_001/final",
        }
    )

    module.WorldlinesModelPlaneAdapter(client).train_policy(
        dataset,
        task_spec_hash="caller-task-hash",
        robot_id="caller-robot",
        model_args={"family": "mock-worldlines", "layers": 2},
    )

    assert len(client.payloads) == 1
    payload = dict(client.payloads[0])
    assert set(payload) == {
        "dataset_uri",
        "observation_schema",
        "action_schema",
        "episode_count",
        "frame_count",
        "model_args",
    }
    assert payload["dataset_uri"] == dataset.storage_uri
    assert payload["observation_schema"] == dataset.observation_schema
    assert payload["action_schema"] == dataset.action_schema
    assert payload["episode_count"] == dataset.episode_count
    assert payload["frame_count"] == dataset.frame_count
    assert payload["model_args"] == {"family": "mock-worldlines", "layers": 2}

    serialized_payload = repr(payload)
    for forbidden in (
        "RobotEnv",
        "reward_spec",
        "success_metric",
        "termination",
        "simulator_handle",
        "callback",
        "live_env",
        "source_backend",
        "source_runs",
        "task_spec_uri",
    ):
        assert forbidden not in serialized_payload
