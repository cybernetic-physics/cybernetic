from __future__ import annotations

import builtins
import importlib
import json
import sys
from dataclasses import replace
from typing import Any, Mapping

import pytest
from test_robotics_contracts import policy_dict, task_dict

from cybernetics.robotics import (
    PolicyArtifact,
    RobotContractError,
    RobotTaskSpec,
    VlaEvalRunRecord,
    build_vla_eval_request,
    create_vla_eval_record,
)


def _vla_policy(task: RobotTaskSpec) -> PolicyArtifact:
    data = policy_dict(task, checkpoint_uri="worldlines://policies/vla_fixture/final")
    data.update(
        {
            "policy_kind": "vla_policy",
            "inference_runtime": "worldlines",
            "control_dt": 0.2,
            "latency_budget_ms": 150,
            "action_chunking": {
                "kind": "receding_horizon",
                "chunk_steps": 8,
                "stride_steps": 2,
            },
        }
    )
    return PolicyArtifact.from_dict(data)


def test_vla_policy_artifact_round_trips_runtime_and_chunk_metadata() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    artifact = _vla_policy(task)
    round_trip = PolicyArtifact.from_dict(artifact.to_dict())

    assert round_trip.policy_kind == "vla_policy"
    assert round_trip.inference_runtime == "worldlines"
    assert round_trip.control_dt == 0.2
    assert round_trip.latency_budget_ms == 150
    assert round_trip.action_chunking == {
        "kind": "receding_horizon",
        "chunk_steps": 8,
        "stride_steps": 2,
    }
    assert round_trip.task_spec_hash == task.task_hash()
    assert round_trip.observation_schema == task.observation_space
    assert round_trip.action_schema == task.action_space


@pytest.mark.parametrize(
    ("field_path", "value", "match"),
    [
        (("control_dt",), 0, "control_dt"),
        (("latency_budget_ms",), 0, "latency_budget_ms"),
        (("action_chunking", "chunk_steps"), 0, "chunk_steps"),
        (("action_chunking", "stride_steps"), 0, "stride_steps"),
        (("action_chunking", "stride_steps"), 9, "stride_steps"),
    ],
)
def test_vla_policy_artifact_rejects_invalid_timing_metadata(
    field_path: tuple[str, ...], value: Any, match: str
) -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    data = _vla_policy(task).to_dict()
    target = data
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value

    with pytest.raises(RobotContractError, match=match):
        PolicyArtifact.from_dict(data)


def test_vla_eval_request_uses_eval_protocol_and_policy_lineage_only() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    policy = _vla_policy(task)

    request = build_vla_eval_request(
        task,
        policy,
        workspace_id="ws_123",
        policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
    )

    assert request["workspace_id"] == "ws_123"
    assert request["task_spec_uri"] == f"task://{task.task_id}"
    assert request["task_spec_hash"] == task.task_hash()
    assert request["policy_artifact_id"] == policy.artifact_id
    assert request["checkpoint_uri"] == policy.checkpoint_uri
    assert request["eval_protocol"] == task.eval_protocol
    assert request["observation_schema"] == task.observation_space
    assert request["action_schema"] == task.action_space
    assert request["action_chunking"] == policy.action_chunking

    encoded = json.dumps(request, sort_keys=True)
    for forbidden in (
        "reward_spec",
        "success_metric",
        "termination",
        "RobotEnv",
        "simulator_handle",
        "callback",
        "live_env",
    ):
        assert forbidden not in encoded


def test_vla_eval_record_emits_metrics_without_deciding_task_success() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    policy = _vla_policy(task)
    original_policy = policy.to_dict()

    record = create_vla_eval_record(
        task,
        policy,
        workspace_id="ws_123",
        policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
        metrics={"success_rate": 0.75, "mean_latency_ms": 88, "episodes": 4},
        created_by_run_id="vla_eval_job_001",
    )

    assert isinstance(record, VlaEvalRunRecord)
    assert record.policy_artifact_id == policy.artifact_id
    assert record.metrics["success_rate"] == 0.75
    assert record.metrics["mean_latency_ms"] == 88
    assert record.metrics["episodes"] == 4
    assert record.artifact_uri.startswith("artifact://workspaces/ws_123/vla-evals/")
    assert "status" not in record.to_dict()
    assert policy.to_dict() == original_policy


def test_vla_eval_requires_workspace_scoped_ownership() -> None:
    task = RobotTaskSpec.from_dict(task_dict())
    policy = _vla_policy(task)

    with pytest.raises(RobotContractError, match="workspace_id"):
        build_vla_eval_request(
            task,
            policy,
            workspace_id="",
            policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
        )

    mismatched = replace(policy, task_spec_hash="other")
    with pytest.raises(RobotContractError, match="task_spec_hash"):
        build_vla_eval_request(
            task,
            mismatched,
            workspace_id="ws_123",
            policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
        )

    data = create_vla_eval_record(
        task,
        policy,
        workspace_id="ws_123",
        policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
        metrics={},
        created_by_run_id="vla_eval_job_002",
    ).to_dict()
    data["artifact_uri"] = "artifact://workspaces/other/vla-evals/eval_1"
    with pytest.raises(RobotContractError, match="workspace"):
        VlaEvalRunRecord.from_dict(data)


def test_vla_eval_import_stays_runtime_dependency_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_roots = {
        "cosmos",
        "isaac",
        "isaaclab",
        "isaacsim",
        "locomujoco",
        "mujoco",
        "omni",
        "pxr",
        "rclpy",
        "rlmesh",
        "ros2",
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
    sys.modules.pop("cybernetics.robotics.vla_eval", None)
    importlib.import_module("cybernetics.robotics.vla_eval")

    assert attempted == []
