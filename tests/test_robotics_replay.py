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
    ReplayImportRequest,
    RobotContractError,
    RobotTaskSpec,
    build_replay_import_request,
    validate_policy_for_replay,
)


def _task() -> RobotTaskSpec:
    return RobotTaskSpec.from_dict(task_dict())


def _policy(task: RobotTaskSpec) -> PolicyArtifact:
    data = policy_dict(task, checkpoint_uri="artifact://policies/fixture.pt")
    data["control_dt"] = task.control_dt
    return PolicyArtifact.from_dict(data)


def test_replay_accepts_fixture_policy_with_matching_schema() -> None:
    task = _task()
    policy = _policy(task)

    request = build_replay_import_request(
        task,
        policy,
        policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
    )

    assert isinstance(request, ReplayImportRequest)
    assert request.target_backend == "isaac_neko"
    assert request.source_backend == "fixture"
    assert request.task_spec_hash == task.task_hash()
    assert request.robot_id == task.robot_id
    assert request.observation_schema == task.observation_space
    assert request.action_schema == task.action_space


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("task_spec_hash", "other", "task_spec_hash"),
        ("robot_id", "other_bot", "robot_id"),
        ("observation_schema", {"other": {}}, "observation_schema"),
        ("action_schema", {"other": {}}, "action_schema"),
        ("control_dt", 999.0, "control_dt"),
    ],
)
def test_replay_rejects_policy_task_mismatches(
    field: str, replacement: Any, match: str
) -> None:
    task = _task()
    policy = replace(_policy(task), **{field: replacement})

    with pytest.raises(RobotContractError, match=match):
        validate_policy_for_replay(task, policy)


def test_replay_rejects_invalid_target_backend() -> None:
    task = _task()
    policy = _policy(task)

    with pytest.raises(RobotContractError, match="target_backend"):
        validate_policy_for_replay(task, policy, target_backend="cosmos")


def test_replay_request_is_metadata_only_and_does_not_start_runtime() -> None:
    task = _task()
    policy = _policy(task)

    request = build_replay_import_request(
        task,
        policy,
        policy_artifact_uri="artifact://workspaces/ws_123/policies/pol_fixture.json",
        metadata={"purpose": "visual_validation"},
    )
    encoded = json.dumps(request.to_dict(), sort_keys=True)

    assert "session_id" not in encoded
    assert "viewer_url" not in encoded
    for forbidden in (
        "httpx",
        "MCP",
        "WebRTC",
        "RobotEnv",
        "callback",
        "credential",
        "warm_pool",
    ):
        assert forbidden not in encoded


def test_replay_import_stays_runtime_dependency_free(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked_roots = {
        "cybernetics.behavior_ci",
        "httpx",
        "isaac",
        "isaac_mcp",
        "isaaclab",
        "isaacsim",
        "locomujoco",
        "mujoco",
        "neko",
        "omni",
        "pxr",
        "rclpy",
        "respx",
        "unitree",
        "worldlines",
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
        if level == 0 and any(name == root or name.startswith(f"{root}.") for root in blocked_roots):
            attempted.append(name)
            raise AssertionError(f"runtime package import attempted: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("cybernetics.robotics.replay", None)
    importlib.import_module("cybernetics.robotics.replay")

    assert attempted == []
