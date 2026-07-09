"""Isaac/Neko replay metadata validation.

This module is a dependency-light import/replay skeleton. It does not create
sessions, call MCP tools, speak WebRTC, or download captures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from .contracts import PolicyArtifact, RobotContractError, RobotTaskSpec, _non_empty_str

REPLAY_BACKENDS = ("isaac_neko", "isaaclab")


@dataclass(frozen=True)
class ReplayImportRequest:
    task_id: str
    task_spec_uri: str
    task_spec_hash: str
    policy_artifact_id: str
    policy_artifact_uri: str
    checkpoint_uri: Optional[str]
    robot_id: str
    source_backend: str
    target_backend: str
    observation_schema: Dict[str, Any]
    action_schema: Dict[str, Any]
    control_dt: Optional[float]
    render_mode: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_policy_for_replay(
    task_spec: RobotTaskSpec,
    policy_artifact: PolicyArtifact,
    *,
    target_backend: str = "isaac_neko",
) -> None:
    """Reject policy/task mismatches before any replay session starts."""

    if target_backend not in REPLAY_BACKENDS:
        raise RobotContractError(
            f"replay target_backend must be one of {list(REPLAY_BACKENDS)}, got {target_backend!r}"
        )
    if policy_artifact.task_spec_hash != task_spec.task_hash():
        raise RobotContractError("replay: task_spec_hash mismatch")
    if policy_artifact.robot_id != task_spec.robot_id:
        raise RobotContractError("replay: robot_id mismatch")
    if policy_artifact.observation_schema != task_spec.observation_space:
        raise RobotContractError("replay: observation_schema mismatch")
    if policy_artifact.action_schema != task_spec.action_space:
        raise RobotContractError("replay: action_schema mismatch")
    if (
        policy_artifact.control_dt is not None
        and float(policy_artifact.control_dt) != float(task_spec.control_dt)
    ):
        raise RobotContractError("replay: control_dt mismatch")


def build_replay_import_request(
    task_spec: RobotTaskSpec,
    policy_artifact: PolicyArtifact,
    *,
    policy_artifact_uri: str,
    target_backend: str = "isaac_neko",
    render_mode: str = "rgb_array",
    metadata: Mapping[str, Any] | None = None,
) -> ReplayImportRequest:
    """Build a serializable replay import request after metadata validation."""

    validate_policy_for_replay(
        task_spec,
        policy_artifact,
        target_backend=target_backend,
    )
    return ReplayImportRequest(
        task_id=task_spec.task_id,
        task_spec_uri=policy_artifact.task_spec_uri,
        task_spec_hash=task_spec.task_hash(),
        policy_artifact_id=policy_artifact.artifact_id,
        policy_artifact_uri=_non_empty_str(
            policy_artifact_uri, "replay policy_artifact_uri"
        ),
        checkpoint_uri=policy_artifact.checkpoint_uri,
        robot_id=task_spec.robot_id,
        source_backend=policy_artifact.simulator_backend,
        target_backend=target_backend,
        observation_schema=dict(policy_artifact.observation_schema),
        action_schema=dict(policy_artifact.action_schema),
        control_dt=policy_artifact.control_dt,
        render_mode=_non_empty_str(render_mode, "replay render_mode"),
        metadata=dict(metadata or {}),
    )
