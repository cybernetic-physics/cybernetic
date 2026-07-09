"""Hosted VLA eval request and record helpers.

These helpers carry lineage and hosted-eval metadata only. They do not import
VLA runtimes, simulator backends, or RobotEnv implementations, and they do not
decide task success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from .contracts import (
    PolicyArtifact,
    RobotContractError,
    RobotTaskSpec,
    _as_dict,
    _non_empty_str,
    stable_hash,
)

VLA_EVAL_RECORD_SCHEMA_VERSION = "robot-vla-eval/v1"


def build_vla_eval_request(
    task_spec: RobotTaskSpec,
    policy_artifact: PolicyArtifact,
    *,
    workspace_id: str,
    policy_artifact_uri: str,
) -> Dict[str, Any]:
    """Build a hosted VLA eval request from stable task/policy artifacts."""

    workspace = _workspace_id(workspace_id)
    artifact_uri = _non_empty_str(policy_artifact_uri, "vla eval policy_artifact_uri")
    _check_policy_matches_task(task_spec, policy_artifact)
    return {
        "workspace_id": workspace,
        "task_spec_uri": policy_artifact.task_spec_uri,
        "task_spec_hash": task_spec.task_hash(),
        "policy_artifact_id": policy_artifact.artifact_id,
        "policy_artifact_uri": artifact_uri,
        "checkpoint_uri": policy_artifact.checkpoint_uri,
        "eval_protocol": task_spec.eval_protocol,
        "observation_schema": policy_artifact.observation_schema,
        "action_schema": policy_artifact.action_schema,
        "policy_kind": policy_artifact.policy_kind,
        "policy_format": policy_artifact.policy_format,
        "inference_runtime": policy_artifact.inference_runtime,
        "control_dt": policy_artifact.control_dt,
        "latency_budget_ms": policy_artifact.latency_budget_ms,
        "action_chunking": policy_artifact.action_chunking,
    }


@dataclass(frozen=True)
class VlaEvalRunRecord:
    schema_version: str
    eval_id: str
    workspace_id: str
    task_spec_uri: str
    task_spec_hash: str
    policy_artifact_id: str
    policy_artifact_uri: str
    checkpoint_uri: Optional[str]
    metrics: Dict[str, Any]
    artifact_uri: str
    created_by_run_id: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VlaEvalRunRecord":
        if data.get("schema_version") != VLA_EVAL_RECORD_SCHEMA_VERSION:
            raise RobotContractError(
                "vla eval record: schema_version must be "
                f"'{VLA_EVAL_RECORD_SCHEMA_VERSION}', got {data.get('schema_version')!r}"
            )
        workspace = _workspace_id(data.get("workspace_id"))
        artifact_uri = _non_empty_str(data.get("artifact_uri"), "vla eval artifact_uri")
        if not artifact_uri.startswith(f"artifact://workspaces/{workspace}/"):
            raise RobotContractError("vla eval artifact_uri must be workspace-scoped")
        return cls(
            schema_version=str(data["schema_version"]),
            eval_id=_non_empty_str(data.get("eval_id"), "vla eval eval_id"),
            workspace_id=workspace,
            task_spec_uri=_non_empty_str(data.get("task_spec_uri"), "vla eval task_spec_uri"),
            task_spec_hash=_non_empty_str(
                data.get("task_spec_hash"), "vla eval task_spec_hash"
            ),
            policy_artifact_id=_non_empty_str(
                data.get("policy_artifact_id"), "vla eval policy_artifact_id"
            ),
            policy_artifact_uri=_non_empty_str(
                data.get("policy_artifact_uri"), "vla eval policy_artifact_uri"
            ),
            checkpoint_uri=data.get("checkpoint_uri"),
            metrics=_as_dict(data.get("metrics", {}), "vla eval metrics"),
            artifact_uri=artifact_uri,
            created_by_run_id=_non_empty_str(
                data.get("created_by_run_id"), "vla eval created_by_run_id"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_vla_eval_record(
    task_spec: RobotTaskSpec,
    policy_artifact: PolicyArtifact,
    *,
    workspace_id: str,
    policy_artifact_uri: str,
    metrics: Mapping[str, Any],
    created_by_run_id: str,
) -> VlaEvalRunRecord:
    """Create a hosted VLA eval record without mutating the policy artifact."""

    request = build_vla_eval_request(
        task_spec,
        policy_artifact,
        workspace_id=workspace_id,
        policy_artifact_uri=policy_artifact_uri,
    )
    workspace = request["workspace_id"]
    eval_id = _eval_id(
        workspace_id=workspace,
        task_spec_hash=task_spec.task_hash(),
        policy_artifact_id=policy_artifact.artifact_id,
        created_by_run_id=created_by_run_id,
    )
    return VlaEvalRunRecord.from_dict(
        {
            "schema_version": VLA_EVAL_RECORD_SCHEMA_VERSION,
            "eval_id": eval_id,
            "workspace_id": workspace,
            "task_spec_uri": request["task_spec_uri"],
            "task_spec_hash": request["task_spec_hash"],
            "policy_artifact_id": request["policy_artifact_id"],
            "policy_artifact_uri": request["policy_artifact_uri"],
            "checkpoint_uri": request["checkpoint_uri"],
            "metrics": dict(metrics),
            "artifact_uri": f"artifact://workspaces/{workspace}/vla-evals/{eval_id}",
            "created_by_run_id": _non_empty_str(
                created_by_run_id, "vla eval created_by_run_id"
            ),
        }
    )


def _workspace_id(value: Any) -> str:
    return _non_empty_str(value, "vla eval workspace_id")


def _check_policy_matches_task(task_spec: RobotTaskSpec, policy_artifact: PolicyArtifact) -> None:
    if policy_artifact.task_spec_hash != task_spec.task_hash():
        raise RobotContractError("vla eval: policy task_spec_hash does not match task spec")
    if policy_artifact.observation_schema != task_spec.observation_space:
        raise RobotContractError("vla eval: policy observation_schema does not match task spec")
    if policy_artifact.action_schema != task_spec.action_space:
        raise RobotContractError("vla eval: policy action_schema does not match task spec")


def _eval_id(
    *,
    workspace_id: str,
    task_spec_hash: str,
    policy_artifact_id: str,
    created_by_run_id: str,
) -> str:
    digest = stable_hash(
        {
            "created_by_run_id": created_by_run_id,
            "policy_artifact_id": policy_artifact_id,
            "task_spec_hash": task_spec_hash,
            "workspace_id": workspace_id,
        }
    )
    return f"vlaeval_{digest[:16]}"
