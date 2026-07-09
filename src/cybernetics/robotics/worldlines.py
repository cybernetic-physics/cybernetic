"""Mocked Worldlines model-plane adapter for RobotTask datasets.

This module defines the SDK-side contract only. It intentionally avoids real
Worldlines, simulator, robotics middleware, and training runtime imports.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .contracts import (
    ROBOT_POLICY_SCHEMA_VERSION,
    PolicyArtifact,
    TrajectoryDatasetArtifact,
    stable_hash,
)


class WorldlinesAdapterError(ValueError):
    """The mocked Worldlines adapter received or returned malformed data."""


class WorldlinesModelPlaneClient(Protocol):
    """Minimal client boundary for a mocked model-plane training request."""

    def train(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Start a mocked training job and return checkpoint metadata."""


_FORBIDDEN_MODEL_ARG_KEYS = {
    "robotenv",
    "robot_env",
    "env",
    "environment",
    "reward_spec",
    "success_metric",
    "termination",
    "simulator",
    "simulator_handle",
    "simulator_handles",
    "callback",
    "callbacks",
    "live_env",
    "live_env_object",
}


@dataclass(frozen=True)
class WorldlinesTrainingConfig:
    """Serializable training knobs accepted by the mocked adapter."""

    model_args: Mapping[str, Any] = field(default_factory=dict)
    backend_version: str = "worldlines-mock/v1"
    policy_kind: str = "world_action_model"


def _copy_mapping(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldlinesAdapterError(f"{where}: expected mapping, got {type(value).__name__}")
    return copy.deepcopy(dict(value))


def _sanitize_model_args(model_args: Mapping[str, Any] | None) -> dict[str, Any]:
    if model_args is None:
        return {}
    copied = _copy_mapping(model_args, "model_args")
    forbidden = sorted(
        key for key in copied if str(key).lower().replace("-", "_") in _FORBIDDEN_MODEL_ARG_KEYS
    )
    if forbidden:
        raise WorldlinesAdapterError(
            "model_args must not contain task/runtime fields: " + ", ".join(forbidden)
        )
    try:
        json.dumps(copied, sort_keys=True)
    except TypeError as exc:
        raise WorldlinesAdapterError("model_args must be JSON-serializable") from exc
    return copied


def build_worldlines_training_payload(
    dataset: TrajectoryDatasetArtifact,
    *,
    model_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the model-plane payload without task semantics or live env handles."""

    return {
        "dataset_uri": dataset.storage_uri,
        "observation_schema": copy.deepcopy(dataset.observation_schema),
        "action_schema": copy.deepcopy(dataset.action_schema),
        "episode_count": int(dataset.episode_count),
        "frame_count": int(dataset.frame_count),
        "model_args": _sanitize_model_args(model_args),
    }


def _required_string(response: Mapping[str, Any], key: str) -> str:
    value = response.get(key)
    if not isinstance(value, str) or not value:
        raise WorldlinesAdapterError(f"Worldlines response field '{key}' must be a non-empty string")
    return value


def _optional_string(response: Mapping[str, Any], key: str) -> str | None:
    value = response.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise WorldlinesAdapterError(f"Worldlines response field '{key}' must be a non-empty string")
    return value


def _response_metrics(response: Mapping[str, Any]) -> dict[str, Any]:
    value = response.get("eval_metrics", {})
    if value is None:
        return {}
    return _copy_mapping(value, "Worldlines response eval_metrics")


def _artifact_id(dataset: TrajectoryDatasetArtifact, job_id: str, checkpoint_uri: str) -> str:
    digest = stable_hash(
        {
            "checkpoint_uri": checkpoint_uri,
            "dataset_uri": dataset.storage_uri,
            "job_id": job_id,
        }
    )
    return f"pol_worldlines_{digest[:16]}"


@dataclass(frozen=True)
class WorldlinesModelPlaneAdapter:
    """Dependency-light adapter from trajectory datasets to policy artifacts."""

    client: WorldlinesModelPlaneClient
    config: WorldlinesTrainingConfig = field(default_factory=WorldlinesTrainingConfig)

    def train_policy(
        self,
        dataset: TrajectoryDatasetArtifact,
        *,
        task_spec_hash: str,
        robot_id: str,
        model_args: Mapping[str, Any] | None = None,
        task_spec_uri: str | None = None,
    ) -> PolicyArtifact:
        """Submit a mocked training request and return the resulting policy artifact."""

        merged_model_args = dict(self.config.model_args)
        if model_args:
            merged_model_args.update(dict(model_args))
        request = build_worldlines_training_payload(dataset, model_args=merged_model_args)
        raw_response = self.client.train(request)
        if not isinstance(raw_response, Mapping):
            raise WorldlinesAdapterError(
                "Worldlines client train() must return checkpoint metadata as a mapping"
            )

        response = dict(raw_response)
        job_id = _required_string(response, "job_id")
        checkpoint_uri = _required_string(response, "checkpoint_uri")
        artifact_id = _optional_string(response, "artifact_id") or _artifact_id(
            dataset, job_id, checkpoint_uri
        )
        backend_version = _optional_string(response, "backend_version") or self.config.backend_version

        return PolicyArtifact.from_dict(
            {
                "schema_version": ROBOT_POLICY_SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "task_spec_uri": task_spec_uri or dataset.task_spec_uri,
                "task_spec_hash": task_spec_hash,
                "checkpoint_uri": checkpoint_uri,
                "policy_format": "worldlines",
                "observation_schema": copy.deepcopy(dataset.observation_schema),
                "action_schema": copy.deepcopy(dataset.action_schema),
                "robot_id": robot_id,
                "simulator_backend": dataset.source_backend,
                "backend_version": backend_version,
                "eval_metrics": _response_metrics(response),
                "rollout_artifacts": [dataset.storage_uri],
                "created_by_run_id": job_id,
                "policy_kind": self.config.policy_kind,
            }
        )


def train_worldlines_policy(
    client: WorldlinesModelPlaneClient,
    dataset: TrajectoryDatasetArtifact,
    *,
    task_spec_hash: str,
    robot_id: str,
    model_args: Mapping[str, Any] | None = None,
    config: WorldlinesTrainingConfig | None = None,
    task_spec_uri: str | None = None,
) -> PolicyArtifact:
    """Convenience wrapper for one mocked dataset-to-policy training request."""

    adapter = WorldlinesModelPlaneAdapter(
        client=client,
        config=config or WorldlinesTrainingConfig(),
    )
    return adapter.train_policy(
        dataset,
        task_spec_hash=task_spec_hash,
        robot_id=robot_id,
        model_args=model_args,
        task_spec_uri=task_spec_uri,
    )
