"""World-model artifact helpers for Cosmos-style model-plane outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    DATA_PROVENANCE,
    ROBOT_DATASET_SCHEMA_VERSION,
    RobotContractError,
    TrajectoryDatasetArtifact,
    WorldModelArtifact,
    _check_choice,
    _positive_int,
    stable_hash,
)

_FORBIDDEN_MODEL_ARG_KEYS = {
    "robotenv",
    "robot_env",
    "env",
    "environment",
    "reward_spec",
    "success_metric",
    "termination",
    "eval_protocol",
    "simulator",
    "simulator_handle",
    "callback",
    "callbacks",
    "live_env",
}


def build_cosmos_world_model_payload(
    dataset: TrajectoryDatasetArtifact,
    *,
    model_args: Mapping[str, Any] | None = None,
    model_family: str = "cosmos3",
) -> dict[str, Any]:
    """Build a Cosmos model-plane payload from a dataset artifact."""

    return {
        "dataset_uri": dataset.storage_uri,
        "observation_schema": copy.deepcopy(dataset.observation_schema),
        "action_schema": copy.deepcopy(dataset.action_schema),
        "episode_count": int(dataset.episode_count),
        "frame_count": int(dataset.frame_count),
        "model_family": str(model_family),
        "data_provenance": dataset.data_provenance,
        "model_args": _sanitize_model_args(model_args),
    }


def create_synthetic_dataset_from_world_model(
    world_model: WorldModelArtifact,
    source_dataset: TrajectoryDatasetArtifact,
    storage_uri: str | Path,
    *,
    frame_count: int,
    episode_count: int = 1,
    data_provenance: str = "synthetic",
    artifact_refs: list[dict[str, Any]] | None = None,
) -> TrajectoryDatasetArtifact:
    """Create a dataset artifact for generated world-model outputs."""

    _check_world_model_matches_dataset(world_model, source_dataset)
    _check_choice(data_provenance, DATA_PROVENANCE, "world model dataset data_provenance")
    refs = list(artifact_refs or [])
    if data_provenance == "mixed":
        _check_mixed_refs(refs)
    elif data_provenance != "synthetic":
        raise RobotContractError(
            "world model generated datasets must be synthetic unless mixed "
            "provenance is explicit"
        )
    frame_total = _positive_int(frame_count, "world model dataset frame_count")
    episode_total = _positive_int(episode_count, "world model dataset episode_count")
    base_refs = [
        {
            "kind": "world_model",
            "artifact_id": world_model.artifact_id,
            "uri": world_model.model_uri,
            "data_provenance": "synthetic",
        },
        {
            "kind": "source_dataset",
            "artifact_id": source_dataset.artifact_id,
            "uri": source_dataset.storage_uri,
            "data_provenance": source_dataset.data_provenance,
        },
    ]
    artifact_id = _generated_dataset_id(
        world_model=world_model,
        source_dataset=source_dataset,
        storage_uri=str(storage_uri),
        data_provenance=data_provenance,
    )
    return TrajectoryDatasetArtifact.from_dict(
        {
            "schema_version": ROBOT_DATASET_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "task_spec_uri": world_model.task_spec_uri,
            "task_spec_hash": world_model.task_spec_hash,
            "source_backend": source_dataset.source_backend,
            "source_runs": [world_model.created_by_run_id],
            "observation_schema": source_dataset.observation_schema,
            "action_schema": source_dataset.action_schema,
            "episode_count": episode_total,
            "frame_count": frame_total,
            "storage_uri": str(storage_uri),
            "data_provenance": data_provenance,
            "artifact_refs": base_refs + refs,
        }
    )


def _sanitize_model_args(model_args: Mapping[str, Any] | None) -> dict[str, Any]:
    if model_args is None:
        return {}
    copied = copy.deepcopy(dict(model_args))
    forbidden = sorted(
        key for key in copied if str(key).lower().replace("-", "_") in _FORBIDDEN_MODEL_ARG_KEYS
    )
    if forbidden:
        raise RobotContractError(
            "world model model_args must not contain task/runtime fields: "
            + ", ".join(forbidden)
        )
    try:
        json.dumps(copied, sort_keys=True)
    except TypeError as exc:
        raise RobotContractError("world model model_args must be JSON-serializable") from exc
    return copied


def _check_world_model_matches_dataset(
    world_model: WorldModelArtifact, source_dataset: TrajectoryDatasetArtifact
) -> None:
    if world_model.task_spec_hash != source_dataset.task_spec_hash:
        raise RobotContractError(
            "world model task_spec_hash does not match source dataset"
        )
    if world_model.source_dataset_artifact_id != source_dataset.artifact_id:
        raise RobotContractError(
            "world model source_dataset_artifact_id does not match source dataset"
        )


def _check_mixed_refs(refs: list[dict[str, Any]]) -> None:
    provenances = {str(ref.get("data_provenance")) for ref in refs}
    has_grounded = bool(provenances & {"sim", "real"})
    if "synthetic" not in provenances or not has_grounded:
        raise RobotContractError(
            "mixed world-model datasets require explicit synthetic and sim/real refs"
        )


def _generated_dataset_id(
    *,
    world_model: WorldModelArtifact,
    source_dataset: TrajectoryDatasetArtifact,
    storage_uri: str,
    data_provenance: str,
) -> str:
    digest = stable_hash(
        {
            "data_provenance": data_provenance,
            "source_dataset_artifact_id": source_dataset.artifact_id,
            "storage_uri": storage_uri,
            "world_model_artifact_id": world_model.artifact_id,
        }
    )
    return f"tds_world_model_{digest[:16]}"
