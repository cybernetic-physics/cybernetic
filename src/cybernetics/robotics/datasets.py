"""Trajectory dataset artifact helpers.

Dataset helpers consume serialized run outputs. They do not hold RobotEnv
instances, simulator handles, model clients, callbacks, or live runtime state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import write_trajectory_dataset_artifact
from .contracts import (
    DATA_PROVENANCE,
    ROBOT_DATASET_SCHEMA_VERSION,
    RobotContractError,
    RobotRunRecord,
    RobotTaskSpec,
    TrajectoryDatasetArtifact,
    _check_choice,
    stable_hash,
)


def create_trajectory_dataset_from_runs(
    task_spec: RobotTaskSpec,
    run_records: Iterable[RobotRunRecord],
    dataset_dir: str | Path,
    *,
    data_provenance: str = "sim",
) -> TrajectoryDatasetArtifact:
    """Create a dataset artifact from already-written run artifacts."""

    _check_choice(data_provenance, DATA_PROVENANCE, "trajectory dataset data_provenance")
    records = list(run_records)
    if not records:
        raise RobotContractError("trajectory dataset: run_records must be non-empty")

    task_hash = task_spec.task_hash()
    source_runs: list[str] = []
    artifact_refs: list[dict[str, Any]] = []
    frame_count = 0

    for record in records:
        if record.task_spec_hash != task_hash:
            raise RobotContractError(
                "trajectory dataset: task_spec_hash mismatch for "
                f"run {record.run_id}"
            )
        source_runs.append(record.run_id)
        rollout_path = Path(record.logs_uri)
        metrics_path = Path(record.metrics_uri)
        steps = _read_rollout_steps(rollout_path)
        frame_count += len(steps)
        artifact_refs.append(
            {"kind": "rollout", "run_id": record.run_id, "uri": str(rollout_path)}
        )
        artifact_refs.append(
            {"kind": "metrics", "run_id": record.run_id, "uri": str(metrics_path)}
        )

    if frame_count <= 0:
        raise RobotContractError("trajectory dataset: frame_count must be positive")

    output = Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_id = _dataset_artifact_id(
        task_hash=task_hash,
        source_runs=source_runs,
        data_provenance=data_provenance,
    )
    artifact = TrajectoryDatasetArtifact.from_dict(
        {
            "schema_version": ROBOT_DATASET_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "task_spec_uri": f"task://{task_spec.task_id}",
            "task_spec_hash": task_hash,
            "source_backend": task_spec.simulator_backend,
            "source_runs": source_runs,
            "observation_schema": task_spec.observation_space,
            "action_schema": task_spec.action_space,
            "episode_count": len(records),
            "frame_count": frame_count,
            "storage_uri": str(output),
            "data_provenance": data_provenance,
            "artifact_refs": artifact_refs,
        }
    )
    write_trajectory_dataset_artifact(output / "trajectory_dataset.json", artifact)
    return artifact


def _dataset_artifact_id(
    *, task_hash: str, source_runs: list[str], data_provenance: str
) -> str:
    digest = stable_hash(
        {
            "data_provenance": data_provenance,
            "source_runs": source_runs,
            "task_spec_hash": task_hash,
        }
    )
    return f"tds_{digest[:16]}"


def _read_rollout_steps(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RobotContractError(f"trajectory dataset: missing rollout {path}") from exc
    if not isinstance(payload, Mapping):
        raise RobotContractError("trajectory dataset: rollout must be an object")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise RobotContractError("trajectory dataset: rollout steps must be a list")
    return steps
