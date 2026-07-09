"""Stable RobotTask SDK contracts.

The v1 skeleton mirrors ``cybernetics.behavior_ci``: dataclasses, explicit
``from_dict`` / ``to_dict`` methods, and no heavyweight runtime dependencies.
Fields may be added compatibly. Existing serialized keys should not be renamed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

ROBOT_TASK_SCHEMA_VERSION = "robot-task/v1"
ROBOT_POLICY_SCHEMA_VERSION = "robot-policy/v1"
ROBOT_RUN_SCHEMA_VERSION = "robot-run/v1"
ROBOT_DATASET_SCHEMA_VERSION = "robot-trajectory-dataset/v1"
WORLD_MODEL_SCHEMA_VERSION = "robot-world-model/v1"

SIMULATOR_BACKENDS = ("fixture", "locomujoco", "mujoco", "isaaclab", "isaac_neko")
POLICY_KINDS = ("rl_policy", "vla_policy", "world_action_model")
POLICY_FORMATS = ("onnx", "torchscript", "rsl_rl", "sb3", "jax", "worldlines", "custom")
RUN_STATUSES = ("queued", "running", "succeeded", "failed", "truncated", "cancelled")
DATA_PROVENANCE = ("sim", "real", "synthetic", "mixed")
WORLD_MODEL_ROLES = ("world_model", "world_action_model", "reasoner", "policy")


class RobotContractError(ValueError):
    """A robot task/artifact contract is malformed or internally inconsistent."""


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise RobotContractError(f"{where}: missing required field '{key}'")
    return data[key]


def _check_schema(data: Mapping[str, Any], expected: str, where: str) -> None:
    got = data.get("schema_version")
    if got != expected:
        raise RobotContractError(f"{where}: schema_version must be '{expected}', got {got!r}")


def _as_dict(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotContractError(f"{where}: expected object, got {type(value).__name__}")
    return dict(value)


def _as_list(value: Any, where: str) -> List[Any]:
    if not isinstance(value, list):
        raise RobotContractError(f"{where}: expected list, got {type(value).__name__}")
    return list(value)


def _check_choice(value: str, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise RobotContractError(f"{where}: must be one of {list(allowed)}, got {value!r}")


def _positive_float(value: Any, where: str) -> float:
    result = float(value)
    if result <= 0:
        raise RobotContractError(f"{where}: must be positive, got {result!r}")
    return result


def _check_control_dt(sim_dt: float, control_dt: float) -> None:
    ratio = control_dt / sim_dt
    if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
        raise RobotContractError("robot task: control_dt must be a multiple of sim_dt")


@dataclass(frozen=True)
class RobotTaskSpec:
    schema_version: str
    task_id: str
    robot_id: str
    simulator_backend: str
    backend_config: Dict[str, Any]
    asset_refs: List[Dict[str, Any]]
    joint_map: Dict[str, str]
    actuator_model: Dict[str, Any]
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    sim_dt: float
    control_dt: float
    reset_spec: Dict[str, Any]
    reward_spec: Dict[str, Any]
    success_metric: Dict[str, Any]
    randomization: Dict[str, Any]
    termination: Dict[str, Any]
    eval_protocol: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotTaskSpec":
        _check_schema(data, ROBOT_TASK_SCHEMA_VERSION, "robot task")
        backend = str(_require(data, "simulator_backend", "robot task"))
        _check_choice(backend, SIMULATOR_BACKENDS, "robot task simulator_backend")
        sim_dt = _positive_float(_require(data, "sim_dt", "robot task"), "robot task sim_dt")
        control_dt = _positive_float(
            _require(data, "control_dt", "robot task"), "robot task control_dt"
        )
        _check_control_dt(sim_dt, control_dt)
        asset_refs = [
            _as_dict(item, "robot task asset_refs[]")
            for item in _as_list(_require(data, "asset_refs", "robot task"), "robot task asset_refs")
        ]
        return cls(
            schema_version=str(data["schema_version"]),
            task_id=str(_require(data, "task_id", "robot task")),
            robot_id=str(_require(data, "robot_id", "robot task")),
            simulator_backend=backend,
            backend_config=_as_dict(_require(data, "backend_config", "robot task"), "robot task backend_config"),
            asset_refs=asset_refs,
            joint_map={str(k): str(v) for k, v in _as_dict(_require(data, "joint_map", "robot task"), "robot task joint_map").items()},
            actuator_model=_as_dict(_require(data, "actuator_model", "robot task"), "robot task actuator_model"),
            observation_space=_as_dict(_require(data, "observation_space", "robot task"), "robot task observation_space"),
            action_space=_as_dict(_require(data, "action_space", "robot task"), "robot task action_space"),
            sim_dt=sim_dt,
            control_dt=control_dt,
            reset_spec=_as_dict(_require(data, "reset_spec", "robot task"), "robot task reset_spec"),
            reward_spec=_as_dict(_require(data, "reward_spec", "robot task"), "robot task reward_spec"),
            success_metric=_as_dict(_require(data, "success_metric", "robot task"), "robot task success_metric"),
            randomization=_as_dict(_require(data, "randomization", "robot task"), "robot task randomization"),
            termination=_as_dict(_require(data, "termination", "robot task"), "robot task termination"),
            eval_protocol=_as_dict(_require(data, "eval_protocol", "robot task"), "robot task eval_protocol"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def task_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class PolicyArtifact:
    schema_version: str
    artifact_id: str
    task_spec_uri: str
    task_spec_hash: str
    checkpoint_uri: Optional[str]
    policy_format: str
    observation_schema: Dict[str, Any]
    action_schema: Dict[str, Any]
    robot_id: str
    simulator_backend: str
    backend_version: str
    eval_metrics: Dict[str, Any]
    rollout_artifacts: List[str]
    created_by_run_id: str
    policy_kind: str = "rl_policy"
    inference_runtime: Optional[str] = None
    control_dt: Optional[float] = None
    latency_budget_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyArtifact":
        _check_schema(data, ROBOT_POLICY_SCHEMA_VERSION, "policy artifact")
        task_spec_hash = str(_require(data, "task_spec_hash", "policy artifact"))
        if not task_spec_hash:
            raise RobotContractError("policy artifact: task_spec_hash must be non-empty")
        policy_format = str(_require(data, "policy_format", "policy artifact"))
        _check_choice(policy_format, POLICY_FORMATS, "policy artifact policy_format")
        policy_kind = str(data.get("policy_kind", "rl_policy"))
        _check_choice(policy_kind, POLICY_KINDS, "policy artifact policy_kind")
        backend = str(_require(data, "simulator_backend", "policy artifact"))
        _check_choice(backend, SIMULATOR_BACKENDS, "policy artifact simulator_backend")
        control_dt = data.get("control_dt")
        return cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(_require(data, "artifact_id", "policy artifact")),
            task_spec_uri=str(_require(data, "task_spec_uri", "policy artifact")),
            task_spec_hash=task_spec_hash,
            checkpoint_uri=data.get("checkpoint_uri"),
            policy_format=policy_format,
            observation_schema=_as_dict(_require(data, "observation_schema", "policy artifact"), "policy artifact observation_schema"),
            action_schema=_as_dict(_require(data, "action_schema", "policy artifact"), "policy artifact action_schema"),
            robot_id=str(_require(data, "robot_id", "policy artifact")),
            simulator_backend=backend,
            backend_version=str(_require(data, "backend_version", "policy artifact")),
            eval_metrics=_as_dict(_require(data, "eval_metrics", "policy artifact"), "policy artifact eval_metrics"),
            rollout_artifacts=[str(item) for item in _as_list(_require(data, "rollout_artifacts", "policy artifact"), "policy artifact rollout_artifacts")],
            created_by_run_id=str(_require(data, "created_by_run_id", "policy artifact")),
            policy_kind=policy_kind,
            inference_runtime=data.get("inference_runtime"),
            control_dt=float(control_dt) if control_dt is not None else None,
            latency_budget_ms=int(data["latency_budget_ms"]) if data.get("latency_budget_ms") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotRunRecord:
    schema_version: str
    run_id: str
    task_spec_uri: str
    task_spec_hash: str
    backend_image: str
    seed: int
    status: str
    logs_uri: str
    metrics_uri: str
    artifacts_uri: str
    policy_artifact_uri: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotRunRecord":
        _check_schema(data, ROBOT_RUN_SCHEMA_VERSION, "robot run record")
        status = str(_require(data, "status", "robot run record"))
        _check_choice(status, RUN_STATUSES, "robot run status")
        task_spec_hash = str(_require(data, "task_spec_hash", "robot run record"))
        if not task_spec_hash:
            raise RobotContractError("robot run record: task_spec_hash must be non-empty")
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(_require(data, "run_id", "robot run record")),
            task_spec_uri=str(_require(data, "task_spec_uri", "robot run record")),
            task_spec_hash=task_spec_hash,
            backend_image=str(_require(data, "backend_image", "robot run record")),
            seed=int(_require(data, "seed", "robot run record")),
            status=status,
            logs_uri=str(_require(data, "logs_uri", "robot run record")),
            metrics_uri=str(_require(data, "metrics_uri", "robot run record")),
            artifacts_uri=str(_require(data, "artifacts_uri", "robot run record")),
            policy_artifact_uri=data.get("policy_artifact_uri"),
            error=data.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryDatasetArtifact:
    schema_version: str
    artifact_id: str
    task_spec_uri: str
    source_backend: str
    source_runs: List[str]
    observation_schema: Dict[str, Any]
    action_schema: Dict[str, Any]
    episode_count: int
    frame_count: int
    storage_uri: str
    data_provenance: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryDatasetArtifact":
        _check_schema(data, ROBOT_DATASET_SCHEMA_VERSION, "trajectory dataset")
        backend = str(_require(data, "source_backend", "trajectory dataset"))
        _check_choice(backend, SIMULATOR_BACKENDS, "trajectory dataset source_backend")
        provenance = str(_require(data, "data_provenance", "trajectory dataset"))
        _check_choice(provenance, DATA_PROVENANCE, "trajectory dataset data_provenance")
        return cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(_require(data, "artifact_id", "trajectory dataset")),
            task_spec_uri=str(_require(data, "task_spec_uri", "trajectory dataset")),
            source_backend=backend,
            source_runs=[str(item) for item in _as_list(_require(data, "source_runs", "trajectory dataset"), "trajectory dataset source_runs")],
            observation_schema=_as_dict(_require(data, "observation_schema", "trajectory dataset"), "trajectory dataset observation_schema"),
            action_schema=_as_dict(_require(data, "action_schema", "trajectory dataset"), "trajectory dataset action_schema"),
            episode_count=int(_require(data, "episode_count", "trajectory dataset")),
            frame_count=int(_require(data, "frame_count", "trajectory dataset")),
            storage_uri=str(_require(data, "storage_uri", "trajectory dataset")),
            data_provenance=provenance,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorldModelArtifact:
    schema_version: str
    artifact_id: str
    model_family: str
    model_uri: str
    model_role: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    horizon: float
    dt: float
    finetune_dataset_uri: str
    synthetic_data_policy: Dict[str, Any]
    calibration_metrics: Dict[str, Any]
    backend_version: str
    created_by_run_id: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelArtifact":
        _check_schema(data, WORLD_MODEL_SCHEMA_VERSION, "world model artifact")
        role = str(_require(data, "model_role", "world model artifact"))
        _check_choice(role, WORLD_MODEL_ROLES, "world model artifact model_role")
        return cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(_require(data, "artifact_id", "world model artifact")),
            model_family=str(_require(data, "model_family", "world model artifact")),
            model_uri=str(_require(data, "model_uri", "world model artifact")),
            model_role=role,
            input_schema=_as_dict(_require(data, "input_schema", "world model artifact"), "world model artifact input_schema"),
            output_schema=_as_dict(_require(data, "output_schema", "world model artifact"), "world model artifact output_schema"),
            horizon=_positive_float(_require(data, "horizon", "world model artifact"), "world model artifact horizon"),
            dt=_positive_float(_require(data, "dt", "world model artifact"), "world model artifact dt"),
            finetune_dataset_uri=str(_require(data, "finetune_dataset_uri", "world model artifact")),
            synthetic_data_policy=_as_dict(_require(data, "synthetic_data_policy", "world model artifact"), "world model artifact synthetic_data_policy"),
            calibration_metrics=_as_dict(_require(data, "calibration_metrics", "world model artifact"), "world model artifact calibration_metrics"),
            backend_version=str(_require(data, "backend_version", "world model artifact")),
            created_by_run_id=str(_require(data, "created_by_run_id", "world model artifact")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
