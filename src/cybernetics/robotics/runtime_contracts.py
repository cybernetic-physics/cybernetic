"""Provider-neutral contracts for managed robotics rollouts.

These types describe execution without importing or constructing simulator
packages. Heavy runtimes consume the serialized contracts in isolated jobs;
the SDK remains safe to import in clients, control planes, and CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Optional

from .contracts import RobotContractError

SIMULATOR_PACKAGE_SCHEMA_VERSION = "robot-simulator-package/v1"
TASK_PACKAGE_SCHEMA_VERSION = "robot-task-package/v1"
POLICY_DEPLOYMENT_SCHEMA_VERSION = "robot-policy-deployment/v1"
ASSET_BUNDLE_REF_SCHEMA_VERSION = "asset-bundle-ref/v1"
ARTIFACT_REF_SCHEMA_VERSION = "robot-artifact-ref/v1"
ROBOTICS_JOB_SCHEMA_VERSION = "robotics-job/v1"
EPISODE_MANIFEST_SCHEMA_VERSION = "robot-episode-manifest/v1"

FACTORY_KINDS = ("gymnasium", "lerobot_envhub", "python")
FACTORY_VECTORIZATION_MODES = ("sync", "native")
READINESS_KINDS = ("factory", "method")
POLICY_DEPLOYMENT_SOURCES = ("fixture", "worldlines", "local")
POLICY_STATE_MODELS = ("stateless", "recurrent", "history", "dual_system")
POLICY_RESET_GRANULARITIES = ("session", "batch", "environment")
PLACEMENT_TOPOLOGIES = ("colocated_required", "colocated_preferred", "separate")
ACTION_OVERLAP_MODES = ("latest", "fifo", "temporal_ensemble")
ACTION_REPRESENTATIONS = (
    "discrete",
    "waypoint",
    "base_velocity",
    "joint_position",
    "joint_velocity",
    "eef_delta",
    "eef_absolute",
    "gripper",
    "native",
)
EPISODE_STATUSES = ("succeeded", "failed", "truncated", "cancelled", "crashed")
DATASET_EXPORTS = ("none", "jsonl", "lerobot_v3")
CHECK_OPERATORS = ("==", "!=", "<", "<=", ">", ">=")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_ASSET_MOUNT_PATH_RE = re.compile(r"^/runtime/assets/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_ASSET_MOUNT_ROOT_RE = re.compile(r"^/runtime/assets(?:/[A-Za-z0-9._-]+)*$")


def canonical_runtime_json(value: Mapping[str, Any]) -> str:
    """Canonical JSON shared with the TypeScript robotics contract parser."""

    return json.dumps(
        _canonical_runtime_value(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def runtime_contract_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_runtime_json(value).encode("utf-8")).hexdigest()


def _canonical_runtime_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_runtime_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_runtime_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise RobotContractError(f"{where}: missing required field {key!r}")
    return data[key]


def _mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotContractError(f"{where}: expected object, got {type(value).__name__}")
    return dict(value)


def _strict_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extras = [str(key) for key in data if key not in allowed]
    if extras:
        raise RobotContractError(f"{where}: unknown fields {sorted(extras)}")


def _list(value: Any, where: str) -> List[Any]:
    if not isinstance(value, list):
        raise RobotContractError(f"{where}: expected list, got {type(value).__name__}")
    return list(value)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobotContractError(f"{where}: must be a non-empty string")
    return value


def _optional_string(value: Any, where: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, where)


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise RobotContractError(f"{where}: must be a boolean")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobotContractError(f"{where}: must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RobotContractError(f"{where}: must be a finite number")
    return result


def _integer(value: Any, where: str) -> int:
    result = _number(value, where)
    if not result.is_integer():
        raise RobotContractError(f"{where}: must be an integer")
    return int(result)


def _positive_float(value: Any, where: str) -> float:
    result = _number(value, where)
    if result <= 0:
        raise RobotContractError(f"{where}: must be positive")
    return result


def _positive_int(value: Any, where: str) -> int:
    result = _integer(value, where)
    if result <= 0:
        raise RobotContractError(f"{where}: must be a positive integer")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    result = _integer(value, where)
    if result < 0:
        raise RobotContractError(f"{where}: must be a non-negative integer")
    return result


def _nonnegative_float(value: Any, where: str) -> float:
    result = _number(value, where)
    if result < 0:
        raise RobotContractError(f"{where}: must be non-negative")
    return result


def _string_list(value: Any, where: str) -> List[str]:
    values = _list(value, where)
    return [_string(item, f"{where}[]") for item in values]


def _at_most(value: Any, maximum: Any, where: str) -> Any:
    if value > maximum:
        raise RobotContractError(f"{where}: must be at most {maximum}")
    return value


def _contract_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _contract_value(field_value)
            for item in fields(value)
            if (field_value := getattr(value, item.name)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_contract_value(item) for item in value]
    if isinstance(value, Mapping):
        # Mapping values are user payloads; preserve intentional null metadata.
        return {key: _contract_value(item) for key, item in value.items()}
    return value


def _contract_dict(value: Any) -> Dict[str, Any]:
    result = _contract_value(value)
    if not isinstance(result, dict):  # pragma: no cover - internal misuse guard
        raise TypeError("contract serialization requires a dataclass")
    return result


def _choice(value: Any, choices: tuple[str, ...], where: str) -> str:
    result = _string(value, where)
    if result not in choices:
        raise RobotContractError(f"{where}: must be one of {list(choices)}, got {result!r}")
    return result


def _schema(data: Mapping[str, Any], expected: str, where: str) -> None:
    actual = data.get("schema_version")
    if actual != expected:
        raise RobotContractError(f"{where}: schema_version must be {expected!r}, got {actual!r}")


def _sha256(value: Any, where: str) -> str:
    result = _string(value, where).lower()
    if not _SHA256_RE.fullmatch(result):
        raise RobotContractError(f"{where}: must be a lowercase SHA-256 digest")
    return result


@dataclass(frozen=True)
class TensorSpec:
    name: str
    semantic: str
    dtype: str
    shape: List[int]
    units: Optional[str] = None
    frame: Optional[str] = None
    bounds: Optional[List[float]] = None
    rate_hz: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSpec":
        _strict_keys(
            data,
            {"name", "semantic", "dtype", "shape", "units", "frame", "bounds", "rate_hz"},
            "tensor spec",
        )
        shape_raw = _list(_require(data, "shape", "tensor spec"), "tensor spec.shape")
        if not shape_raw:
            raise RobotContractError("tensor spec.shape must contain at least one dimension")
        shape = [_positive_int(item, "tensor spec.shape[]") for item in shape_raw]
        bounds_raw = data.get("bounds")
        bounds = (
            None
            if bounds_raw is None
            else [
                _number(v, "tensor spec.bounds[]") for v in _list(bounds_raw, "tensor spec.bounds")
            ]
        )
        if bounds is not None and len(bounds) != 2:
            raise RobotContractError("tensor spec.bounds must contain [minimum, maximum]")
        if bounds is not None and bounds[0] > bounds[1]:
            raise RobotContractError("tensor spec.bounds minimum cannot exceed maximum")
        return cls(
            name=_string(_require(data, "name", "tensor spec"), "tensor spec.name"),
            semantic=_string(_require(data, "semantic", "tensor spec"), "tensor spec.semantic"),
            dtype=_string(_require(data, "dtype", "tensor spec"), "tensor spec.dtype"),
            shape=shape,
            units=_optional_string(data.get("units"), "tensor spec.units"),
            frame=_optional_string(data.get("frame"), "tensor spec.frame"),
            bounds=bounds,
            rate_hz=_positive_float(data["rate_hz"], "tensor spec.rate_hz")
            if data.get("rate_hz") is not None
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class ActionSpec:
    representation: str
    tensor: TensorSpec
    control_hz: float
    horizon: int = 1
    normalization_id: Optional[str] = None
    joint_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionSpec":
        _strict_keys(
            data,
            {
                "representation",
                "tensor",
                "control_hz",
                "horizon",
                "normalization_id",
                "joint_names",
                "metadata",
            },
            "action spec",
        )
        return cls(
            representation=_choice(
                _require(data, "representation", "action spec"),
                ACTION_REPRESENTATIONS,
                "action spec.representation",
            ),
            tensor=TensorSpec.from_dict(
                _mapping(_require(data, "tensor", "action spec"), "action spec.tensor")
            ),
            control_hz=_positive_float(
                _require(data, "control_hz", "action spec"), "action spec.control_hz"
            ),
            horizon=_positive_int(data.get("horizon", 1), "action spec.horizon"),
            normalization_id=_optional_string(
                data.get("normalization_id"), "action spec.normalization_id"
            ),
            joint_names=_string_list(data.get("joint_names", []), "action spec.joint_names"),
            metadata=_mapping(data.get("metadata", {}), "action spec.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class ActionChunk:
    values: Any
    representation: str
    requested_horizon: int
    produced_horizon: int
    timestamps: List[float] = field(default_factory=list)
    valid_mask: Any = None
    inference_latency_ms: Optional[float] = None
    auxiliary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionChunk":
        _strict_keys(
            data,
            {
                "values",
                "representation",
                "requested_horizon",
                "produced_horizon",
                "timestamps",
                "valid_mask",
                "inference_latency_ms",
                "auxiliary",
            },
            "action chunk",
        )
        requested = _positive_int(
            _require(data, "requested_horizon", "action chunk"),
            "action chunk.requested_horizon",
        )
        produced = _positive_int(
            _require(data, "produced_horizon", "action chunk"),
            "action chunk.produced_horizon",
        )
        if produced > requested:
            raise RobotContractError(
                "action chunk.produced_horizon cannot exceed requested_horizon"
            )
        timestamps = [
            _number(v, "action chunk.timestamps[]")
            for v in _list(data.get("timestamps", []), "action chunk.timestamps")
        ]
        if timestamps and len(timestamps) != produced:
            raise RobotContractError(
                "action chunk.timestamps must be empty or match produced_horizon"
            )
        return cls(
            values=_require(data, "values", "action chunk"),
            representation=_choice(
                _require(data, "representation", "action chunk"),
                ACTION_REPRESENTATIONS,
                "action chunk.representation",
            ),
            requested_horizon=requested,
            produced_horizon=produced,
            timestamps=timestamps,
            valid_mask=data.get("valid_mask"),
            inference_latency_ms=_nonnegative_float(
                data["inference_latency_ms"], "action chunk.inference_latency_ms"
            )
            if data.get("inference_latency_ms") is not None
            else None,
            auxiliary=_mapping(data.get("auxiliary", {}), "action chunk.auxiliary"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    media_type: str
    role: str
    size_bytes: Optional[int] = None
    episode_id: Optional[str] = None
    step_range: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACT_REF_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        _strict_keys(
            data,
            {
                "schema_version",
                "uri",
                "sha256",
                "media_type",
                "role",
                "size_bytes",
                "episode_id",
                "step_range",
                "metadata",
            },
            "artifact ref",
        )
        _schema(data, ARTIFACT_REF_SCHEMA_VERSION, "artifact ref")
        step_range_raw = data.get("step_range")
        step_range = (
            None
            if step_range_raw is None
            else [
                _nonnegative_int(v, "artifact ref.step_range[]")
                for v in _list(step_range_raw, "artifact ref.step_range")
            ]
        )
        if step_range is not None and (len(step_range) != 2 or step_range[0] > step_range[1]):
            raise RobotContractError("artifact ref.step_range must be [start, end]")
        return cls(
            uri=_string(_require(data, "uri", "artifact ref"), "artifact ref.uri"),
            sha256=_sha256(_require(data, "sha256", "artifact ref"), "artifact ref.sha256"),
            media_type=_string(
                _require(data, "media_type", "artifact ref"), "artifact ref.media_type"
            ),
            role=_string(_require(data, "role", "artifact ref"), "artifact ref.role"),
            size_bytes=_positive_int(data["size_bytes"], "artifact ref.size_bytes")
            if data.get("size_bytes") is not None
            else None,
            episode_id=_optional_string(data.get("episode_id"), "artifact ref.episode_id"),
            step_range=step_range,
            metadata=_mapping(data.get("metadata", {}), "artifact ref.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class AssetBundleRef:
    uri: str
    content_sha256: str
    media_type: str
    size_bytes: int
    source: Dict[str, Any] = field(default_factory=dict)
    license: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = ASSET_BUNDLE_REF_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetBundleRef":
        _strict_keys(
            data,
            {
                "schema_version",
                "uri",
                "content_sha256",
                "media_type",
                "size_bytes",
                "source",
                "license",
                "metadata",
            },
            "asset bundle ref",
        )
        _schema(data, ASSET_BUNDLE_REF_SCHEMA_VERSION, "asset bundle ref")
        return cls(
            uri=_string(_require(data, "uri", "asset bundle ref"), "asset bundle ref.uri"),
            content_sha256=_sha256(
                _require(data, "content_sha256", "asset bundle ref"),
                "asset bundle ref.content_sha256",
            ),
            media_type=_string(
                _require(data, "media_type", "asset bundle ref"),
                "asset bundle ref.media_type",
            ),
            size_bytes=_positive_int(
                _require(data, "size_bytes", "asset bundle ref"),
                "asset bundle ref.size_bytes",
            ),
            source=_mapping(data.get("source", {}), "asset bundle ref.source"),
            license=_optional_string(data.get("license"), "asset bundle ref.license"),
            metadata=_mapping(data.get("metadata", {}), "asset bundle ref.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class AssetMountSpec:
    mount_path: str
    ref: Dict[str, Any]
    read_only: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetMountSpec":
        _strict_keys(data, {"mount_path", "ref", "read_only"}, "asset mount")
        mount_path = _string(_require(data, "mount_path", "asset mount"), "asset mount.mount_path")
        if not _ASSET_MOUNT_PATH_RE.fullmatch(mount_path):
            raise RobotContractError(
                "asset mount.mount_path must be a canonical child of /runtime/assets"
            )
        ref = _mapping(_require(data, "ref", "asset mount"), "asset mount.ref")
        schema = _string(
            _require(ref, "schema_version", "asset mount.ref"), "asset mount.ref.schema_version"
        )
        if schema == ASSET_BUNDLE_REF_SCHEMA_VERSION:
            ref = AssetBundleRef.from_dict(ref).to_dict()
        elif schema == "simulation-asset-ref/v1":
            _string(_require(ref, "uri", "asset mount.ref"), "asset mount.ref.uri")
            _string(_require(ref, "env_id", "asset mount.ref"), "asset mount.ref.env_id")
            _string(
                _require(ref, "version_id", "asset mount.ref"),
                "asset mount.ref.version_id",
            )
            _sha256(
                _require(ref, "content_sha256", "asset mount.ref"),
                "asset mount.ref.content_sha256",
            )
        else:
            raise RobotContractError(f"asset mount.ref has unsupported schema {schema!r}")
        read_only = _boolean(data.get("read_only", True), "asset mount.read_only")
        if not read_only:
            raise RobotContractError("asset mount.read_only must be true")
        return cls(mount_path=mount_path, ref=ref, read_only=True)

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


def _validate_asset_mount_paths(mounts: List[AssetMountSpec]) -> None:
    paths = sorted(mount.mount_path for mount in mounts)
    for index, path in enumerate(paths):
        for parent in paths[:index]:
            if path == parent or path.startswith(f"{parent}/"):
                raise RobotContractError(
                    "environment package.asset_mounts cannot use duplicate or overlapping paths"
                )


@dataclass(frozen=True)
class EnvironmentFactorySpec:
    kind: str
    target: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    vectorization: str = "native"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentFactorySpec":
        _strict_keys(
            data,
            {"kind", "target", "kwargs", "vectorization"},
            "environment factory",
        )
        kind = _choice(
            _require(data, "kind", "environment factory"), FACTORY_KINDS, "environment factory.kind"
        )
        target = _string(
            _require(data, "target", "environment factory"), "environment factory.target"
        )
        if kind in {"python", "lerobot_envhub"} and ":" not in target:
            raise RobotContractError(f"{kind} environment factory.target must be 'module:callable'")
        vectorization = _choice(
            data.get("vectorization", "sync" if kind == "gymnasium" else "native"),
            FACTORY_VECTORIZATION_MODES,
            "environment factory.vectorization",
        )
        if kind != "gymnasium" and vectorization != "native":
            raise RobotContractError(
                "non-Gymnasium environment factories require vectorization='native'"
            )
        return cls(
            kind=kind,
            target=target,
            kwargs=_mapping(data.get("kwargs", {}), "environment factory.kwargs"),
            vectorization=vectorization,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class EnvironmentReadinessSpec:
    """Readiness gate applied after factory construction and before policy load."""

    kind: str = "factory"
    target: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentReadinessSpec":
        _strict_keys(data, {"kind", "target"}, "readiness")
        kind = _choice(data.get("kind", "factory"), READINESS_KINDS, "readiness.kind")
        target = _optional_string(data.get("target"), "readiness.target")
        if kind == "method" and target is None:
            raise RobotContractError("readiness.target is required when kind='method'")
        if kind == "factory" and target is not None:
            raise RobotContractError("readiness.target is only valid when kind='method'")
        return cls(kind=kind, target=target)

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class RuntimeResources:
    cpu_cores: float
    memory_gb: float
    disk_gb: float
    gpu_count: int
    timeout_seconds: int
    gpu_type: Optional[str] = None
    shm_size_gb: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeResources":
        _strict_keys(
            data,
            {
                "cpu_cores",
                "memory_gb",
                "disk_gb",
                "gpu_count",
                "timeout_seconds",
                "gpu_type",
                "shm_size_gb",
            },
            "runtime resources",
        )
        cpu_cores = _at_most(
            _positive_float(
                _require(data, "cpu_cores", "runtime resources"),
                "runtime resources.cpu_cores",
            ),
            256,
            "runtime resources.cpu_cores",
        )
        memory_gb = _at_most(
            _positive_float(
                _require(data, "memory_gb", "runtime resources"),
                "runtime resources.memory_gb",
            ),
            2048,
            "runtime resources.memory_gb",
        )
        disk_gb = _at_most(
            _positive_float(
                _require(data, "disk_gb", "runtime resources"),
                "runtime resources.disk_gb",
            ),
            8192,
            "runtime resources.disk_gb",
        )
        gpu_count = _at_most(
            _nonnegative_int(
                _require(data, "gpu_count", "runtime resources"),
                "runtime resources.gpu_count",
            ),
            16,
            "runtime resources.gpu_count",
        )
        timeout_seconds = _at_most(
            _positive_int(
                _require(data, "timeout_seconds", "runtime resources"),
                "runtime resources.timeout_seconds",
            ),
            7 * 24 * 60 * 60,
            "runtime resources.timeout_seconds",
        )
        shm_size_gb = (
            _at_most(
                _positive_float(data["shm_size_gb"], "runtime resources.shm_size_gb"),
                1024,
                "runtime resources.shm_size_gb",
            )
            if data.get("shm_size_gb") is not None
            else None
        )
        return cls(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            gpu_count=gpu_count,
            timeout_seconds=timeout_seconds,
            gpu_type=_optional_string(data.get("gpu_type"), "runtime resources.gpu_type"),
            shm_size_gb=shm_size_gb,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


def _validate_resources_cover(
    requested: RuntimeResources,
    required: RuntimeResources,
    where: str,
) -> None:
    for name in ("cpu_cores", "memory_gb", "disk_gb", "gpu_count", "timeout_seconds"):
        if getattr(requested, name) < getattr(required, name):
            raise RobotContractError(f"{where}.{name} is below the package requirement")
    if (
        required.shm_size_gb is not None
        and (requested.shm_size_gb or 0) < required.shm_size_gb
    ):
        raise RobotContractError(f"{where}.shm_size_gb is below the package requirement")
    if (
        required.gpu_count > 0
        and required.gpu_type is not None
        and requested.gpu_type != required.gpu_type
    ):
        raise RobotContractError(f"{where}.gpu_type does not satisfy the package requirement")


@dataclass(frozen=True)
class SimulatorPackageSpec:
    schema_version: str
    package_id: str
    simulator: str
    simulator_version: str
    source_repo: str
    source_ref: str
    runtime_image: str
    service_entrypoint: str
    factory: EnvironmentFactorySpec
    resources: RuntimeResources
    supports_vectorization: bool
    default_vector_width: int
    capabilities: List[str]
    supported_asset_formats: List[str]
    mount_roots: List[str]
    readiness: EnvironmentReadinessSpec = field(default_factory=EnvironmentReadinessSpec)
    license: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulatorPackageSpec":
        _strict_keys(
            data,
            {
                "schema_version",
                "package_id",
                "simulator",
                "simulator_version",
                "source_repo",
                "source_ref",
                "runtime_image",
                "service_entrypoint",
                "factory",
                "resources",
                "supports_vectorization",
                "default_vector_width",
                "capabilities",
                "supported_asset_formats",
                "mount_roots",
                "readiness",
                "license",
                "metadata",
            },
            "simulator package",
        )
        _schema(data, SIMULATOR_PACKAGE_SCHEMA_VERSION, "simulator package")
        runtime_image = _string(
            _require(data, "runtime_image", "simulator package"),
            "simulator package.runtime_image",
        )
        if not _IMAGE_DIGEST_RE.fullmatch(runtime_image):
            raise RobotContractError(
                "simulator package.runtime_image must be pinned by an OCI sha256 digest"
            )
        supports_vectorization = _boolean(
            _require(data, "supports_vectorization", "simulator package"),
            "simulator package.supports_vectorization",
        )
        width = _at_most(
            _positive_int(
                _require(data, "default_vector_width", "simulator package"),
                "simulator package.default_vector_width",
            ),
            4096,
            "simulator package.default_vector_width",
        )
        if width != 1 and not supports_vectorization:
            raise RobotContractError(
                "simulator package.default_vector_width must be 1 when vectorization is unsupported"
            )
        capabilities = _string_list(
            _require(data, "capabilities", "simulator package"),
            "simulator package.capabilities",
        )
        if not capabilities:
            raise RobotContractError("simulator package.capabilities must not be empty")
        mount_roots = _string_list(
            _require(data, "mount_roots", "simulator package"),
            "simulator package.mount_roots",
        )
        if not mount_roots or any(not _ASSET_MOUNT_ROOT_RE.fullmatch(root) for root in mount_roots):
            raise RobotContractError(
                "simulator package.mount_roots must contain canonical /runtime/assets children"
            )
        return cls(
            schema_version=SIMULATOR_PACKAGE_SCHEMA_VERSION,
            package_id=_string(
                _require(data, "package_id", "simulator package"),
                "simulator package.package_id",
            ),
            simulator=_string(
                _require(data, "simulator", "simulator package"),
                "simulator package.simulator",
            ),
            simulator_version=_string(
                _require(data, "simulator_version", "simulator package"),
                "simulator package.simulator_version",
            ),
            source_repo=_string(
                _require(data, "source_repo", "simulator package"),
                "simulator package.source_repo",
            ),
            source_ref=_string(
                _require(data, "source_ref", "simulator package"),
                "simulator package.source_ref",
            ),
            runtime_image=runtime_image,
            service_entrypoint=_string(
                _require(data, "service_entrypoint", "simulator package"),
                "simulator package.service_entrypoint",
            ),
            factory=EnvironmentFactorySpec.from_dict(
                _mapping(
                    _require(data, "factory", "simulator package"),
                    "simulator package.factory",
                )
            ),
            resources=RuntimeResources.from_dict(
                _mapping(
                    _require(data, "resources", "simulator package"),
                    "simulator package.resources",
                )
            ),
            supports_vectorization=supports_vectorization,
            default_vector_width=width,
            capabilities=capabilities,
            supported_asset_formats=_string_list(
                data.get("supported_asset_formats", []),
                "simulator package.supported_asset_formats",
            ),
            mount_roots=mount_roots,
            readiness=EnvironmentReadinessSpec.from_dict(
                _mapping(data.get("readiness", {}), "simulator package.readiness")
            ),
            license=_optional_string(data.get("license"), "simulator package.license"),
            metadata=_mapping(data.get("metadata", {}), "simulator package.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)

    def package_hash(self) -> str:
        return runtime_contract_hash(self.to_dict())


@dataclass(frozen=True)
class TaskPackageSpec:
    schema_version: str
    package_id: str
    task_id: str
    revision: str
    source_repo: str
    source_ref: str
    compatible_simulators: List[str]
    required_capabilities: List[str]
    embodiment_id: str
    observation_schema: Dict[str, TensorSpec]
    action_spec: ActionSpec
    asset_mounts: List[AssetMountSpec]
    adapter_config: Dict[str, Any]
    dataset: Dict[str, Any]
    native_metrics: List[str]
    license: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskPackageSpec":
        _strict_keys(
            data,
            {
                "schema_version",
                "package_id",
                "task_id",
                "revision",
                "source_repo",
                "source_ref",
                "compatible_simulators",
                "required_capabilities",
                "embodiment_id",
                "observation_schema",
                "action_spec",
                "asset_mounts",
                "adapter_config",
                "dataset",
                "native_metrics",
                "license",
                "metadata",
            },
            "task package",
        )
        _schema(data, TASK_PACKAGE_SCHEMA_VERSION, "task package")
        compatible = _string_list(
            _require(data, "compatible_simulators", "task package"),
            "task package.compatible_simulators",
        )
        if not compatible:
            raise RobotContractError("task package.compatible_simulators must not be empty")
        observation_raw = _mapping(
            _require(data, "observation_schema", "task package"),
            "task package.observation_schema",
        )
        if not observation_raw:
            raise RobotContractError("task package.observation_schema must not be empty")
        asset_mounts_raw = _list(data.get("asset_mounts", []), "task package.asset_mounts")
        if len(asset_mounts_raw) > 128:
            raise RobotContractError("task package.asset_mounts must contain at most 128 mounts")
        asset_mounts = [
            AssetMountSpec.from_dict(_mapping(item, "task package.asset_mounts[]"))
            for item in asset_mounts_raw
        ]
        _validate_asset_mount_paths(asset_mounts)
        native_metrics = _string_list(
            _require(data, "native_metrics", "task package"),
            "task package.native_metrics",
        )
        if not native_metrics:
            raise RobotContractError("task package.native_metrics must not be empty")
        return cls(
            schema_version=TASK_PACKAGE_SCHEMA_VERSION,
            package_id=_string(
                _require(data, "package_id", "task package"), "task package.package_id"
            ),
            task_id=_string(_require(data, "task_id", "task package"), "task package.task_id"),
            revision=_string(
                _require(data, "revision", "task package"), "task package.revision"
            ),
            source_repo=_string(
                _require(data, "source_repo", "task package"), "task package.source_repo"
            ),
            source_ref=_string(
                _require(data, "source_ref", "task package"), "task package.source_ref"
            ),
            compatible_simulators=compatible,
            required_capabilities=_string_list(
                data.get("required_capabilities", []),
                "task package.required_capabilities",
            ),
            embodiment_id=_string(
                _require(data, "embodiment_id", "task package"),
                "task package.embodiment_id",
            ),
            observation_schema={
                str(name): TensorSpec.from_dict(_mapping(spec, f"observation_schema.{name}"))
                for name, spec in observation_raw.items()
            },
            action_spec=ActionSpec.from_dict(
                _mapping(
                    _require(data, "action_spec", "task package"),
                    "task package.action_spec",
                )
            ),
            asset_mounts=asset_mounts,
            adapter_config=_mapping(
                data.get("adapter_config", {}), "task package.adapter_config"
            ),
            dataset=_mapping(data.get("dataset", {}), "task package.dataset"),
            native_metrics=native_metrics,
            license=_string(
                _require(data, "license", "task package"), "task package.license"
            ),
            metadata=_mapping(data.get("metadata", {}), "task package.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)

    def package_hash(self) -> str:
        return runtime_contract_hash(self.to_dict())


@dataclass(frozen=True)
class ActionSelectionSpec:
    execution_horizon: int = 1
    queue_threshold: int = 0
    overlap: str = "latest"
    temporal_ensemble_weight: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionSelectionSpec":
        _strict_keys(
            data,
            {"execution_horizon", "queue_threshold", "overlap", "temporal_ensemble_weight"},
            "action selection",
        )
        overlap = _choice(data.get("overlap", "latest"), ACTION_OVERLAP_MODES, "action selection.overlap")
        weight = (
            _number(data["temporal_ensemble_weight"], "action selection.temporal_ensemble_weight")
            if data.get("temporal_ensemble_weight") is not None
            else None
        )
        if weight is not None and not 0 < weight <= 1:
            raise RobotContractError(
                "action selection.temporal_ensemble_weight must be in (0, 1]"
            )
        if overlap != "temporal_ensemble" and weight is not None:
            raise RobotContractError(
                "action selection.temporal_ensemble_weight requires temporal_ensemble overlap"
            )
        return cls(
            execution_horizon=_at_most(
                _positive_int(
                    data.get("execution_horizon", 1), "action selection.execution_horizon"
                ),
                4096,
                "action selection.execution_horizon",
            ),
            queue_threshold=_at_most(
                _nonnegative_int(
                    data.get("queue_threshold", 0), "action selection.queue_threshold"
                ),
                4096,
                "action selection.queue_threshold",
            ),
            overlap=overlap,
            temporal_ensemble_weight=weight,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class PolicyDeploymentSpec:
    schema_version: str
    deployment_id: str
    model_id: str
    source: str
    runtime_family: str
    revision: str
    embodiment_id: str
    observation_schema: Dict[str, TensorSpec]
    action_spec: ActionSpec
    resources: RuntimeResources
    max_batch_size: int
    max_horizon: int
    state_model: str
    reset_granularity: str
    deterministic: bool
    default_action_selection: ActionSelectionSpec
    checkpoint_ref: Optional[str] = None
    processor_revision: Optional[str] = None
    normalization_revision: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyDeploymentSpec":
        _strict_keys(
            data,
            {
                "schema_version",
                "deployment_id",
                "model_id",
                "source",
                "runtime_family",
                "revision",
                "embodiment_id",
                "observation_schema",
                "action_spec",
                "resources",
                "max_batch_size",
                "max_horizon",
                "state_model",
                "reset_granularity",
                "deterministic",
                "default_action_selection",
                "checkpoint_ref",
                "processor_revision",
                "normalization_revision",
                "config",
                "metadata",
            },
            "policy deployment",
        )
        _schema(data, POLICY_DEPLOYMENT_SCHEMA_VERSION, "policy deployment")
        source = _choice(
            _require(data, "source", "policy deployment"),
            POLICY_DEPLOYMENT_SOURCES,
            "policy deployment.source",
        )
        config = _mapping(data.get("config", {}), "policy deployment.config")
        if source == "worldlines" and {"url", "token", "api_key", "factory"} & config.keys():
            raise RobotContractError(
                "worldlines policy deployment endpoints and credentials are control-plane resolved"
            )
        observation_raw = _mapping(
            _require(data, "observation_schema", "policy deployment"),
            "policy deployment.observation_schema",
        )
        if not observation_raw:
            raise RobotContractError("policy deployment.observation_schema must not be empty")
        max_horizon = _at_most(
            _positive_int(
                _require(data, "max_horizon", "policy deployment"),
                "policy deployment.max_horizon",
            ),
            4096,
            "policy deployment.max_horizon",
        )
        default_action_selection = ActionSelectionSpec.from_dict(
            _mapping(
                data.get("default_action_selection", {}),
                "policy deployment.default_action_selection",
            )
        )
        if default_action_selection.execution_horizon > max_horizon:
            raise RobotContractError(
                "policy deployment default execution_horizon exceeds max_horizon"
            )
        return cls(
            schema_version=POLICY_DEPLOYMENT_SCHEMA_VERSION,
            deployment_id=_string(
                _require(data, "deployment_id", "policy deployment"),
                "policy deployment.deployment_id",
            ),
            model_id=_string(
                _require(data, "model_id", "policy deployment"),
                "policy deployment.model_id",
            ),
            source=source,
            runtime_family=_string(
                _require(data, "runtime_family", "policy deployment"),
                "policy deployment.runtime_family",
            ),
            revision=_string(
                _require(data, "revision", "policy deployment"),
                "policy deployment.revision",
            ),
            embodiment_id=_string(
                _require(data, "embodiment_id", "policy deployment"),
                "policy deployment.embodiment_id",
            ),
            observation_schema={
                str(name): TensorSpec.from_dict(_mapping(spec, f"observation_schema.{name}"))
                for name, spec in observation_raw.items()
            },
            action_spec=ActionSpec.from_dict(
                _mapping(
                    _require(data, "action_spec", "policy deployment"),
                    "policy deployment.action_spec",
                )
            ),
            resources=RuntimeResources.from_dict(
                _mapping(
                    _require(data, "resources", "policy deployment"),
                    "policy deployment.resources",
                )
            ),
            max_batch_size=_at_most(
                _positive_int(
                    _require(data, "max_batch_size", "policy deployment"),
                    "policy deployment.max_batch_size",
                ),
                4096,
                "policy deployment.max_batch_size",
            ),
            max_horizon=max_horizon,
            state_model=_choice(
                _require(data, "state_model", "policy deployment"),
                POLICY_STATE_MODELS,
                "policy deployment.state_model",
            ),
            reset_granularity=_choice(
                _require(data, "reset_granularity", "policy deployment"),
                POLICY_RESET_GRANULARITIES,
                "policy deployment.reset_granularity",
            ),
            deterministic=_boolean(
                _require(data, "deterministic", "policy deployment"),
                "policy deployment.deterministic",
            ),
            default_action_selection=default_action_selection,
            checkpoint_ref=_optional_string(
                data.get("checkpoint_ref"), "policy deployment.checkpoint_ref"
            ),
            processor_revision=_optional_string(
                data.get("processor_revision"), "policy deployment.processor_revision"
            ),
            normalization_revision=_optional_string(
                data.get("normalization_revision"),
                "policy deployment.normalization_revision",
            ),
            config=config,
            metadata=_mapping(data.get("metadata", {}), "policy deployment.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)

    def deployment_hash(self) -> str:
        return runtime_contract_hash(self.to_dict())


@dataclass(frozen=True)
class RolloutSpec:
    episodes: int
    root_seed: int
    vector_width: int
    max_steps: int
    control_rate_hz: float
    action_selection: ActionSelectionSpec
    seeds: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RolloutSpec":
        _strict_keys(
            data,
            {
                "episodes",
                "root_seed",
                "vector_width",
                "max_steps",
                "control_rate_hz",
                "action_selection",
                "seeds",
            },
            "rollout",
        )
        episodes = _at_most(
            _positive_int(_require(data, "episodes", "rollout"), "rollout.episodes"),
            100_000,
            "rollout.episodes",
        )
        seeds = [
            _integer(v, "rollout.seeds[]")
            for v in _list(data.get("seeds", []), "rollout.seeds")
        ]
        if seeds and len(seeds) != episodes:
            raise RobotContractError(
                "rollout.seeds must be empty or contain exactly rollout.episodes seeds"
            )
        return cls(
            episodes=episodes,
            root_seed=_integer(
                _require(data, "root_seed", "rollout"), "rollout.root_seed"
            ),
            vector_width=_at_most(
                _positive_int(
                    _require(data, "vector_width", "rollout"), "rollout.vector_width"
                ),
                4096,
                "rollout.vector_width",
            ),
            max_steps=_at_most(
                _positive_int(
                    _require(data, "max_steps", "rollout"), "rollout.max_steps"
                ),
                10_000_000,
                "rollout.max_steps",
            ),
            control_rate_hz=_positive_float(
                _require(data, "control_rate_hz", "rollout"),
                "rollout.control_rate_hz",
            ),
            action_selection=ActionSelectionSpec.from_dict(
                _mapping(data.get("action_selection", {}), "rollout.action_selection")
            ),
            seeds=seeds,
        )

    def resolved_seeds(self) -> List[int]:
        return (
            list(self.seeds)
            if self.seeds
            else [self.root_seed + index for index in range(self.episodes)]
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class PlacementSpec:
    topology: str
    simulator_resources: RuntimeResources
    policy_resources: RuntimeResources
    coordinator_resources: RuntimeResources
    gpu_sharing: bool = False
    provider: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlacementSpec":
        _strict_keys(
            data,
            {
                "topology",
                "simulator_resources",
                "policy_resources",
                "coordinator_resources",
                "gpu_sharing",
                "provider",
            },
            "placement",
        )
        return cls(
            topology=_choice(
                _require(data, "topology", "placement"),
                PLACEMENT_TOPOLOGIES,
                "placement.topology",
            ),
            simulator_resources=RuntimeResources.from_dict(
                _mapping(
                    _require(data, "simulator_resources", "placement"),
                    "placement.simulator_resources",
                )
            ),
            policy_resources=RuntimeResources.from_dict(
                _mapping(
                    _require(data, "policy_resources", "placement"),
                    "placement.policy_resources",
                )
            ),
            coordinator_resources=RuntimeResources.from_dict(
                _mapping(
                    _require(data, "coordinator_resources", "placement"),
                    "placement.coordinator_resources",
                )
            ),
            gpu_sharing=_boolean(data.get("gpu_sharing", False), "placement.gpu_sharing"),
            provider=_optional_string(data.get("provider"), "placement.provider"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class RecordingSpec:
    video: bool = True
    cameras: List[str] = field(default_factory=list)
    observations: bool = True
    actions: bool = True
    predictions: bool = False
    failure_clips: bool = True
    dataset_export: str = "jsonl"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordingSpec":
        _strict_keys(
            data,
            {
                "video",
                "cameras",
                "observations",
                "actions",
                "predictions",
                "failure_clips",
                "dataset_export",
            },
            "recording",
        )
        return cls(
            video=_boolean(data.get("video", True), "recording.video"),
            cameras=_string_list(data.get("cameras", []), "recording.cameras"),
            observations=_boolean(data.get("observations", True), "recording.observations"),
            actions=_boolean(data.get("actions", True), "recording.actions"),
            predictions=_boolean(data.get("predictions", False), "recording.predictions"),
            failure_clips=_boolean(data.get("failure_clips", True), "recording.failure_clips"),
            dataset_export=_choice(
                data.get("dataset_export", "jsonl"), DATASET_EXPORTS, "recording.dataset_export"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class EvaluationCheck:
    metric: str
    operator: str
    value: Any
    required: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationCheck":
        _strict_keys(data, {"metric", "operator", "value", "required"}, "evaluation check")
        return cls(
            metric=_string(_require(data, "metric", "evaluation check"), "evaluation check.metric"),
            operator=_choice(
                _require(data, "operator", "evaluation check"),
                CHECK_OPERATORS,
                "evaluation check.operator",
            ),
            value=_require(data, "value", "evaluation check"),
            required=_boolean(data.get("required", True), "evaluation check.required"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class EvaluationSpec:
    behavior: str
    primary_metric: str
    checks: Dict[str, EvaluationCheck]
    native_metrics: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationSpec":
        _strict_keys(
            data,
            {"behavior", "primary_metric", "checks", "native_metrics"},
            "evaluation",
        )
        checks_raw = _mapping(_require(data, "checks", "evaluation"), "evaluation.checks")
        if not checks_raw:
            raise RobotContractError("evaluation.checks must contain at least one check")
        return cls(
            behavior=_string(_require(data, "behavior", "evaluation"), "evaluation.behavior"),
            primary_metric=_string(
                _require(data, "primary_metric", "evaluation"), "evaluation.primary_metric"
            ),
            checks={
                _string(name, "evaluation.checks key"): EvaluationCheck.from_dict(
                    _mapping(value, f"evaluation.checks.{name}")
                )
                for name, value in checks_raw.items()
            },
            native_metrics=_string_list(
                data.get("native_metrics", []), "evaluation.native_metrics"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)


@dataclass(frozen=True)
class RoboticsJobSpec:
    schema_version: str
    job_name: str
    simulator: SimulatorPackageSpec
    task: TaskPackageSpec
    policy: PolicyDeploymentSpec
    rollout: RolloutSpec
    placement: PlacementSpec
    recording: RecordingSpec
    evaluation: EvaluationSpec
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboticsJobSpec":
        _strict_keys(
            data,
            {
                "schema_version",
                "job_name",
                "simulator",
                "task",
                "policy",
                "rollout",
                "placement",
                "recording",
                "evaluation",
                "metadata",
            },
            "robotics job",
        )
        _schema(data, ROBOTICS_JOB_SCHEMA_VERSION, "robotics job")
        simulator = SimulatorPackageSpec.from_dict(
            _mapping(_require(data, "simulator", "robotics job"), "robotics job.simulator")
        )
        task = TaskPackageSpec.from_dict(
            _mapping(_require(data, "task", "robotics job"), "robotics job.task")
        )
        policy = PolicyDeploymentSpec.from_dict(
            _mapping(_require(data, "policy", "robotics job"), "robotics job.policy")
        )
        rollout = RolloutSpec.from_dict(
            _mapping(_require(data, "rollout", "robotics job"), "robotics job.rollout")
        )
        placement = PlacementSpec.from_dict(
            _mapping(_require(data, "placement", "robotics job"), "robotics job.placement")
        )
        evaluation = EvaluationSpec.from_dict(
            _mapping(_require(data, "evaluation", "robotics job"), "robotics job.evaluation")
        )
        if rollout.vector_width > 1 and not simulator.supports_vectorization:
            raise RobotContractError(
                "robotics job requests vector_width > 1 for a non-vector simulator"
            )
        if simulator.package_id not in task.compatible_simulators and simulator.simulator not in task.compatible_simulators:
            raise RobotContractError("robotics job task is incompatible with the selected simulator")
        missing_capabilities = sorted(set(task.required_capabilities) - set(simulator.capabilities))
        if missing_capabilities:
            raise RobotContractError(
                f"robotics job simulator is missing task capabilities {missing_capabilities}"
            )
        for mount in task.asset_mounts:
            if not any(
                mount.mount_path == root or mount.mount_path.startswith(f"{root}/")
                for root in simulator.mount_roots
            ):
                raise RobotContractError(
                    f"task asset mount {mount.mount_path!r} is outside simulator mount roots"
                )
        if task.embodiment_id != policy.embodiment_id:
            raise RobotContractError("robotics job task and policy embodiment_id must match")
        if task.action_spec.to_dict() != policy.action_spec.to_dict():
            raise RobotContractError("robotics job task and policy action specs must match")
        for name, expected in policy.observation_schema.items():
            actual = task.observation_schema.get(name)
            if actual is None or actual.to_dict() != expected.to_dict():
                raise RobotContractError(
                    f"robotics job task observation {name!r} does not match policy deployment"
                )
        if rollout.vector_width > policy.max_batch_size:
            raise RobotContractError(
                "robotics job rollout.vector_width exceeds policy max_batch_size"
            )
        if rollout.action_selection.execution_horizon > policy.max_horizon:
            raise RobotContractError(
                "robotics job execution_horizon exceeds policy max_horizon"
            )
        if rollout.action_selection.execution_horizon > policy.action_spec.horizon:
            raise RobotContractError(
                "robotics job execution_horizon exceeds the policy action-spec horizon"
            )
        if not math.isclose(
            rollout.control_rate_hz,
            task.action_spec.control_hz,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise RobotContractError(
                "robotics job rollout.control_rate_hz must match the task action control_hz"
            )
        required_metrics = {check.metric for check in evaluation.checks.values()}
        missing_metrics = sorted(required_metrics - set(task.native_metrics))
        if missing_metrics:
            raise RobotContractError(
                f"robotics job evaluation references undeclared task metrics {missing_metrics}"
            )
        _validate_resources_cover(
            placement.simulator_resources,
            simulator.resources,
            "placement.simulator_resources",
        )
        _validate_resources_cover(
            placement.policy_resources,
            policy.resources,
            "placement.policy_resources",
        )
        if placement.gpu_sharing:
            gpu_types = {
                item.gpu_type
                for item in (placement.simulator_resources, placement.policy_resources)
                if item.gpu_count > 0 and item.gpu_type is not None
            }
            if len(gpu_types) > 1:
                raise RobotContractError(
                    "robotics job shared simulator/policy GPUs require compatible gpu_type values"
                )
        return cls(
            schema_version=ROBOTICS_JOB_SCHEMA_VERSION,
            job_name=_string(_require(data, "job_name", "robotics job"), "robotics job.job_name"),
            simulator=simulator,
            task=task,
            policy=policy,
            rollout=rollout,
            placement=placement,
            recording=RecordingSpec.from_dict(
                _mapping(data.get("recording", {}), "robotics job.recording")
            ),
            evaluation=evaluation,
            metadata=_mapping(data.get("metadata", {}), "robotics job.metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)

    def job_hash(self) -> str:
        return runtime_contract_hash(self.to_dict())

    def runtime_resources(self) -> RuntimeResources:
        simulator = self.placement.simulator_resources
        policy = self.placement.policy_resources
        coordinator = self.placement.coordinator_resources
        gpu_count = coordinator.gpu_count + (
            max(simulator.gpu_count, policy.gpu_count)
            if self.placement.gpu_sharing
            else simulator.gpu_count + policy.gpu_count
        )
        gpu_types = {
            item.gpu_type
            for item in (simulator, policy, coordinator)
            if item.gpu_count > 0 and item.gpu_type is not None
        }
        shm_values = [
            item.shm_size_gb
            for item in (simulator, policy, coordinator)
            if item.shm_size_gb is not None
        ]
        return RuntimeResources.from_dict(
            {
                "cpu_cores": simulator.cpu_cores
                + policy.cpu_cores
                + coordinator.cpu_cores,
                "memory_gb": simulator.memory_gb
                + policy.memory_gb
                + coordinator.memory_gb,
                "disk_gb": simulator.disk_gb + policy.disk_gb + coordinator.disk_gb,
                "gpu_count": gpu_count,
                "timeout_seconds": max(
                    simulator.timeout_seconds,
                    policy.timeout_seconds,
                    coordinator.timeout_seconds,
                ),
                **({"gpu_type": next(iter(gpu_types))} if len(gpu_types) == 1 else {}),
                **({"shm_size_gb": max(shm_values)} if shm_values else {}),
            }
        )


@dataclass(frozen=True)
class EpisodeManifest:
    schema_version: str
    run_id: str
    episode_id: str
    seed: int
    status: str
    job_hash: str
    simulator_package_hash: str
    task_package_hash: str
    policy_deployment_hash: str
    policy_deployment_id: str
    policy_revision: str
    simulator_image: str
    step_count: int
    metrics: Dict[str, Any]
    artifacts: List[ArtifactRef]
    started_at: str
    finished_at: str
    requested_placement: Dict[str, Any]
    actual_placement: Dict[str, Any]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeManifest":
        _strict_keys(
            data,
            {
                "schema_version",
                "run_id",
                "episode_id",
                "seed",
                "status",
                "job_hash",
                "simulator_package_hash",
                "task_package_hash",
                "policy_deployment_hash",
                "policy_deployment_id",
                "policy_revision",
                "simulator_image",
                "step_count",
                "metrics",
                "artifacts",
                "started_at",
                "finished_at",
                "requested_placement",
                "actual_placement",
                "error_code",
                "error_message",
                "runtime_metadata",
            },
            "episode manifest",
        )
        _schema(data, EPISODE_MANIFEST_SCHEMA_VERSION, "episode manifest")
        simulator_image = _string(
            _require(data, "simulator_image", "episode manifest"),
            "episode manifest.simulator_image",
        )
        if not _IMAGE_DIGEST_RE.fullmatch(simulator_image):
            raise RobotContractError(
                "episode manifest.simulator_image must be pinned by an OCI sha256 digest"
            )
        return cls(
            schema_version=EPISODE_MANIFEST_SCHEMA_VERSION,
            run_id=_string(_require(data, "run_id", "episode manifest"), "episode manifest.run_id"),
            episode_id=_string(
                _require(data, "episode_id", "episode manifest"), "episode manifest.episode_id"
            ),
            seed=_integer(_require(data, "seed", "episode manifest"), "episode manifest.seed"),
            status=_choice(
                _require(data, "status", "episode manifest"),
                EPISODE_STATUSES,
                "episode manifest.status",
            ),
            job_hash=_sha256(
                _require(data, "job_hash", "episode manifest"), "episode manifest.job_hash"
            ),
            simulator_package_hash=_sha256(
                _require(data, "simulator_package_hash", "episode manifest"),
                "episode manifest.simulator_package_hash",
            ),
            task_package_hash=_sha256(
                _require(data, "task_package_hash", "episode manifest"),
                "episode manifest.task_package_hash",
            ),
            policy_deployment_hash=_sha256(
                _require(data, "policy_deployment_hash", "episode manifest"),
                "episode manifest.policy_deployment_hash",
            ),
            policy_deployment_id=_string(
                _require(data, "policy_deployment_id", "episode manifest"),
                "episode manifest.policy_deployment_id",
            ),
            policy_revision=_string(
                _require(data, "policy_revision", "episode manifest"),
                "episode manifest.policy_revision",
            ),
            simulator_image=simulator_image,
            step_count=_nonnegative_int(
                _require(data, "step_count", "episode manifest"),
                "episode manifest.step_count",
            ),
            metrics=_mapping(data.get("metrics", {}), "episode manifest.metrics"),
            artifacts=[
                ArtifactRef.from_dict(_mapping(value, "episode manifest.artifacts[]"))
                for value in _list(data.get("artifacts", []), "episode manifest.artifacts")
            ],
            started_at=_string(
                _require(data, "started_at", "episode manifest"), "episode manifest.started_at"
            ),
            finished_at=_string(
                _require(data, "finished_at", "episode manifest"), "episode manifest.finished_at"
            ),
            requested_placement=_mapping(
                _require(data, "requested_placement", "episode manifest"),
                "episode manifest.requested_placement",
            ),
            actual_placement=_mapping(
                _require(data, "actual_placement", "episode manifest"),
                "episode manifest.actual_placement",
            ),
            error_code=_optional_string(data.get("error_code"), "episode manifest.error_code"),
            error_message=_optional_string(
                data.get("error_message"), "episode manifest.error_message"
            ),
            runtime_metadata=_mapping(
                data.get("runtime_metadata", {}), "episode manifest.runtime_metadata"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _contract_dict(self)
