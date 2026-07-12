"""Benchmark-first robotics experiment and preflight contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .contracts import RobotContractError
from .runtime_contracts import RoboticsJobSpec

ROBOTICS_EXPERIMENT_SCHEMA_VERSION = "robotics-experiment/v1"
ROBOTICS_CATALOG_SCHEMA_VERSION = "robotics-benchmark-catalog/v1"
ROBOTICS_PREFLIGHT_SCHEMA_VERSION = "robotics-preflight/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_STATUSES = {"available", "setup_required", "planned"}
_PREFLIGHT_STATUSES = {"pass", "warning", "blocked"}
_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1


@dataclass(frozen=True)
class RoboticsPolicyTemplate:
    id: str
    name: str
    model_id: str
    runtime_family: str
    source: str
    status: str
    description: str
    requirements: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboticsPolicyTemplate":
        return cls(
            id=_string(data.get("id"), "catalog policy.id"),
            name=_string(data.get("name"), "catalog policy.name"),
            model_id=_string(data.get("modelId"), "catalog policy.modelId"),
            runtime_family=_string(
                data.get("runtimeFamily"), "catalog policy.runtimeFamily"
            ),
            source=_choice(
                data.get("source"), {"fixture", "worldlines", "local"}, "catalog policy.source"
            ),
            status=_choice(data.get("status"), _CATALOG_STATUSES, "catalog policy.status"),
            description=_string(data.get("description"), "catalog policy.description"),
            requirements=_string_tuple(
                data.get("requirements", []), "catalog policy.requirements"
            ),
        )


@dataclass(frozen=True)
class RoboticsBenchmarkTemplate:
    id: str
    version: str
    name: str
    description: str
    family: str
    benchmark: str
    split: str
    status: str
    tags: tuple[str, ...]
    simulator: Mapping[str, Any]
    primary_metric: str
    native_metrics: tuple[str, ...]
    defaults: Mapping[str, Any]
    default_policy_id: str
    policies: tuple[RoboticsPolicyTemplate, ...]
    requirements: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboticsBenchmarkTemplate":
        policies_raw = _list(data.get("policies"), "catalog benchmark.policies")
        policies = tuple(
            RoboticsPolicyTemplate.from_dict(_mapping(item, "catalog benchmark policy"))
            for item in policies_raw
        )
        if not policies:
            raise RobotContractError("catalog benchmark.policies must not be empty")
        default_policy_id = _string(
            data.get("defaultPolicyId"), "catalog benchmark.defaultPolicyId"
        )
        if default_policy_id not in {policy.id for policy in policies}:
            raise RobotContractError("catalog default policy is not in benchmark.policies")
        return cls(
            id=_string(data.get("id"), "catalog benchmark.id"),
            version=_string(data.get("version"), "catalog benchmark.version"),
            name=_string(data.get("name"), "catalog benchmark.name"),
            description=_string(data.get("description"), "catalog benchmark.description"),
            family=_string(data.get("family"), "catalog benchmark.family"),
            benchmark=_string(data.get("benchmark"), "catalog benchmark.benchmark"),
            split=_string(data.get("split"), "catalog benchmark.split"),
            status=_choice(data.get("status"), _CATALOG_STATUSES, "catalog benchmark.status"),
            tags=_string_tuple(data.get("tags", []), "catalog benchmark.tags"),
            simulator=_mapping(data.get("simulator"), "catalog benchmark.simulator"),
            primary_metric=_string(
                data.get("primaryMetric"), "catalog benchmark.primaryMetric"
            ),
            native_metrics=_string_tuple(
                data.get("nativeMetrics", []), "catalog benchmark.nativeMetrics"
            ),
            defaults=_mapping(data.get("defaults"), "catalog benchmark.defaults"),
            default_policy_id=default_policy_id,
            policies=policies,
            requirements=_string_tuple(
                data.get("requirements", []), "catalog benchmark.requirements"
            ),
        )

    def policy(self, policy_id: Optional[str] = None) -> RoboticsPolicyTemplate:
        selected = policy_id or self.default_policy_id
        for policy in self.policies:
            if policy.id == selected:
                return policy
        raise RobotContractError(
            f"policy {selected!r} is not compatible with benchmark {self.id!r}"
        )


@dataclass(frozen=True)
class RoboticsPreflightCheck:
    id: str
    status: str
    title: str
    message: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboticsPreflightCheck":
        return cls(
            id=_string(data.get("id"), "preflight check.id"),
            status=_choice(data.get("status"), _PREFLIGHT_STATUSES, "preflight check.status"),
            title=_string(data.get("title"), "preflight check.title"),
            message=_string(data.get("message"), "preflight check.message"),
        )


@dataclass(frozen=True)
class RoboticsPreflight:
    valid: bool
    launchable: bool
    source: Mapping[str, Any]
    checks: tuple[RoboticsPreflightCheck, ...]
    worldlines_billing: str
    job: Optional[RoboticsJobSpec] = None
    job_hash: Optional[str] = None
    resources: Optional[Mapping[str, Any]] = None
    estimated_max_usd: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboticsPreflight":
        if data.get("schemaVersion") != ROBOTICS_PREFLIGHT_SCHEMA_VERSION:
            raise RobotContractError(
                f"preflight.schemaVersion must be {ROBOTICS_PREFLIGHT_SCHEMA_VERSION!r}"
            )
        valid = _boolean(data.get("valid"), "preflight.valid")
        launchable = _boolean(data.get("launchable"), "preflight.launchable")
        checks = tuple(
            RoboticsPreflightCheck.from_dict(_mapping(item, "preflight check"))
            for item in _list(data.get("checks"), "preflight.checks")
        )
        if not checks:
            raise RobotContractError("preflight.checks must not be empty")
        job_raw = data.get("job")
        job = RoboticsJobSpec.from_dict(_mapping(job_raw, "preflight.job")) if job_raw else None
        job_hash = data.get("jobHash")
        if job_hash is not None:
            job_hash = _string(job_hash, "preflight.jobHash")
            if not _SHA256_RE.fullmatch(job_hash):
                raise RobotContractError("preflight.jobHash must be a lowercase SHA-256 digest")
        if job is not None and job_hash != job.job_hash():
            raise RobotContractError("preflight jobHash does not match the normalized job")
        if launchable and (
            not valid
            or job is None
            or job_hash is None
            or any(check.status == "blocked" for check in checks)
        ):
            raise RobotContractError("launchable preflight is missing a valid unblocked job")
        resources_raw = data.get("resources")
        estimate_raw = data.get("estimatedMaxUsd")
        if estimate_raw is not None and (
            isinstance(estimate_raw, bool) or not isinstance(estimate_raw, (int, float))
        ):
            raise RobotContractError("preflight.estimatedMaxUsd must be numeric")
        return cls(
            valid=valid,
            launchable=launchable,
            source=_mapping(data.get("source"), "preflight.source"),
            checks=checks,
            worldlines_billing=_choice(
                data.get("worldlinesBilling"),
                {"not_applicable", "separate_hosted_service"},
                "preflight.worldlinesBilling",
            ),
            job=job,
            job_hash=job_hash,
            resources=(
                _mapping(resources_raw, "preflight.resources")
                if resources_raw is not None
                else None
            ),
            estimated_max_usd=float(estimate_raw) if estimate_raw is not None else None,
        )

    def require_launchable_job(self) -> RoboticsJobSpec:
        if not self.launchable or self.job is None or self.job_hash is None:
            blocked = "; ".join(
                f"{check.title}: {check.message}"
                for check in self.checks
                if check.status == "blocked"
            )
            raise RobotContractError(f"robotics preflight is not launchable: {blocked or 'unknown'}")
        return self.job


def experiment_request(
    *,
    benchmark_id: str,
    policy_id: str,
    episode_start: int = 0,
    episodes: Optional[int] = None,
    root_seed: Optional[int] = None,
    vector_width: Optional[int] = None,
    max_steps: Optional[int] = None,
    job_name: Optional[str] = None,
    video: Optional[bool] = None,
    observations: Optional[bool] = None,
    actions: Optional[bool] = None,
    predictions: Optional[bool] = None,
    failure_clips: Optional[bool] = None,
    dataset_export: Optional[str] = None,
) -> dict[str, Any]:
    """Build the public benchmark composition request without simulator details."""

    episode_start = _integer(episode_start, "episode_start")
    if episode_start < 0 or episode_start > 99_999:
        raise RobotContractError("episode_start must be within episodes 0..99999")
    if episodes is not None:
        episodes = _integer(episodes, "episodes")
        if episodes <= 0 or episodes > 100_000 or episode_start + episodes > 100_000:
            raise RobotContractError("episodes must define a shard within episodes 0..99999")
    for name, value in (("vector_width", vector_width), ("max_steps", max_steps)):
        if value is not None:
            parsed = _integer(value, name)
            if parsed <= 0:
                raise RobotContractError(f"{name} must be positive")
            if name == "vector_width":
                vector_width = parsed
            else:
                max_steps = parsed
    if root_seed is not None:
        root_seed = _integer(root_seed, "root_seed", json_safe=True)
        last_seed = root_seed + episode_start + (episodes or 1) - 1
        if abs(last_seed) > _MAX_JSON_SAFE_INTEGER:
            raise RobotContractError("resolved episode seeds must be JSON-safe integers")
    if dataset_export is not None and dataset_export not in {"none", "jsonl", "lerobot_v3"}:
        raise RobotContractError("dataset_export must be none, jsonl, or lerobot_v3")
    for name, value in (
        ("video", video),
        ("observations", observations),
        ("actions", actions),
        ("predictions", predictions),
        ("failure_clips", failure_clips),
    ):
        if value is not None:
            _boolean(value, name)

    shard: dict[str, int] = {"start": episode_start}
    if episodes is not None:
        shard["count"] = episodes
    rollout = {
        key: value
        for key, value in {
            "rootSeed": root_seed,
            "vectorWidth": vector_width,
            "maxSteps": max_steps,
        }.items()
        if value is not None
    }
    evidence = {
        key: value
        for key, value in {
            "video": video,
            "observations": observations,
            "actions": actions,
            "predictions": predictions,
            "failureClips": failure_clips,
            "datasetExport": dataset_export,
        }.items()
        if value is not None
    }
    request: dict[str, Any] = {
        "schemaVersion": ROBOTICS_EXPERIMENT_SCHEMA_VERSION,
        "benchmarkId": _string(benchmark_id, "benchmark_id"),
        "policyId": _string(policy_id, "policy_id"),
        "episodeShard": shard,
        "rollout": rollout,
        "evidence": evidence,
    }
    if job_name is not None:
        request["jobName"] = _string(job_name, "job_name")
    return request


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotContractError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RobotContractError(f"{where} keys must be strings")
    return dict(value)


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise RobotContractError(f"{where} must be an array")
    return list(value)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RobotContractError(f"{where} must be a non-empty string")
    return value


def _choice(value: Any, choices: set[str], where: str) -> str:
    result = _string(value, where)
    if result not in choices:
        raise RobotContractError(f"{where} must be one of {sorted(choices)}")
    return result


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise RobotContractError(f"{where} must be a boolean")
    return value


def _integer(value: Any, where: str, *, json_safe: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RobotContractError(f"{where} must be an integer")
    if json_safe and abs(value) > _MAX_JSON_SAFE_INTEGER:
        raise RobotContractError(f"{where} must be a JSON-safe integer")
    return value


def _string_tuple(value: Any, where: str) -> tuple[str, ...]:
    values = _list(value, where)
    if any(not isinstance(item, str) or not item for item in values):
        raise RobotContractError(f"{where} must contain non-empty strings")
    return tuple(values)
