"""Transport-neutral service contracts for robotics rollout components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import RobotContractError
from .env import StepResult, VectorStepResult
from .runtime_contracts import ActionChunk, RoboticsJobSpec, canonical_runtime_json

SIM_SERVICE_PROTOCOL_VERSION = "sim-service/v1"
POLICY_SERVICE_PROTOCOL_VERSION = "policy-service/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLICY_STATE_MODELS = {"stateless", "recurrent", "history", "dual_system"}
_RESET_GRANULARITIES = {"session", "batch", "environment"}


class RobotServiceContractError(RobotContractError):
    """A simulator or policy service descriptor violates the public contract."""


def _mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotServiceContractError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RobotServiceContractError(f"{where} keys must be strings")
    return dict(value)


def _strict_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    if any(not isinstance(key, str) for key in data):
        raise RobotServiceContractError(f"{where} keys must be strings")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise RobotServiceContractError(f"{where} has unknown fields {unknown}")


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RobotServiceContractError(f"{where} must be a non-empty string")
    return value


def _sha256(value: Any, where: str) -> str:
    result = _string(value, where)
    if not _SHA256_RE.fullmatch(result):
        raise RobotServiceContractError(f"{where} must be a lowercase SHA-256 digest")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RobotServiceContractError(f"{where} must be a positive integer")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise RobotServiceContractError(f"{where} must be a boolean")
    return value


def _string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RobotServiceContractError(f"{where} must be a string array")
    if len(set(value)) != len(value):
        raise RobotServiceContractError(f"{where} cannot contain duplicates")
    return list(value)


def _choice(value: Any, choices: set[str], where: str) -> str:
    result = _string(value, where)
    if result not in choices:
        raise RobotServiceContractError(f"{where} must be one of {sorted(choices)}")
    return result


@dataclass(frozen=True)
class SimulatorServiceDescriptor:
    protocol_version: str
    session_id: str
    simulator_package_hash: str
    task_package_hash: str
    vector_width: int
    capabilities: list[str]
    observation_schema: Dict[str, Any]
    action_spec: Dict[str, Any]
    transport: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulatorServiceDescriptor":
        _strict_keys(
            data,
            {
                "protocol_version",
                "session_id",
                "simulator_package_hash",
                "task_package_hash",
                "vector_width",
                "capabilities",
                "observation_schema",
                "action_spec",
                "transport",
            },
            "simulator descriptor",
        )
        protocol = _string(data.get("protocol_version"), "simulator descriptor.protocol_version")
        if protocol != SIM_SERVICE_PROTOCOL_VERSION:
            raise RobotServiceContractError(
                f"simulator descriptor.protocol_version must be {SIM_SERVICE_PROTOCOL_VERSION!r}"
            )
        return cls(
            protocol_version=protocol,
            session_id=_string(data.get("session_id"), "simulator descriptor.session_id"),
            simulator_package_hash=_sha256(
                data.get("simulator_package_hash"),
                "simulator descriptor.simulator_package_hash",
            ),
            task_package_hash=_sha256(
                data.get("task_package_hash"), "simulator descriptor.task_package_hash"
            ),
            vector_width=_positive_int(
                data.get("vector_width"), "simulator descriptor.vector_width"
            ),
            capabilities=_string_list(
                data.get("capabilities"), "simulator descriptor.capabilities"
            ),
            observation_schema=_mapping(
                data.get("observation_schema"), "simulator descriptor.observation_schema"
            ),
            action_spec=_mapping(data.get("action_spec"), "simulator descriptor.action_spec"),
            transport=_string(data.get("transport"), "simulator descriptor.transport"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "session_id": self.session_id,
            "simulator_package_hash": self.simulator_package_hash,
            "task_package_hash": self.task_package_hash,
            "vector_width": self.vector_width,
            "capabilities": list(self.capabilities),
            "observation_schema": dict(self.observation_schema),
            "action_spec": dict(self.action_spec),
            "transport": self.transport,
        }


@dataclass(frozen=True)
class PolicyServiceDescriptor:
    protocol_version: str
    session_id: str
    policy_deployment_hash: str
    policy_deployment_id: str
    policy_revision: str
    batch_size: int
    max_horizon: int
    state_model: str
    reset_granularity: str
    deterministic: bool
    observation_schema: Dict[str, Any]
    action_spec: Dict[str, Any]
    transport: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyServiceDescriptor":
        _strict_keys(
            data,
            {
                "protocol_version",
                "session_id",
                "policy_deployment_hash",
                "policy_deployment_id",
                "policy_revision",
                "batch_size",
                "max_horizon",
                "state_model",
                "reset_granularity",
                "deterministic",
                "observation_schema",
                "action_spec",
                "transport",
            },
            "policy descriptor",
        )
        protocol = _string(data.get("protocol_version"), "policy descriptor.protocol_version")
        if protocol != POLICY_SERVICE_PROTOCOL_VERSION:
            raise RobotServiceContractError(
                f"policy descriptor.protocol_version must be {POLICY_SERVICE_PROTOCOL_VERSION!r}"
            )
        return cls(
            protocol_version=protocol,
            session_id=_string(data.get("session_id"), "policy descriptor.session_id"),
            policy_deployment_hash=_sha256(
                data.get("policy_deployment_hash"),
                "policy descriptor.policy_deployment_hash",
            ),
            policy_deployment_id=_string(
                data.get("policy_deployment_id"), "policy descriptor.policy_deployment_id"
            ),
            policy_revision=_string(
                data.get("policy_revision"), "policy descriptor.policy_revision"
            ),
            batch_size=_positive_int(data.get("batch_size"), "policy descriptor.batch_size"),
            max_horizon=_positive_int(data.get("max_horizon"), "policy descriptor.max_horizon"),
            state_model=_choice(
                data.get("state_model"), _POLICY_STATE_MODELS, "policy descriptor.state_model"
            ),
            reset_granularity=_choice(
                data.get("reset_granularity"),
                _RESET_GRANULARITIES,
                "policy descriptor.reset_granularity",
            ),
            deterministic=_boolean(data.get("deterministic"), "policy descriptor.deterministic"),
            observation_schema=_mapping(
                data.get("observation_schema"), "policy descriptor.observation_schema"
            ),
            action_spec=_mapping(data.get("action_spec"), "policy descriptor.action_spec"),
            transport=_string(data.get("transport"), "policy descriptor.transport"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "session_id": self.session_id,
            "policy_deployment_hash": self.policy_deployment_hash,
            "policy_deployment_id": self.policy_deployment_id,
            "policy_revision": self.policy_revision,
            "batch_size": self.batch_size,
            "max_horizon": self.max_horizon,
            "state_model": self.state_model,
            "reset_granularity": self.reset_granularity,
            "deterministic": self.deterministic,
            "observation_schema": dict(self.observation_schema),
            "action_spec": dict(self.action_spec),
            "transport": self.transport,
        }


@runtime_checkable
class SimulatorServiceClient(Protocol):
    num_envs: int

    def describe(self) -> SimulatorServiceDescriptor: ...

    def reset(self, seed: Any = None, options: Mapping[str, Any] | None = None) -> Any: ...

    def step(self, action: Any) -> StepResult | VectorStepResult: ...

    def capture(self, request: Mapping[str, Any]) -> Any: ...

    def close(self) -> None: ...


@runtime_checkable
class PolicyServiceClient(Protocol):
    def describe(self) -> PolicyServiceDescriptor: ...

    def reset(self, indices: Sequence[int] | None = None) -> None: ...

    def act(
        self,
        observation: Any,
        *,
        seed: int | Sequence[int] | None = None,
        step_ids: Sequence[int] | None = None,
    ) -> ActionChunk: ...

    def close(self) -> None: ...


def validate_simulator_descriptor(
    job: RoboticsJobSpec,
    simulator: SimulatorServiceDescriptor,
) -> None:
    """Validate one simulator service independently from policy deployment."""

    expected_observation = {
        name: spec.to_dict() for name, spec in job.task.observation_schema.items()
    }
    checks = {
        "simulator package hash": simulator.simulator_package_hash == job.simulator.package_hash(),
        "task package hash": simulator.task_package_hash == job.task.package_hash(),
        "simulator vector width": simulator.vector_width == job.rollout.vector_width,
        "simulator capabilities": set(simulator.capabilities) == set(job.simulator.capabilities),
        "simulator observation schema": canonical_runtime_json(simulator.observation_schema)
        == canonical_runtime_json(expected_observation),
        "simulator action spec": canonical_runtime_json(simulator.action_spec)
        == canonical_runtime_json(job.task.action_spec.to_dict()),
    }
    _raise_descriptor_failures(checks)


def validate_policy_descriptor(
    job: RoboticsJobSpec,
    policy: PolicyServiceDescriptor,
) -> None:
    """Validate one local or hosted policy service independently from a simulator."""

    expected_observation = {
        name: spec.to_dict() for name, spec in job.policy.observation_schema.items()
    }
    checks = {
        "policy deployment hash": policy.policy_deployment_hash == job.policy.deployment_hash(),
        "policy deployment id": policy.policy_deployment_id == job.policy.deployment_id,
        "policy revision": policy.policy_revision == job.policy.revision,
        "policy batch size": policy.batch_size == job.rollout.vector_width,
        "policy max horizon": policy.max_horizon == job.policy.max_horizon,
        "policy state model": policy.state_model == job.policy.state_model,
        "policy reset granularity": policy.reset_granularity == job.policy.reset_granularity,
        "policy deterministic": policy.deterministic == job.policy.deterministic,
        "policy observation schema": canonical_runtime_json(policy.observation_schema)
        == canonical_runtime_json(expected_observation),
        "policy action spec": canonical_runtime_json(policy.action_spec)
        == canonical_runtime_json(job.policy.action_spec.to_dict()),
    }
    _raise_descriptor_failures(checks)


def validate_component_descriptors(
    job: RoboticsJobSpec,
    simulator: SimulatorServiceDescriptor,
    policy: PolicyServiceDescriptor,
) -> None:
    """Validate separately owned simulator and policy services against one job."""

    validate_simulator_descriptor(job, simulator)
    validate_policy_descriptor(job, policy)


def _raise_descriptor_failures(checks: Mapping[str, bool]) -> None:
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise RobotServiceContractError(f"service descriptor mismatch: {', '.join(failures)}")


__all__ = [
    "POLICY_SERVICE_PROTOCOL_VERSION",
    "SIM_SERVICE_PROTOCOL_VERSION",
    "PolicyServiceClient",
    "PolicyServiceDescriptor",
    "RobotServiceContractError",
    "SimulatorServiceClient",
    "SimulatorServiceDescriptor",
    "validate_component_descriptors",
    "validate_policy_descriptor",
    "validate_simulator_descriptor",
]
