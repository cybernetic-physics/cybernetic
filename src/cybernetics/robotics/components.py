"""Developer-facing drivers for simulator and hosted policy components."""

from __future__ import annotations

import importlib
import time
from contextlib import suppress
from functools import partial
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .env import RobotEnv, VectorRobotEnv
from .gymnasium import GymnasiumRobotEnvAdapter, GymnasiumVectorEnvAdapter
from .runtime_contracts import (
    ActionChunk,
    PolicyDeploymentSpec,
    RoboticsJobSpec,
    TensorSpec,
)
from .runtime_services import (
    POLICY_SERVICE_PROTOCOL_VERSION,
    SIM_SERVICE_PROTOCOL_VERSION,
    PolicyServiceDescriptor,
    SimulatorServiceDescriptor,
)


class RobotComponentError(RuntimeError):
    """A simulator or policy component could not satisfy its public contract."""


class LocalSimulatorComponent:
    """Own one manifest-created environment behind ``SimulatorServiceClient``."""

    def __init__(self, job: RoboticsJobSpec, environment: Any) -> None:
        self.job = job
        self.environment = environment
        self.num_envs = job.rollout.vector_width
        self._closed = False
        self._descriptor = SimulatorServiceDescriptor(
            protocol_version=SIM_SERVICE_PROTOCOL_VERSION,
            session_id=f"sim-{uuid4().hex}",
            simulator_package_hash=job.simulator.package_hash(),
            task_package_hash=job.task.package_hash(),
            vector_width=job.rollout.vector_width,
            capabilities=list(job.simulator.capabilities),
            observation_schema={
                name: spec.to_dict() for name, spec in job.task.observation_schema.items()
            },
            action_spec=job.task.action_spec.to_dict(),
            transport="in_process",
        )

    def describe(self) -> SimulatorServiceDescriptor:
        self._require_open()
        return self._descriptor

    def reset(self, seed: Any = None, options: Mapping[str, Any] | None = None) -> Any:
        self._require_open()
        return self.environment.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        self._require_open()
        return self.environment.step(action)

    def capture(self, request: Mapping[str, Any]) -> Any:
        self._require_open()
        capture = getattr(self.environment, "capture", None)
        if not callable(capture):
            raise RobotComponentError("simulator component does not support capture()")
        return capture(request)

    def close(self) -> None:
        if self._closed:
            return
        self.environment.close()
        self._closed = True

    def __enter__(self) -> "LocalSimulatorComponent":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RobotComponentError("simulator component is closed")


class HostedWorldlinesPolicyClient:
    """Drive a hosted policy through the existing tenant-aware sampling API."""

    def __init__(
        self,
        job_or_spec: RoboticsJobSpec | PolicyDeploymentSpec,
        descriptor_or_sampling_client: PolicyServiceDescriptor | Any,
        sampling_client: Any | None = None,
        *,
        service_client: Any = None,
        request_timeout_seconds: float = 120,
    ) -> None:
        if isinstance(job_or_spec, RoboticsJobSpec):
            job = job_or_spec
            spec = job.policy
            descriptor = _policy_descriptor(job)
            resolved_sampling_client = descriptor_or_sampling_client
        else:
            job = None
            spec = job_or_spec
            if not isinstance(descriptor_or_sampling_client, PolicyServiceDescriptor):
                raise RobotComponentError("hosted policy descriptor is required")
            descriptor = descriptor_or_sampling_client
            resolved_sampling_client = sampling_client
        if spec.source != "worldlines":
            raise RobotComponentError("hosted policy requires source='worldlines'")
        if resolved_sampling_client is None:
            raise RobotComponentError("hosted policy sampling client is required")
        if request_timeout_seconds <= 0:
            raise RobotComponentError("Worldlines request timeout must be positive")
        self.job = job
        self.spec = spec
        self._sampling_client = resolved_sampling_client
        self._service_client = service_client
        self._request_timeout_seconds = request_timeout_seconds
        self._sequence_ids = [
            self._new_sequence_id(index) for index in range(descriptor.batch_size)
        ]
        self._last_step_ids = [-1] * descriptor.batch_size
        self._pending_reset = [True] * descriptor.batch_size
        self._closed = False
        self._descriptor = descriptor

    @classmethod
    def connect(
        cls,
        job_or_spec: RoboticsJobSpec | PolicyDeploymentSpec,
        descriptor: PolicyServiceDescriptor | None = None,
        *,
        run_id: str,
    ) -> "HostedWorldlinesPolicyClient":
        from cybernetics.lib.public_interfaces import ServiceClient

        service_client = ServiceClient(
            user_metadata={
                "robotics_run_id": run_id,
                "policy_deployment_id": (
                    job_or_spec.policy.deployment_id
                    if isinstance(job_or_spec, RoboticsJobSpec)
                    else job_or_spec.deployment_id
                ),
            }
        )
        try:
            spec = job_or_spec.policy if isinstance(job_or_spec, RoboticsJobSpec) else job_or_spec
            configured_base_model = spec.config.get("base_model", spec.model_id)
            if not isinstance(configured_base_model, str) or not configured_base_model:
                raise RobotComponentError("Worldlines policy base_model must be a non-empty string")
            sampling_client = service_client.create_sampling_client(
                model_path=spec.checkpoint_ref,
                base_model=None if spec.checkpoint_ref else configured_base_model,
                timeout=float(spec.config.get("startup_timeout_seconds", 900)),
            )
        except Exception:
            with suppress(Exception):
                _cancel_owned_session(service_client)
            raise
        return cls(
            job_or_spec,
            sampling_client if descriptor is None else descriptor,
            None if descriptor is None else sampling_client,
            service_client=service_client,
            request_timeout_seconds=float(spec.config.get("request_timeout_seconds", 120)),
        )

    def describe(self) -> PolicyServiceDescriptor:
        self._require_open()
        return self._descriptor

    def reset(self, indices: Sequence[int] | None = None) -> None:
        self._require_open()
        targets = (
            range(self._descriptor.batch_size)
            if indices is None
            else _indices(indices, self._descriptor.batch_size)
        )
        for index in targets:
            self._sequence_ids[index] = self._new_sequence_id(index)
            self._last_step_ids[index] = -1
            self._pending_reset[index] = True

    def act(
        self,
        observation: Any,
        *,
        seed: int | Sequence[int] | None = None,
        step_ids: Sequence[int] | None = None,
    ) -> ActionChunk:
        from cybernetics import types

        self._require_open()
        width = self._descriptor.batch_size
        normalized_steps = _step_ids(step_ids, self._last_step_ids, self._pending_reset)
        seeds = _seeds(seed, width)
        conditioning = _policy_conditioning(
            observation,
            self.spec.observation_schema,
            batch_size=width,
        )
        context = types.PolicySessionContext(
            sequence_ids=list(self._sequence_ids),
            step_ids=normalized_steps,
            reset_mask=list(self._pending_reset),
            seeds=seeds,
        )
        started = time.perf_counter()
        response = self._sampling_client.sample(
            prompt=types.ModelInput.empty(),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=1,
                seed=seeds[0] if width == 1 else None,
            ),
            conditioning=conditioning,
            policy_context=context,
        ).result(timeout=self._request_timeout_seconds)
        round_trip_ms = (time.perf_counter() - started) * 1000
        if response.action_chunk is None:
            raise RobotComponentError("Worldlines sampler returned no action_chunk")
        values = response.action_chunk.to_numpy()
        produced_horizon = _validate_action_shape(
            values,
            batch_size=width,
            action_shape=self.spec.action_spec.tensor.shape,
            max_horizon=self.spec.max_horizon,
        )
        self._last_step_ids = normalized_steps
        self._pending_reset = [False] * width
        return ActionChunk(
            values=values,
            representation=self.spec.action_spec.representation,
            requested_horizon=self.spec.action_spec.horizon,
            produced_horizon=produced_horizon,
            inference_latency_ms=round_trip_ms,
            auxiliary={
                "policy_transport": "cybernetics_sampling_v1",
                "sampling_session_id": self._sampling_client.sampling_session_id,
                "sequence_ids": list(self._sequence_ids),
                "step_ids": list(normalized_steps),
                "reset_mask": list(context.reset_mask),
                "trajectory_returned": response.trajectory is not None,
                "predicted_video_returned": (
                    response.predicted_video is not None or response.video is not None
                ),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._service_client is not None:
            _cancel_owned_session(self._service_client)
        self._closed = True

    def __enter__(self) -> "HostedWorldlinesPolicyClient":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _new_sequence_id(self, index: int) -> str:
        return f"lane-{index}-{uuid4().hex}"

    def _require_open(self) -> None:
        if self._closed:
            raise RobotComponentError("hosted Worldlines policy client is closed")


def open_simulator_component(job: RoboticsJobSpec) -> LocalSimulatorComponent:
    """Construct, readiness-check, and own the simulator selected by ``job``."""

    environment = _load_environment(job)
    try:
        _check_environment_readiness(job, environment)
        return LocalSimulatorComponent(job, environment)
    except Exception:
        environment.close()
        raise


def open_worldlines_policy_component(
    job: RoboticsJobSpec,
    *,
    run_id: str,
) -> HostedWorldlinesPolicyClient:
    """Open the job's hosted policy without exposing a worker endpoint."""

    return HostedWorldlinesPolicyClient.connect(job, run_id=run_id)


def _load_environment(job: RoboticsJobSpec) -> Any:
    factory = job.simulator.factory
    if factory.kind == "gymnasium":
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RobotComponentError(
                "Gymnasium environment requested but gymnasium is not installed"
            ) from exc
        kwargs = {**factory.kwargs, **job.task.adapter_config}
        if job.rollout.vector_width > 1 and factory.vectorization == "sync":
            native = gym.vector.SyncVectorEnv(
                [
                    partial(gym.make, factory.target, **kwargs)
                    for _ in range(job.rollout.vector_width)
                ]
            )
        else:
            if job.rollout.vector_width > 1:
                kwargs["num_envs"] = job.rollout.vector_width
            native = gym.make(factory.target, **kwargs)
    elif factory.kind in {"python", "lerobot_envhub"}:
        native = _load_callable(factory.target)(
            simulator=job.simulator,
            task=job.task,
            vector_width=job.rollout.vector_width,
            **factory.kwargs,
        )
    else:  # pragma: no cover - contract parsing rejects unknown kinds
        raise RobotComponentError(f"unsupported environment factory kind {factory.kind!r}")

    if job.rollout.vector_width > 1:
        if isinstance(native, VectorRobotEnv):
            if native.num_envs != job.rollout.vector_width:
                native.close()
                raise RobotComponentError(
                    f"factory returned num_envs={native.num_envs}, "
                    f"expected {job.rollout.vector_width}"
                )
            return native
        adapter = GymnasiumVectorEnvAdapter(native, backend_id=job.simulator.simulator)
        if adapter.num_envs != job.rollout.vector_width:
            adapter.close()
            raise RobotComponentError(
                f"factory returned num_envs={adapter.num_envs}, expected {job.rollout.vector_width}"
            )
        return adapter
    if isinstance(native, RobotEnv):
        return native
    return GymnasiumRobotEnvAdapter(native, backend_id=job.simulator.simulator)


def _check_environment_readiness(job: RoboticsJobSpec, environment: Any) -> None:
    readiness = job.simulator.readiness
    if readiness.kind == "factory":
        return
    method = getattr(environment, readiness.target or "", None)
    if not callable(method):
        raise RobotComponentError(
            f"environment readiness method {readiness.target!r} is not callable"
        )
    result = method()
    if result is False or (isinstance(result, Mapping) and result.get("ready") is False):
        raise RobotComponentError(
            f"environment readiness method {readiness.target!r} reported not ready"
        )


def _load_callable(target: str) -> Any:
    if ":" not in target:
        raise RobotComponentError("native environment factory must be 'module:callable'")
    module_name, attribute = target.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(value):
        raise RobotComponentError(f"environment factory {target!r} is not callable")
    return value


def _policy_conditioning(
    observation: Any,
    schema: Mapping[str, TensorSpec],
    *,
    batch_size: int,
) -> Any:
    from cybernetics import types

    if isinstance(observation, Mapping):
        values = observation
    elif len(schema) == 1:
        values = {next(iter(schema)): observation}
    else:
        raise RobotComponentError("Worldlines policy observation must be a mapping")
    conditioning: dict[str, Any] = {}
    for name, spec in schema.items():
        if name not in values:
            raise RobotComponentError(f"Worldlines policy observation is missing {name!r}")
        array = np.asarray(values[name])
        expected_shape = (batch_size, *spec.shape)
        if batch_size == 1 and tuple(array.shape) == tuple(spec.shape):
            array = np.expand_dims(array, axis=0)
        if tuple(array.shape) != expected_shape:
            raise RobotComponentError(
                f"Worldlines policy observation {name!r} has shape {array.shape}, "
                f"expected {expected_shape}"
            )
        if spec.dtype == "utf8":
            conditioning[name] = types.TextData(
                data=[str(value) for value in array.flatten().tolist()],
                dtype="utf8",
                shape=list(array.shape),
            )
            continue
        if array.dtype.kind not in {"b", "f", "i", "u"}:
            raise RobotComponentError(
                f"Worldlines policy observation {name!r} must be numeric; got {array.dtype}"
            )
        conditioning[name] = types.TensorData.from_numpy(array)
    return conditioning


def _step_ids(
    step_ids: Sequence[int] | None,
    previous: Sequence[int],
    pending_reset: Sequence[bool],
) -> list[int]:
    normalized = (
        [last + 1 for last in previous]
        if step_ids is None
        else [int(step_id) for step_id in step_ids]
    )
    if len(normalized) != len(previous):
        raise RobotComponentError("policy step_ids must match the rollout vector width")
    for index, step_id in enumerate(normalized):
        if step_id < 0:
            raise RobotComponentError("policy step_ids must be nonnegative")
        if not pending_reset[index] and step_id <= previous[index]:
            raise RobotComponentError("policy step_ids must increase monotonically per lane")
    return normalized


def _seeds(seed: int | Sequence[int] | None, width: int) -> list[int]:
    if seed is None:
        return [0] * width
    if isinstance(seed, int):
        return [seed] * width
    normalized = [int(value) for value in seed]
    if len(normalized) != width:
        raise RobotComponentError("policy seeds must match the rollout vector width")
    return normalized


def _indices(indices: Sequence[int], width: int) -> list[int]:
    normalized = [int(index) for index in indices]
    if len(set(normalized)) != len(normalized):
        raise RobotComponentError("policy reset indices must be unique")
    if any(index < 0 or index >= width for index in normalized):
        raise RobotComponentError("policy reset index is outside the rollout vector width")
    return normalized


def _validate_action_shape(
    values: Any,
    *,
    batch_size: int,
    action_shape: Sequence[int],
    max_horizon: int,
) -> int:
    shape = tuple(int(size) for size in np.asarray(values).shape)
    expected_suffix = tuple(action_shape)
    if len(shape) != len(expected_suffix) + 2:
        raise RobotComponentError(
            "Worldlines action_chunk must have shape [batch, horizon, *action_shape]"
        )
    if shape[0] != batch_size or shape[2:] != expected_suffix:
        raise RobotComponentError(
            f"Worldlines action_chunk has shape {shape}, expected "
            f"[{batch_size}, horizon, {', '.join(str(size) for size in expected_suffix)}]"
        )
    if shape[1] <= 0 or shape[1] > max_horizon:
        raise RobotComponentError("Worldlines action_chunk horizon exceeds deployment limits")
    return shape[1]


def _cancel_owned_session(service_client: Any) -> None:
    service_client.create_rest_client().cancel_session(service_client.session_id).result(timeout=30)


def _policy_descriptor(job: RoboticsJobSpec) -> PolicyServiceDescriptor:
    spec = job.policy
    return PolicyServiceDescriptor(
        protocol_version=POLICY_SERVICE_PROTOCOL_VERSION,
        session_id=f"policy-{uuid4().hex}",
        policy_deployment_hash=spec.deployment_hash(),
        policy_deployment_id=spec.deployment_id,
        policy_revision=spec.revision,
        batch_size=job.rollout.vector_width,
        max_horizon=spec.max_horizon,
        state_model=spec.state_model,
        reset_granularity=spec.reset_granularity,
        deterministic=spec.deterministic,
        observation_schema={name: value.to_dict() for name, value in spec.observation_schema.items()},
        action_spec=spec.action_spec.to_dict(),
        transport="cybernetics_sampling_v1",
    )


__all__ = [
    "HostedWorldlinesPolicyClient",
    "LocalSimulatorComponent",
    "RobotComponentError",
    "open_simulator_component",
    "open_worldlines_policy_component",
]
