from __future__ import annotations

import sys
from concurrent.futures import Future
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pytest
from test_robotics_runtime_contracts import job_dict

from cybernetics import types
from cybernetics.robotics import (
    POLICY_SERVICE_PROTOCOL_VERSION,
    PolicyDeploymentSpec,
    PolicyServiceDescriptor,
    RobotComponentError,
    RoboticsJobSpec,
    StepResult,
    open_simulator_component,
    validate_simulator_descriptor,
)
from cybernetics.robotics.components import HostedWorldlinesPolicyClient


class _Environment:
    def __init__(self) -> None:
        self.closed = False
        self.last_seed: int | None = None

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del options
        self.last_seed = seed
        return {"position": [0.0]}

    def step(self, action: Mapping[str, Any]) -> StepResult:
        del action
        return StepResult(
            observation={"position": [1.0]},
            reward=1,
            terminated=False,
            truncated=False,
        )

    def render(self, mode: str = "rgb_array") -> list[int]:
        del mode
        return [0, 0, 0]

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"request": dict(request), "frame": self.render()}

    def get_state(self) -> Mapping[str, Any]:
        return {"position": [0.0]}

    def set_state(self, state: Mapping[str, Any]) -> None:
        del state

    def close(self) -> None:
        self.closed = True


def test_open_simulator_component_constructs_and_owns_manifest_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("robotics_component_fixture")
    environment = _Environment()
    module.create = lambda **kwargs: environment
    monkeypatch.setitem(sys.modules, module.__name__, module)
    payload = job_dict(vectorized=False)
    payload["simulator"]["factory"] = {
        "kind": "python",
        "target": "robotics_component_fixture:create",
        "kwargs": {},
    }
    job = RoboticsJobSpec.from_dict(payload)

    with open_simulator_component(job) as simulator:
        validate_simulator_descriptor(job, simulator.describe())
        simulator.reset(seed=17)
        assert environment.last_seed == 17
        assert simulator.capture({"camera": "rgb"})["frame"] == [0, 0, 0]

    assert environment.closed is True
    with pytest.raises(RobotComponentError, match="closed"):
        simulator.describe()


class _SamplingClient:
    sampling_session_id = "sample-owned"

    def __init__(self, action: types.TensorData, *, timing: dict[str, float] | None = None) -> None:
        self.action = action
        self.timing = timing
        self.calls: list[dict[str, Any]] = []

    def sample(self, **kwargs: Any) -> Future[types.SampleResponse]:
        self.calls.append(kwargs)
        future: Future[types.SampleResponse] = Future()
        future.set_result(
            types.SampleResponse(
                sequences=[],
                action_chunk=self.action,
                policy_timing=self.timing,
            )
        )
        return future


def _hosted_policy() -> PolicyDeploymentSpec:
    payload = job_dict(vectorized=False)["policy"]
    payload["source"] = "worldlines"
    payload["state_model"] = "recurrent"
    payload["checkpoint_ref"] = "worldlines://run/sampler_weights/checkpoint"
    payload["observation_schema"]["rgb"]["shape"] = [2, 2, 3]
    return PolicyDeploymentSpec.from_dict(payload)


def _hosted_descriptor(spec: PolicyDeploymentSpec) -> PolicyServiceDescriptor:
    return PolicyServiceDescriptor(
        protocol_version=POLICY_SERVICE_PROTOCOL_VERSION,
        session_id="policy-component",
        policy_deployment_hash=spec.deployment_hash(),
        policy_deployment_id=spec.deployment_id,
        policy_revision=spec.revision,
        batch_size=1,
        max_horizon=spec.max_horizon,
        state_model=spec.state_model,
        reset_granularity=spec.reset_granularity,
        deterministic=spec.deterministic,
        observation_schema={
            name: value.to_dict() for name, value in spec.observation_schema.items()
        },
        action_spec=spec.action_spec.to_dict(),
        transport="cybernetics_sampling_v1",
    )


def _observation() -> dict[str, Any]:
    return {
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "instruction": np.asarray(["go"], dtype=str),
    }


def test_hosted_policy_preserves_dtype_reset_and_timing_contracts() -> None:
    spec = _hosted_policy()
    sampling = _SamplingClient(
        types.TensorData(data=[1], dtype="int64", shape=[1, 1, 1]),
        timing={"queue_ms": 2.5, "inference_ms": 4.0},
    )
    client = HostedWorldlinesPolicyClient(spec, _hosted_descriptor(spec), sampling)

    first = client.act(_observation(), step_ids=[0])
    first_context = sampling.calls[-1]["policy_context"]
    client.reset([0])
    second = client.act(_observation(), step_ids=[0])
    second_context = sampling.calls[-1]["policy_context"]

    assert first_context.sequence_ids == second_context.sequence_ids
    assert second_context.reset_mask == [True]
    assert second_context.step_ids == [1]
    assert sampling.calls[0]["conditioning"]["rgb.__dtype__"] == types.TextData(
        data=["uint8"], dtype="utf8", shape=[1]
    )
    assert first.inference_latency_ms == 4.0
    assert second.auxiliary["policy_queue_latency_ms"] == 2.5
    assert second.auxiliary["policy_service_round_trip_ms"] >= 0


def test_hosted_policy_rejects_observation_dtype_and_action_bounds() -> None:
    spec = _hosted_policy()
    sampling = _SamplingClient(types.TensorData(data=[9], dtype="int64", shape=[1, 1, 1]))
    client = HostedWorldlinesPolicyClient(spec, _hosted_descriptor(spec), sampling)

    invalid_observation = _observation()
    invalid_observation["rgb"] = invalid_observation["rgb"].astype(np.float32)
    with pytest.raises(RobotComponentError, match="dtype float32, expected uint8"):
        client.act(invalid_observation, step_ids=[0])
    with pytest.raises(RobotComponentError, match="violates bounds"):
        client.act(_observation(), step_ids=[0])
