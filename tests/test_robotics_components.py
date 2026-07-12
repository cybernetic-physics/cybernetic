from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Mapping

import pytest
from test_robotics_runtime_contracts import job_dict

from cybernetics.robotics import (
    RobotComponentError,
    RoboticsJobSpec,
    StepResult,
    open_simulator_component,
    validate_simulator_descriptor,
)


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
