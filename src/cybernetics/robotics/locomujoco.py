"""Inert LocoMuJoCo/MuJoCo RobotEnv adapter skeleton."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .contracts import RobotTaskSpec
from .env import StepResult


class LocoMuJoCoRobotEnv:
    """Protocol-shaped adapter placeholder with no runtime dependency.

    This class lets callers and tests bind to the RobotEnv surface before the
    real LocoMuJoCo/MuJoCo runtime is installed or configured.
    """

    backend_id = "locomujoco"

    def __init__(self, task_spec: Optional[RobotTaskSpec] = None, **backend_config: Any) -> None:
        self.task_spec = task_spec
        self.backend_config = dict(backend_config)
        self.closed = False

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        raise self._not_implemented("reset")

    def step(self, action: Mapping[str, Any]) -> StepResult:
        raise self._not_implemented("step")

    def render(self, mode: str = "rgb_array") -> Any:
        raise self._not_implemented("render")

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise self._not_implemented("capture")

    def get_state(self) -> Mapping[str, Any]:
        raise self._not_implemented("get_state")

    def set_state(self, state: Mapping[str, Any]) -> None:
        raise self._not_implemented("set_state")

    def close(self) -> None:
        self.closed = True

    def _not_implemented(self, method: str) -> NotImplementedError:
        return NotImplementedError(
            f"LocoMuJoCoRobotEnv.{method} is an inert skeleton; install and wire "
            "the real LocoMuJoCo/MuJoCo adapter in a follow-up goal."
        )
