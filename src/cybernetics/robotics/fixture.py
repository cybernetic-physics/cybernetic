"""Deterministic fixture RobotEnv used before simulator adapters exist."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .env import StepResult


class FixtureRobotEnv:
    """Small, deterministic env for contract and runner tests.

    The state is one scalar position. Actions carry ``delta`` and the task
    succeeds once ``position >= success_position``. It is intentionally not a
    simulator; it is a test double for the RobotEnv boundary.
    """

    def __init__(self, *, max_steps: int = 8, success_position: float = 3.0) -> None:
        self.max_steps = int(max_steps)
        self.success_position = float(success_position)
        self._closed = False
        self._seed: Optional[int] = None
        self._position = 0.0
        self._step_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        self._closed = False
        self._seed = seed
        opts = dict(options or {})
        self._position = float(opts.get("position", 0.0))
        self._step_count = 0
        return self._observation()

    def step(self, action: Mapping[str, Any]) -> StepResult:
        if self._closed:
            raise RuntimeError("fixture robot env is closed")
        self._step_count += 1
        self._position += float(action.get("delta", 0.0))
        terminated = self._position >= self.success_position
        truncated = self._step_count >= self.max_steps and not terminated
        reward = self._position
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"step_count": self._step_count, "seed": self._seed},
        )

    def render(self, mode: str = "rgb_array") -> Mapping[str, Any]:
        return {"mode": mode, "state": self.get_state()}

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"kind": "fixture_capture", "request": dict(request), "state": self.get_state()}

    def get_state(self) -> Mapping[str, Any]:
        return {
            "position": self._position,
            "step_count": self._step_count,
            "seed": self._seed,
            "closed": self._closed,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self._position = float(state.get("position", 0.0))
        self._step_count = int(state.get("step_count", 0))
        self._seed = state.get("seed")  # type: ignore[assignment]
        self._closed = bool(state.get("closed", False))

    def close(self) -> None:
        self._closed = True

    def _observation(self) -> Mapping[str, Any]:
        return {"position": self._position, "step_count": self._step_count}
