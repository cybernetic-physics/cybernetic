"""Robot environment protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": dict(self.observation),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": dict(self.info),
        }


@runtime_checkable
class RobotEnv(Protocol):
    """Gymnasium-shaped robot environment boundary."""

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]: ...

    def step(self, action: Mapping[str, Any]) -> StepResult: ...

    def render(self, mode: str = "rgb_array") -> Any: ...

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_state(self) -> Mapping[str, Any]: ...

    def set_state(self, state: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...
