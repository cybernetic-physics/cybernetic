"""Robot environment protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ObservationBundle:
    """One policy-facing observation with explicit timing and native metadata."""

    values: Mapping[str, Any]
    timestamp_seconds: Optional[float] = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": dict(self.values),
            "timestamp_seconds": self.timestamp_seconds,
            "info": dict(self.info),
        }


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


@dataclass(frozen=True)
class VectorStepResult:
    """Batched step result that preserves NumPy/Torch/native tensor values."""

    observations: Any
    rewards: Any
    terminated: Any
    truncated: Any
    info: Any
    num_envs: int

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("VectorStepResult.num_envs must be positive")


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


@runtime_checkable
class VectorRobotEnv(Protocol):
    """Tensor-preserving vector environment boundary.

    Implementations may return dictionary-of-batches or sequence-of-mappings
    observations. The adapter owns validation of batch width; the SDK does not
    coerce GPU tensors through Python lists.
    """

    num_envs: int

    def reset(
        self,
        seed: Optional[int | Sequence[int]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Any: ...

    def step(self, actions: Any) -> VectorStepResult: ...

    def render(self, mode: str = "rgb_array") -> Any: ...

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_state(self) -> Any: ...

    def set_state(self, state: Any) -> None: ...

    def close(self) -> None: ...
