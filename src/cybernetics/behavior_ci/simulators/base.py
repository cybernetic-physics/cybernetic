"""Simulator adapter protocol + shared scene/replay types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..backends.base import LoadedPolicy
from ..schemas import TrialObservation

# Minimum bytes + the ISO-BMFF ``ftyp`` box that marks a real MP4 container.
_MP4_MIN_BYTES = 32


def looks_like_mp4(data: bytes) -> bool:
    """True if ``data`` starts like a real MP4 (``ftyp`` box), not a placeholder."""

    return len(data) >= _MP4_MIN_BYTES and b"ftyp" in data[:32]


@dataclass(frozen=True)
class SceneSpec:
    """What the adapter must instantiate or load before running trials."""

    world: str
    scene_env: str
    camera: str
    robot: str
    env_id: Optional[str] = None


@dataclass
class ReplayResult:
    """A captured replay clip + its honest provenance."""

    name: str  # e.g. "replay-failed" / "replay-passed"
    data: bytes
    source: str  # one of schemas.REPLAY_SOURCES
    camera: str
    content_type: str = "video/mp4"


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Runs trials for a loaded policy and captures pass/fail replays.

    Implementations are context managers so hosted resources (sessions) are
    always released, even on failure.
    """

    adapter_id: str
    replay_source: str
    session_id: Optional[str]

    def __enter__(self) -> "SimulatorAdapter": ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

    def prepare(self, scene: SceneSpec) -> None: ...

    def run_trial(
        self, policy: LoadedPolicy, run: int, scenario: Dict[str, Any]
    ) -> TrialObservation: ...

    def capture_replays(
        self,
        scene: SceneSpec,
        failed_run: Optional[int],
        passed_run: Optional[int],
        replay_inputs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[ReplayResult]: ...
