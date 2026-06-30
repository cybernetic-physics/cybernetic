"""Deterministic, no-network simulator adapter.

This is the fast path for local dev, unit tests, and the default public CI job.
It is NOT Isaac. It computes trial metrics as a transparent function of the
per-run scenario (``required_clearance_cm``, ``stresses``) and the policy's
readable controller parameters (``clearance_margin_cm``). A trial fails its
stressed check exactly when the controller's clearance margin is smaller than
the scenario requires — so behavior regressions are a property of parameters,
not a hidden lookup by ``policy_id``.

Replay clips in this mode come from checked-in demo evidence (real Isaac session
captures committed to the repo). A minimal placeholder is only produced when
explicitly allowed, and is always labelled ``fixture-generated`` in provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backends.base import LoadedPolicy
from ..schemas import ConfigError, Event, TrialObservation
from .base import ReplayResult, SceneSpec, looks_like_mp4

# Deterministic nominal (passing) per-run baselines; indexed by run % 8.
_BASE_DIST = [1.3, 1.5, 1.6, 1.4, 1.7, 1.2, 1.8, 1.4]
_BASE_ELAPSED = [22.1, 23.4, 21.9, 24.0, 22.8, 23.1, 24.4, 22.6]
_BASE_TILT = [1.4, 1.8, 2.0, 1.6, 2.2, 1.7, 2.4, 1.5]

# How each stress mode manifests when the controller fails to clear the scenario.
_STRESS_OUTCOMES = {
    "safety_zone": {
        "metrics": {"restricted_zone_intrusions": 1, "torch_tip_distance_to_target_cm": 2.4},
        "event": ("SAFETY_ZONE_INTRUSION", 13.8, "torch path entered the red restricted zone"),
    },
    "collision": {
        "metrics": {"collision_count": 2, "torch_tip_distance_to_target_cm": 4.6},
        "event": ("OBSTACLE_COLLISION", 12.4, "end-effector collided with the shifted obstacle"),
    },
    "timeout": {
        "metrics": {"elapsed_seconds": 34.8, "torch_tip_distance_to_target_cm": 3.8},
        "event": ("TARGET_TIMEOUT", 30.0, "failed to reach the weld start pose before timeout"),
    },
}


class FixtureSimulatorAdapter:
    adapter_id = "fixture"

    def __init__(
        self, replay_dir: Optional[Path] = None, allow_placeholder_replays: bool = True
    ) -> None:
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self.allow_placeholder_replays = allow_placeholder_replays
        self.session_id: Optional[str] = None
        self.replay_source = "none"
        # Set by the runner for a pinned task: the platform-owned action/measure contract.
        self.task = None

    def __enter__(self) -> "FixtureSimulatorAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def prepare(self, scene: SceneSpec) -> None:
        # No-op: the fixture model has no scene to instantiate.
        return None

    def run_action_trial(
        self,
        action: Dict[str, Any],
        run: int,
        observation: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> TrialObservation:
        """Pinned-task path: the policy emitted ``action`` (a trajectory); the task's pure,
        independent ``measure`` derives the metrics from the geometry. No policy-supplied
        number is read as ground truth, and the same ``measure`` runs in the hosted grader,
        so the offline and hosted verdicts cannot drift."""
        if self.task is None:  # pragma: no cover - guarded by the runner
            raise ConfigError("run_action_trial requires a pinned task")
        metrics = self.task.measure(action, observation)
        return TrialObservation(
            run=run,
            metrics={k: v for k, v in metrics.items()},
            events=[],
            trajectory_id=f"g1-run{run:02d}",
        )

    def run_trial(
        self, policy: LoadedPolicy, run: int, scenario: Dict[str, Any]
    ) -> TrialObservation:
        idx = run % len(_BASE_DIST)
        metrics: Dict[str, float] = {
            "torch_tip_distance_to_target_cm": _BASE_DIST[idx],
            "collision_count": 0,
            "restricted_zone_intrusions": 0,
            "max_base_tilt_degrees": _BASE_TILT[idx],
            "elapsed_seconds": _BASE_ELAPSED[idx],
            "obstacle_shift_cm": float(scenario.get("obstacle_shift_cm", 0.0)),
        }
        events: List[Event] = []

        required = float(scenario.get("required_clearance_cm", 0.0))
        margin = float(policy.param("clearance_margin_cm", 0.0))
        stresses = scenario.get("stresses")
        handled = margin >= required

        if not handled and stresses in _STRESS_OUTCOMES:
            outcome = _STRESS_OUTCOMES[stresses]
            metrics.update(outcome["metrics"])  # type: ignore[arg-type]
            code, t, msg = outcome["event"]  # type: ignore[misc]
            events.append(Event(run=run, time_seconds=t, code=code, message=msg))

        return TrialObservation(
            run=run,
            metrics=metrics,
            events=events,
            trajectory_id=f"{policy.policy_id}-run{run:02d}",
        )

    def capture_replays(
        self, scene: SceneSpec, failed_run: Optional[int], passed_run: Optional[int]
    ) -> List[ReplayResult]:
        wanted: List[str] = []
        if failed_run is not None:
            wanted.append("replay-failed")
        if passed_run is not None:
            wanted.append("replay-passed")
        if not wanted:
            wanted.append("replay-passed")

        replays: List[ReplayResult] = []
        for name in wanted:
            replays.append(self._resolve(name, scene.camera))
        # Provenance is the weakest source across produced clips.
        self.replay_source = (
            "fixture-generated"
            if any(r.source == "fixture-generated" for r in replays)
            else "checked-in-demo-evidence"
        )
        return replays

    def _resolve(self, name: str, camera: str) -> ReplayResult:
        if self.replay_dir is not None:
            candidate = self.replay_dir / f"{name}.mp4"
            if candidate.exists():
                data = candidate.read_bytes()
                if looks_like_mp4(data):
                    return ReplayResult(
                        name=name, data=data, source="checked-in-demo-evidence", camera=camera
                    )
        if not self.allow_placeholder_replays:
            raise ConfigError(
                f"no valid checked-in replay for '{name}.mp4' in "
                f"{self.replay_dir} and placeholders are disabled"
            )
        return ReplayResult(
            name=name, data=_placeholder_mp4(name), source="fixture-generated", camera=camera
        )


def _placeholder_mp4(label: str) -> bytes:
    """A minimal, clearly-not-real MP4 stub (valid ftyp box + a text marker)."""

    ftyp = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    return ftyp + b"\n[fixture-generated placeholder replay: " + label.encode() + b"]\n"
