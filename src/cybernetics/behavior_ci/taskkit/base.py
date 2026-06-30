"""The dev-facing Behavior CI task API.

Author a task by subclassing :class:`Task` in YOUR repo and decorating it with
``@register_task("<id>")`` -- defining a new behavior task needs NO SDK release. The SDK
discovers your task from a ``tasks/<id>/`` directory, verifies its integrity, runs it, and
owns the artifact bundle, the replay GIF, and the PR comment.

Two tiers:
  * PURE tier (this class): ``scenarios``/``build_observation``/``plan``/``measure``/``checks``
    -- deterministic, stdlib-only, runs both offline (fixture) and as the authoritative grade.
    ``measure`` is authored ONCE here; the SDK injects it into the hosted grader at upload, so
    the offline and hosted verdicts cannot drift.
  * HOSTED tier (a sibling ``grader_isaac.py``, text-only): the omni.* scene build + actuation
    for the real-robot replay. It does NOT re-implement ``measure``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Lightweight typed aliases (kept as dicts so authors need no imports to construct them).
Scenario = Dict[str, Any]
Observation = Dict[str, Any]
Trajectory = Dict[str, Any]
Metrics = Dict[str, Any]


@dataclass(frozen=True)
class Check:
    """One scalar pass/fail rule over a measured metric."""

    metric: str
    operator: str  # one of == != < <= > >=
    value: Any
    required: bool = True


class Task(ABC):
    """Subclass + ``@register_task("<id>")`` to define a behavior-CI task in your repo."""

    # Scene/robot metadata (1:1 with the legacy task.yaml). Set as class attributes.
    behavior: str = ""
    robot: str = ""
    world: str = ""
    scene_env: str = ""
    camera: str = ""
    env_id: str = ""
    action_contract: str = "trajectory/v1"
    # Hosted tier pointers (the in-session Isaac grader).
    grader_module: str = "grader_isaac.py"
    grader_entrypoint: str = "behavior_ci_run_trial"

    # Bound by @register_task; never set by hand.
    task_id: str = ""

    @abstractmethod
    def scenarios(self) -> Tuple[List[Scenario], List[Scenario]]:
        """Return ``(visible, held_out)``.

        ``visible`` is the published suite; ``held_out`` is the anti-overfit perturbation bank.
        In-repo ``held_out`` is advisory/preview -- the binding bank is the canonical one the
        authoritative check grades against.
        """

    @abstractmethod
    def build_observation(self, scenario: Scenario) -> Observation:
        """Turn a scenario into the geometry the policy observes + the env measures."""

    @abstractmethod
    def plan(self, checkpoint: Dict[str, Any], observation: Observation) -> Trajectory:
        """Turn the policy's opaque checkpoint + an observation into an action (a trajectory)."""

    @abstractmethod
    def measure(self, trajectory: Trajectory, observation: Observation) -> Metrics:
        """Independently measure outcomes from the emitted trajectory. PURE; authored ONCE."""

    @abstractmethod
    def checks(self) -> Dict[str, Check]:
        """The pass/fail rules applied to each trial's measured metrics."""
