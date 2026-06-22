"""Cybernetic Physics Behavior CI.

Run a pinned simulation eval against a robot policy and emit a red/green verdict,
metrics, and replay evidence as a stable ``behavior-ci/v1`` artifact bundle —
"CodeRabbit reviews whether the code looks right; Cybernetic Physics reviews
whether the robot still works."

    from cybernetics.behavior_ci import BehaviorCiRunner

    runner = BehaviorCiRunner.from_config("cybernetic-behavior-ci.yaml")
    result = runner.run_policy(
        policy_ref="policies/g1_weld_approach_v19.pt",
        eval_ref="obstacle_shift",
        out_dir="artifacts/behavior-ci",
    )
    assert result.passed

This package is imported lazily by the ``cybernetics behavior-ci`` CLI; it is not
part of the core ``import cybernetics`` surface and pulls no heavy dependencies.
"""

from .backends import LoadedPolicy, PolicyBackend, ScriptedPolicyBackend, select_backend
from .runner import BehaviorCiRunner
from .schemas import (
    BehaviorCiConfig,
    BehaviorCiError,
    BehaviorCiResult,
    Check,
    ConfigError,
    ContractError,
    EvalSpec,
    HonestyProvenance,
    PolicyManifest,
    TaskSpec,
    TrialResult,
)
from .simulators import (
    FixtureSimulatorAdapter,
    IsaacSessionAdapter,
    SceneSpec,
    SimulatorAdapter,
)

__all__ = [
    "BehaviorCiRunner",
    "BehaviorCiConfig",
    "BehaviorCiResult",
    "PolicyManifest",
    "EvalSpec",
    "TaskSpec",
    "Check",
    "TrialResult",
    "HonestyProvenance",
    "SceneSpec",
    "SimulatorAdapter",
    "FixtureSimulatorAdapter",
    "IsaacSessionAdapter",
    "PolicyBackend",
    "ScriptedPolicyBackend",
    "LoadedPolicy",
    "select_backend",
    "BehaviorCiError",
    "ConfigError",
    "ContractError",
]
