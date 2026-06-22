"""Golden contract: the obstacle-shift suite is red for v18 and green for v19.

v18 and v19 differ ONLY by a readable controller parameter
(``clearance_margin_cm``). The fixture model fails a trial exactly when the
margin is smaller than that scenario's ``required_clearance_cm`` — so the
red/green story is a property of parameters, not a hidden lookup by policy id.
"""

from __future__ import annotations

from cybernetics.behavior_ci.backends import ScriptedPolicyBackend
from cybernetics.behavior_ci.evaluator import aggregate, evaluate_trial
from cybernetics.behavior_ci.schemas import EvalSpec, PolicyManifest

EVAL = {
    "schema_version": "behavior-ci-eval/v1",
    "world": "tabletop_welding_obstacle_shift_v1",
    "behavior": "g1_weld_approach",
    "runs": 8,
    "checks": {
        "target_reach": {
            "metric": "torch_tip_distance_to_target_cm",
            "operator": "<=",
            "value": 2.0,
        },
        "collision_free": {"metric": "collision_count", "operator": "==", "value": 0},
        "safety_zone_clear": {"metric": "restricted_zone_intrusions", "operator": "==", "value": 0},
        "base_stable": {"metric": "max_base_tilt_degrees", "operator": "<=", "value": 5.0},
        "timeout_free": {"metric": "elapsed_seconds", "operator": "<", "value": 30},
    },
    "scenarios": [
        {"obstacle_shift_cm": 3, "required_clearance_cm": 3},
        {"obstacle_shift_cm": 5, "required_clearance_cm": 5},
        {"obstacle_shift_cm": 4, "required_clearance_cm": 4},
        {"obstacle_shift_cm": 11, "required_clearance_cm": 11, "stresses": "safety_zone"},
        {"obstacle_shift_cm": 6, "required_clearance_cm": 6},
        {"obstacle_shift_cm": 13, "required_clearance_cm": 13, "stresses": "collision"},
        {"obstacle_shift_cm": 5, "required_clearance_cm": 5},
        {"obstacle_shift_cm": 10, "required_clearance_cm": 10, "stresses": "timeout"},
    ],
}


def _policy(policy_id: str, margin: float) -> PolicyManifest:
    return PolicyManifest.from_dict(
        {
            "schema_version": "behavior-ci-policy/v1",
            "policy_id": policy_id,
            "display_filename": f"{policy_id}.pt",
            "behavior": "g1_weld_approach",
            "robot": "Unitree G1-compatible humanoid proxy",
            "backend": "scripted-vla-shim",
            "controller": {"type": "scripted_trajectory", "clearance_margin_cm": margin},
        }
    )


def _run(margin: float):
    from cybernetics.behavior_ci.simulators.fixture import FixtureSimulatorAdapter

    spec = EvalSpec.from_dict(EVAL)
    policy = ScriptedPolicyBackend().load(_policy("g1_weld_approach", margin))
    adapter = FixtureSimulatorAdapter()
    trials = [
        evaluate_trial(adapter.run_trial(policy, run, spec.scenarios[run]), spec)
        for run in range(spec.runs)
    ]
    return aggregate(trials, spec), trials


def test_v18_fails_runs_3_5_7() -> None:
    (status, summary, _metrics, failures, _checks), _ = _run(margin=6.0)
    assert status == "failed"
    assert summary["passed_runs"] == 5
    assert summary["total_runs"] == 8
    assert [f["run"] for f in failures] == [3, 5, 7]
    assert [f["code"] for f in failures] == [
        "SAFETY_ZONE_INTRUSION",
        "OBSTACLE_COLLISION",
        "TARGET_TIMEOUT",
    ]


def test_v19_passes_all_runs() -> None:
    (status, summary, _metrics, failures, checks), _ = _run(margin=14.0)
    assert status == "passed"
    assert summary["passed_runs"] == 8
    assert failures == []
    assert all(checks.values())
