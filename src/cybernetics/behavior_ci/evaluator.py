"""Pure metric -> check -> pass/fail evaluation.

No network, no simulator, no IO. Takes raw :class:`TrialObservation`s plus an
:class:`EvalSpec` and produces per-trial and aggregate verdicts. This is the
deterministic core both the fixture and hosted adapters feed into, and it is the
unit under the golden v18-fails / v19-passes contract test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .schemas import OPS, CheckResult, EvalSpec, Event, TrialObservation, TrialResult

# Map a failed required check to a human PR-comment failure code.
_CHECK_FAILURE_CODES = {
    "safety_zone_clear": "SAFETY_ZONE_INTRUSION",
    "collision_free": "OBSTACLE_COLLISION",
    "timeout_free": "TARGET_TIMEOUT",
    "target_reach": "TARGET_MISS",
    "base_stable": "BASE_INSTABILITY",
}


def evaluate_trial(obs: TrialObservation, spec: EvalSpec) -> TrialResult:
    """Apply every check in ``spec`` to one trial's metrics."""

    checks: Dict[str, CheckResult] = {}
    all_required_passed = True
    for name, check in spec.checks.items():
        if check.metric not in obs.metrics:
            raise KeyError(
                f"trial {obs.run}: check '{name}' needs metric '{check.metric}' "
                f"which the observation did not produce"
            )
        actual = obs.metrics[check.metric]
        passed = bool(OPS[check.operator](actual, check.value))
        checks[name] = CheckResult(
            passed=passed,
            metric=check.metric,
            actual=actual,
            operator=check.operator,
            expected=check.value,
            required=check.required,
        )
        if check.required and not passed:
            all_required_passed = False

    events = list(obs.events)
    # If a required check failed but the rollout produced no explanatory event,
    # synthesize one so the PR comment always names a failure code.
    if not all_required_passed and not events:
        for name, result in checks.items():
            if result.required and not result.passed:
                events.append(
                    Event(
                        run=obs.run,
                        time_seconds=0.0,
                        code=_CHECK_FAILURE_CODES.get(name, "CHECK_FAILURE"),
                        message=(
                            f"{name}: {result.metric}={result.actual} "
                            f"failed {result.operator} {result.expected}"
                        ),
                    )
                )

    return TrialResult(
        run=obs.run,
        passed=all_required_passed,
        checks=checks,
        metrics=dict(obs.metrics),
        events=events,
        trajectory_id=obs.trajectory_id,
    )


def aggregate(
    trials: List[TrialResult], spec: EvalSpec
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, bool]]:
    """Roll per-trial results up into (status, summary, metrics, failures, checks).

    Status is ``passed`` iff every trial passed all its required checks.
    """

    total = len(trials)
    passed_runs = sum(1 for t in trials if t.passed)
    status = "passed" if passed_runs == total and total > 0 else "failed"

    failures: List[Dict[str, Any]] = []
    for t in trials:
        for ev in t.events:
            failures.append({"run": ev.run, "code": ev.code, "message": ev.message})

    # Aggregate check pass: a check is green overall iff it passed in every trial.
    check_pass: Dict[str, bool] = {}
    for name in spec.checks:
        check_pass[name] = all(t.checks[name].passed for t in trials) if trials else False

    summary = {
        "passed_runs": passed_runs,
        "total_runs": total,
        "failed_runs": total - passed_runs,
        "world": spec.world,
        "behavior": spec.behavior,
    }

    metrics = _aggregate_metrics(trials)
    metrics["task_success"] = f"{passed_runs} / {total}"
    return status, summary, metrics, failures, check_pass


def _aggregate_metrics(trials: List[TrialResult]) -> Dict[str, Any]:
    """Summarize the standard welding-behavior metrics across trials."""

    if not trials:
        return {}

    def _vals(metric: str) -> List[float]:
        return [float(t.metrics[metric]) for t in trials if metric in t.metrics]

    def _mean(metric: str) -> float:
        vals = _vals(metric)
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _sum(metric: str) -> float:
        return round(sum(_vals(metric)), 2)

    def _max(metric: str) -> float:
        vals = _vals(metric)
        return round(max(vals), 2) if vals else 0.0

    return {
        "mean_torch_tip_error_cm": _mean("torch_tip_distance_to_target_cm"),
        "collision_events": int(_sum("collision_count")),
        "safety_zone_violations": int(_sum("restricted_zone_intrusions")),
        "max_base_tilt_degrees": _max("max_base_tilt_degrees"),
        "mean_trial_seconds": _mean("elapsed_seconds"),
    }
