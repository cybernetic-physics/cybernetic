"""Task-pack integrity + anti-gaming core, exercised THROUGH the installed SDK package.

These assert that the g1_weld_approach pack loaded via ``load_task`` reproduces the proven
behavior: the golden v18-style regression fails exactly visible runs 3/5/7 with matching
failure kinds, a well-tuned obstacle-relative policy passes the whole family, and -- the
anti-gaming guarantee -- NO geometry-blind constant apex can pass every visible + held-out
scenario, while the detour axis is non-monotone and torch speed is clamped.
"""

from __future__ import annotations

import hashlib

from cybernetics.behavior_ci.tasks import load_task, sha256_text


def _checks(m):
    return {
        "target_reach": m["torch_tip_distance_to_target_cm"] <= 2.0,
        "collision_free": m["collision_count"] == 0,
        "safety_zone_clear": m["restricted_zone_intrusions"] == 0,
        "base_stable": m["max_base_tilt_degrees"] <= 5.0,
        "timeout_free": m["elapsed_seconds"] < 30.0,
    }


def _failed(task, ck, scenario):
    obs = task.build_observation(scenario)
    m = task.measure(task.plan(ck, obs), obs)
    return [k for k, v in _checks(m).items() if not v]


def _pass_count(task, ck, scen_list):
    return sum(1 for s in scen_list if not _failed(task, ck, s))


CK_GOOD = {
    "detour_mode": "relative",
    "detour_gain": 1.0,
    "clearance_margin_cm": 12.0,
    "top_halfwidth_cm": 30.0,
    "approach_speed_mps": 0.12,
}
CK_UNDER = {
    "detour_mode": "relative",
    "detour_gain": 0.85,
    "clearance_margin_cm": 6.5,
    "top_halfwidth_cm": 30.0,
    "approach_speed_mps": 0.075,
}


def test_task_loads():
    t = load_task("g1_weld_approach")
    assert t.behavior == "g1_weld_approach"
    assert t.env_id == "env_7d904291a384a1ae"
    assert t.grader_entrypoint == "behavior_ci_run_trial"
    assert len(t.visible) == 8 and len(t.held_out) == 8


def test_eval_scenarios_match_module():
    """eval.yaml visible/held_out must match the geometry module's lists (no drift)."""
    t = load_task("g1_weld_approach")
    from cybernetics.behavior_ci.tasks.g1_weld_approach import scenarios as S

    assert t.visible == S.VISIBLE
    assert t.held_out == S.HELD_OUT


def test_golden_under_fails_exactly_3_5_7_with_matching_kinds():
    t = load_task("g1_weld_approach")
    fails = [i for i, s in enumerate(t.visible) if _failed(t, CK_UNDER, s)]
    assert fails == [3, 5, 7]
    assert "safety_zone_clear" in _failed(t, CK_UNDER, t.visible[3])
    assert "collision_free" in _failed(t, CK_UNDER, t.visible[5])
    assert "timeout_free" in _failed(t, CK_UNDER, t.visible[7])


def test_good_policy_passes_visible_and_held_out():
    t = load_task("g1_weld_approach")
    assert _pass_count(t, CK_GOOD, t.visible) == 8
    assert _pass_count(t, CK_GOOD, t.held_out) == 8


def test_no_constant_apex_passes_the_family():
    """The anti-gaming guarantee: no geometry-blind fixed apex clears visible+held-out."""
    t = load_task("g1_weld_approach")
    allscen = t.visible + t.held_out
    best = 0
    for apex10 in range(0, 1601):  # apex 0..160 in 0.1 steps
        ck = {
            "detour_mode": "absolute",
            "absolute_apex_cm": apex10 / 10.0,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": 2.0,
        }
        pc = _pass_count(t, ck, allscen)
        if pc > best:
            best = pc
    assert best < len(allscen), f"a constant apex passed all {len(allscen)} scenarios"


def test_detour_axis_is_non_monotone():
    """No monotone safety knob: pass-count rises then falls as the margin grows."""
    t = load_task("g1_weld_approach")
    allscen = t.visible + t.held_out
    counts = []
    for m10 in range(0, 1401, 10):
        ck = {
            "detour_mode": "relative",
            "detour_gain": 1.0,
            "clearance_margin_cm": m10 / 10.0,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": 0.12,
        }
        counts.append(_pass_count(t, ck, allscen))
    peak = max(counts)
    assert counts[-1] < peak  # large margin degrades -> single-peaked, not monotone


def test_speed_is_clamped():
    t = load_task("g1_weld_approach")
    obs = t.build_observation(t.visible[0])
    traj = t.plan({"detour_mode": "relative", "approach_speed_mps": 1e9}, obs)
    from cybernetics.behavior_ci.tasks.g1_weld_approach import planner

    assert traj["speed_mps"] <= planner.MAX_SPEED_MPS + 1e-9


def test_hosted_grader_measure_matches_offline_fixture():
    """No offline/hosted drift: the grader's embedded measure() must produce identical
    metrics to the offline measure.py for every scenario + a range of checkpoints."""
    from cybernetics.behavior_ci.tasks.g1_weld_approach import grader, measure

    t = load_task("g1_weld_approach")
    checkpoints = [
        {
            "detour_mode": "relative",
            "detour_gain": g,
            "clearance_margin_cm": m,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": s,
        }
        for g in (0.0, 0.85, 1.0)
        for m in (0.0, 6.5, 12.0)
        for s in (0.075, 0.12)
    ] + [
        {
            "detour_mode": "absolute",
            "absolute_apex_cm": a,
            "top_halfwidth_cm": 30.0,
            "approach_speed_mps": 0.12,
        }
        for a in (40.0, 55.0, 69.0)
    ]
    for scenario in t.visible + t.held_out:
        obs = t.build_observation(scenario)
        for ck in checkpoints:
            traj = t.plan(ck, obs)
            assert grader.measure(traj, obs) == measure.measure(traj, obs)


def test_lock_digests_match_pack_files():
    """The integrity lock must match the actual pinned bytes."""
    import importlib.resources as ir

    t = load_task("g1_weld_approach")
    assert t.lock is not None
    base = ir.files("cybernetics.behavior_ci.tasks.g1_weld_approach")
    for name, digest in t.lock.digests.items():
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} digest drift"
