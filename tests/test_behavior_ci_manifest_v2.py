"""Closed-capability v2 policy manifest: the structural half of the anti-gaming boundary.

A v2 manifest carries only an opaque ``checkpoint`` + a pinned ``task``. There is no field
to hand the grader a trusted scalar and no ``session_entrypoint`` to choose the grader.
"""

import pytest

from cybernetics.behavior_ci.schemas import ConfigError, PolicyManifest


def _v2(**overrides):
    base = {
        "schema_version": "behavior-ci-policy/v2",
        "policy_id": "g1_weld_approach_v21",
        "display_filename": "g1_weld_approach_v21.pt",
        "behavior": "g1_weld_approach",
        "robot": "Unitree G1-compatible humanoid proxy",
        "backend": "scripted-vla-shim",
        "task": "g1_weld_approach",
        "checkpoint": {"detour_mode": "relative", "detour_gain": 1.0},
    }
    base.update(overrides)
    return base


def test_valid_v2_parses():
    m = PolicyManifest.from_dict(_v2())
    assert m.is_v2 and m.task == "g1_weld_approach"
    assert m.params == {"detour_mode": "relative", "detour_gain": 1.0}
    assert m.controller == {}


def test_v2_requires_task_and_checkpoint():
    d = _v2()
    d.pop("task")
    with pytest.raises(ConfigError):
        PolicyManifest.from_dict(d)
    d = _v2()
    d.pop("checkpoint")
    with pytest.raises(ConfigError):
        PolicyManifest.from_dict(d)


def test_v2_rejects_session_entrypoint_smuggle():
    """The classic capability smuggle: choosing which function grades you. Closed out."""
    with pytest.raises(ConfigError) as e:
        PolicyManifest.from_dict(_v2(session_entrypoint="my_grader"))
    assert "session_entrypoint" in str(e.value) or "unknown" in str(e.value).lower()


def test_v2_rejects_arbitrary_top_level_key():
    with pytest.raises(ConfigError):
        PolicyManifest.from_dict(_v2(secret_backdoor=1))


def test_v2_checkpoint_is_opaque_not_trusted():
    """A 'clearance' number is allowed INSIDE the checkpoint, but it is NOT ground truth:
    the planner consumes it and the environment measures the resulting trajectory. So a
    huge value does not buy a pass -- it just produces a bad (over-detoured) trajectory.
    The schema permits it precisely because it is harmless / measured, not trusted."""
    m = PolicyManifest.from_dict(_v2(checkpoint={"clearance_margin_cm": 999.0}))
    assert m.params["clearance_margin_cm"] == 999.0  # carried, never read as a verdict


def test_v1_still_parses_during_migration():
    m = PolicyManifest.from_dict(
        {
            "schema_version": "behavior-ci-policy/v1",
            "policy_id": "g1_weld_approach_v18",
            "display_filename": "g1_weld_approach_v18.pt",
            "behavior": "g1_weld_approach",
            "robot": "Unitree G1-compatible humanoid proxy",
            "backend": "scripted-vla-shim",
            "controller": {"clearance_margin_cm": 6.0},
        }
    )
    assert not m.is_v2 and m.params == {"clearance_margin_cm": 6.0}
