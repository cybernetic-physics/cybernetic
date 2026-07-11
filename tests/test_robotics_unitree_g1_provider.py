from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any, Mapping

import pytest

import cybernetics.robotics as robotics
from cybernetics.robotics import RobotTaskSpec, TransportSpec
from cybernetics.robotics.providers import unitree_g1


def test_unitree_g1_provider_emits_valid_robot_task_spec() -> None:
    spec_dict = unitree_g1.build_unitree_g1_task_spec()
    spec = RobotTaskSpec.from_dict(spec_dict)

    assert spec.robot_id == "unitree_g1"
    assert spec.simulator_backend == "isaac_neko"
    assert spec.control_dt / spec.sim_dt == 4
    assert spec.task_hash() == RobotTaskSpec.from_dict(spec.to_dict()).task_hash()
    assert any(ref["kind"] == "provider_template" for ref in spec.asset_refs)
    assert any(ref["kind"] == "isaac_usd" for ref in spec.asset_refs)
    assert "transport" not in spec.to_dict()
    assert "ros2" not in spec.to_dict()


def test_unitree_g1_provider_joint_maps_are_name_based_not_index_based() -> None:
    joints = unitree_g1.CANONICAL_BODY_JOINTS
    resolved = unitree_g1.resolve_isaac_neko_joint_indices()

    assert len(joints) == 29
    assert len(set(joints)) == 29
    assert "waist_yaw" in joints
    assert "waist_roll" in joints
    assert "waist_pitch" in joints
    assert "left_wrist_pitch" in joints
    assert "right_wrist_yaw" in joints
    assert "left_elbow" in joints
    assert not any("dex3" in joint or "hand" in joint for joint in joints)
    assert resolved["joint_indices"]["right_hip_pitch"] == 1
    assert resolved["joint_indices"]["waist_yaw"] == 2
    assert resolved["joint_indices"]["left_ankle_roll"] == 17
    assert len(resolved["ignored_backend_joints"]) == 14


def test_unitree_g1_actuator_model_uses_physical_limits_not_freefall_clamps() -> None:
    actuator_model = unitree_g1.unitree_g1_actuator_model()
    joints = actuator_model["joints"]

    assert joints["left_knee"]["effort"] == 139.0
    assert joints["left_hip_roll"]["effort"] == 139.0
    assert joints["right_wrist_pitch"]["effort"] == 5.0
    assert joints["left_knee"]["velocity"] == 20.0
    assert joints["left_hip_yaw"]["velocity"] == 32.0
    assert all(limit["effort"] != 300.0 for limit in joints.values())
    assert "robot_params.yaml" not in repr(actuator_model)


def test_unitree_g1_parallel_linkage_caveats_are_explicit() -> None:
    actuator_model = unitree_g1.unitree_g1_actuator_model()
    parallel = actuator_model["parallel_linkages"]

    confirmed = parallel[0]
    assert set(confirmed["joint_names"]) == {
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_ankle_pitch",
        "right_ankle_roll",
    }
    assert confirmed["motor_space_clamp_required"] is True

    inferred = parallel[1]
    assert inferred["status"] == "inferred_requires_runtime_confirmation"
    assert "waist_roll" in inferred["joint_names"]
    assert "motor-space A/B linkage clamps are still required" in " ".join(
        actuator_model["caveats"]
    )


def test_unitree_g1_transport_template_is_optional() -> None:
    base_spec = unitree_g1.build_unitree_g1_task_spec()
    transport = unitree_g1.unitree_g1_transport_template("ros2")

    assert RobotTaskSpec.from_dict(base_spec).robot_id == "unitree_g1"
    parsed = TransportSpec.from_dict(transport)
    assert transport["kind"] == "ros2"
    assert parsed.topics[0]["name"] == "lowcmd"
    assert parsed.topics[0]["topic"] == "rt/lowcmd"
    assert parsed.topics[0]["direction"] == "publish"
    assert parsed.topics[1]["name"] == "lowstate"
    assert parsed.topics[1]["topic"] == "rt/lowstate"
    assert parsed.topics[1]["direction"] == "subscribe"
    assert parsed.qos["vendor"] == "cyclonedds"
    assert parsed.qos["ros_distro"] == "jazzy"
    assert "transport" not in base_spec


def test_unitree_g1_provider_does_not_pollute_base_robotics_api() -> None:
    forbidden_exports = {
        "UNITREE_G1",
        "G1_JOINT_NAMES",
        "LowCmd",
        "LowState",
        "rclpy",
        "ros2",
    }

    assert forbidden_exports.isdisjoint(set(robotics.__all__))
    assert unitree_g1.ROBOT_ID == "unitree_g1"


def test_unitree_g1_provider_import_stays_dependency_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_roots = {
        "g1_brain",
        "isaac",
        "isaac_sim_mcp_extension",
        "isaacsim",
        "locomujoco",
        "mujoco",
        "omni",
        "pxr",
        "rclpy",
        "ros2",
        "unitree",
        "unitree_hg",
        "unitree_sdk2",
    }
    attempted: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if level == 0 and root in blocked_roots:
            attempted.append(name)
            raise AssertionError(f"runtime package import attempted: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("cybernetics.robotics.providers.unitree_g1", None)
    importlib.import_module("cybernetics.robotics")
    importlib.import_module("cybernetics.robotics.providers.unitree_g1")

    assert attempted == []
