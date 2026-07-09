"""Dependency-light Unitree G1 RobotTask provider template.

This module contains provider data only. It does not import Unitree SDK, ROS2,
Isaac, MuJoCo, MCP, or existing G1 runtime code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..contracts import ROBOT_TASK_SCHEMA_VERSION, SIMULATOR_BACKENDS, RobotContractError

ROBOT_ID = "unitree_g1"

LEG_LEAVES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM_LEAVES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")

CANONICAL_BODY_JOINTS = (
    *(f"left_{leaf}" for leaf in LEG_LEAVES),
    *(f"right_{leaf}" for leaf in LEG_LEAVES),
    *WAIST_JOINTS,
    *(f"left_{leaf}" for leaf in ARM_LEAVES),
    *(f"right_{leaf}" for leaf in ARM_LEAVES),
)

ISAAC_NEKO_PROBE_DOF_ORDER = (
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "waist_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "waist_pitch",
    "left_knee",
    "right_knee",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
    *(f"dex3_hand_dof_{index}" for index in range(14)),
)


@dataclass(frozen=True)
class JointLimit:
    effort: float
    velocity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


LEG_LIMITS = {
    "hip_pitch": JointLimit(effort=88.0, velocity=32.0),
    "hip_roll": JointLimit(effort=139.0, velocity=32.0),
    "hip_yaw": JointLimit(effort=88.0, velocity=32.0),
    "knee": JointLimit(effort=139.0, velocity=20.0),
    "ankle_pitch": JointLimit(effort=50.0, velocity=37.0),
    "ankle_roll": JointLimit(effort=50.0, velocity=37.0),
}
ARM_LIMITS = {
    "shoulder_pitch": JointLimit(effort=25.0, velocity=37.0),
    "shoulder_roll": JointLimit(effort=25.0, velocity=37.0),
    "shoulder_yaw": JointLimit(effort=25.0, velocity=37.0),
    "elbow": JointLimit(effort=25.0, velocity=37.0),
    "wrist_roll": JointLimit(effort=25.0, velocity=37.0),
    "wrist_pitch": JointLimit(effort=5.0, velocity=22.0),
    "wrist_yaw": JointLimit(effort=5.0, velocity=22.0),
}
WAIST_LIMITS = {
    "waist_yaw": JointLimit(effort=88.0, velocity=32.0),
    "waist_roll": JointLimit(effort=50.0, velocity=37.0),
    "waist_pitch": JointLimit(effort=50.0, velocity=37.0),
}

CONFIRMED_PARALLEL_LINKAGE_JOINTS = (
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_pitch",
    "right_ankle_roll",
)
INFERRED_PARALLEL_LINKAGE_JOINTS = ("waist_roll", "waist_pitch", "waist_yaw")


def build_unitree_g1_task_spec(
    *,
    task_id: str = "unitree_g1_locomotion_template",
    simulator_backend: str = "isaac_neko",
    sim_dt: float = 0.005,
    control_dt: float = 0.02,
) -> dict[str, Any]:
    """Return a valid RobotTaskSpec dictionary for a generic G1 locomotion task."""

    if simulator_backend not in SIMULATOR_BACKENDS:
        raise RobotContractError(
            f"unitree_g1 simulator_backend must be one of {list(SIMULATOR_BACKENDS)}"
        )
    return {
        "schema_version": ROBOT_TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "robot_id": ROBOT_ID,
        "simulator_backend": simulator_backend,
        "backend_config": {
            "headless": True,
            "requires_gpu": simulator_backend in {"isaaclab", "isaac_neko"},
        },
        "asset_refs": [
            {
                "kind": "provider_template",
                "uri": "cybernetics://robot-providers/unitree_g1",
                "robot_id": ROBOT_ID,
            },
            {
                "kind": "isaac_usd",
                "uri": "omniverse://Isaac/Robots/Unitree/G1/g1.usd",
                "variant": "29_body_plus_14_dex3_hands",
            },
        ],
        "joint_map": {joint: joint for joint in CANONICAL_BODY_JOINTS},
        "actuator_model": unitree_g1_actuator_model(),
        "observation_space": {
            "base_angular_velocity": {"dtype": "float32", "shape": [3]},
            "projected_gravity": {"dtype": "float32", "shape": [3]},
            "joint_positions": {"dtype": "float32", "shape": [len(CANONICAL_BODY_JOINTS)]},
            "joint_velocities": {"dtype": "float32", "shape": [len(CANONICAL_BODY_JOINTS)]},
        },
        "action_space": {
            "joint_position_targets": {
                "dtype": "float32",
                "shape": [len(CANONICAL_BODY_JOINTS)],
            }
        },
        "sim_dt": sim_dt,
        "control_dt": control_dt,
        "reset_spec": {"default_pose": "stand"},
        "reward_spec": {"kind": "locomotion_template", "review_required": True},
        "success_metric": {"metric": "distance_without_fall", "review_required": True},
        "randomization": {},
        "termination": {"fall": True, "timeout_s": 20.0, "unsafe_joint_limit": True},
        "eval_protocol": {"episodes": 1, "max_steps": 1000},
    }


def unitree_g1_actuator_model() -> dict[str, Any]:
    return {
        "kind": "pd_position_with_feedforward",
        "units": "si_radian",
        "source": "g1_usd_urdf_reconciled_provider_template",
        "joints": {name: _limit_for_joint(name).to_dict() for name in CANONICAL_BODY_JOINTS},
        "parallel_linkages": [
            {
                "joint_names": list(CONFIRMED_PARALLEL_LINKAGE_JOINTS),
                "status": "confirmed_ankle_parallel_linkage",
                "motor_space_clamp_required": True,
            },
            {
                "joint_names": list(INFERRED_PARALLEL_LINKAGE_JOINTS),
                "status": "inferred_requires_runtime_confirmation",
                "motor_space_clamp_required": True,
            },
        ],
        "caveats": [
            "Ankle pitch/roll serial-space effort clamps are not final hardware-faithful "
            "limits; motor-space A/B linkage clamps are still required.",
            "Waist roll/pitch/yaw parallel linkage status is inferred and requires "
            "runtime confirmation before motor-space clamp enforcement.",
        ],
    }


def resolve_isaac_neko_joint_indices(
    joint_order: tuple[str, ...] = ISAAC_NEKO_PROBE_DOF_ORDER,
) -> dict[str, Any]:
    """Resolve canonical G1 joints by name against an Isaac articulation order."""

    index_by_name = {name: index for index, name in enumerate(joint_order)}
    missing = [joint for joint in CANONICAL_BODY_JOINTS if joint not in index_by_name]
    if missing:
        raise RobotContractError(f"unitree_g1 missing backend joints: {missing}")
    ignored = [name for name in joint_order if name not in CANONICAL_BODY_JOINTS]
    return {
        "joint_indices": {joint: index_by_name[joint] for joint in CANONICAL_BODY_JOINTS},
        "ignored_backend_joints": ignored,
    }


def unitree_g1_transport_template(kind: str = "ros2") -> dict[str, Any]:
    if kind not in {"ros2", "dds"}:
        raise RobotContractError("unitree_g1 transport kind must be 'ros2' or 'dds'")
    return {
        "kind": kind,
        "optional": True,
        "domain_id": 0,
        "isolation": "per-session",
        "ros_distro": "jazzy",
        "dds_vendor": "cyclonedds",
        "topics": {
            "lowcmd": "rt/lowcmd",
            "lowstate": "rt/lowstate",
        },
        "notes": "TransportSpec lane only; not required by the base RobotTaskSpec.",
    }


def _limit_for_joint(name: str) -> JointLimit:
    leaf = _leaf(name)
    if name in WAIST_LIMITS:
        return WAIST_LIMITS[name]
    if leaf in LEG_LIMITS:
        return LEG_LIMITS[leaf]
    if leaf in ARM_LIMITS:
        return ARM_LIMITS[leaf]
    raise RobotContractError(f"unitree_g1 missing limit for joint {name!r}")


def _leaf(name: str) -> str:
    if name.startswith("left_"):
        return name[len("left_") :]
    if name.startswith("right_"):
        return name[len("right_") :]
    return name


__all__ = [
    "CANONICAL_BODY_JOINTS",
    "CONFIRMED_PARALLEL_LINKAGE_JOINTS",
    "INFERRED_PARALLEL_LINKAGE_JOINTS",
    "ISAAC_NEKO_PROBE_DOF_ORDER",
    "ROBOT_ID",
    "build_unitree_g1_task_spec",
    "resolve_isaac_neko_joint_indices",
    "unitree_g1_actuator_model",
    "unitree_g1_transport_template",
]
