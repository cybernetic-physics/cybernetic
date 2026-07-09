"""Dependency-light RobotTask SDK skeleton.

This package owns the public robot-task contracts before any simulator backend
does. It intentionally imports only stdlib modules so ``import
cybernetics.robotics`` works without MuJoCo, LocoMuJoCo, Isaac, ROS2, or
Worldlines installed.
"""

from .artifacts import write_json_artifact, write_policy_artifact, write_robot_run_record
from .contracts import (
    ROBOT_DATASET_SCHEMA_VERSION,
    ROBOT_POLICY_SCHEMA_VERSION,
    ROBOT_RUN_SCHEMA_VERSION,
    ROBOT_TASK_SCHEMA_VERSION,
    WORLD_MODEL_SCHEMA_VERSION,
    PolicyArtifact,
    RobotContractError,
    RobotRunRecord,
    RobotTaskSpec,
    TrajectoryDatasetArtifact,
    WorldModelArtifact,
    stable_hash,
)
from .env import RobotEnv, StepResult
from .fixture import FixtureRobotEnv
from .runner import default_action, deterministic_run_id, run_robot_episode

__all__ = [
    "ROBOT_DATASET_SCHEMA_VERSION",
    "ROBOT_POLICY_SCHEMA_VERSION",
    "ROBOT_RUN_SCHEMA_VERSION",
    "ROBOT_TASK_SCHEMA_VERSION",
    "WORLD_MODEL_SCHEMA_VERSION",
    "FixtureRobotEnv",
    "PolicyArtifact",
    "RobotContractError",
    "RobotEnv",
    "RobotRunRecord",
    "RobotTaskSpec",
    "StepResult",
    "TrajectoryDatasetArtifact",
    "WorldModelArtifact",
    "default_action",
    "deterministic_run_id",
    "run_robot_episode",
    "stable_hash",
    "write_json_artifact",
    "write_policy_artifact",
    "write_robot_run_record",
]
