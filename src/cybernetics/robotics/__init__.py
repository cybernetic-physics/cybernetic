"""Dependency-light RobotTask SDK skeleton.

This package owns the public robot-task contracts before any simulator backend
does. It intentionally imports only stdlib modules so ``import
cybernetics.robotics`` works without MuJoCo, LocoMuJoCo, Isaac, ROS2, or
Worldlines installed.
"""

from .artifacts import (
    write_json_artifact,
    write_policy_artifact,
    write_robot_run_record,
    write_trajectory_dataset_artifact,
)
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
from .datasets import create_trajectory_dataset_from_runs
from .env import RobotEnv, StepResult
from .fixture import FixtureRobotEnv
from .locomujoco import LocoMuJoCoRobotEnv
from .runner import default_action, deterministic_run_id, run_robot_episode
from .vla_eval import (
    VLA_EVAL_RECORD_SCHEMA_VERSION,
    VlaEvalRunRecord,
    build_vla_eval_request,
    create_vla_eval_record,
)
from .worldlines import (
    WorldlinesAdapterError,
    WorldlinesModelPlaneAdapter,
    WorldlinesTrainingConfig,
    build_worldlines_training_payload,
    train_worldlines_policy,
)

__all__ = [
    "ROBOT_DATASET_SCHEMA_VERSION",
    "ROBOT_POLICY_SCHEMA_VERSION",
    "ROBOT_RUN_SCHEMA_VERSION",
    "ROBOT_TASK_SCHEMA_VERSION",
    "VLA_EVAL_RECORD_SCHEMA_VERSION",
    "WORLD_MODEL_SCHEMA_VERSION",
    "FixtureRobotEnv",
    "LocoMuJoCoRobotEnv",
    "PolicyArtifact",
    "RobotContractError",
    "RobotEnv",
    "RobotRunRecord",
    "RobotTaskSpec",
    "StepResult",
    "TrajectoryDatasetArtifact",
    "VlaEvalRunRecord",
    "WorldModelArtifact",
    "WorldlinesAdapterError",
    "WorldlinesModelPlaneAdapter",
    "WorldlinesTrainingConfig",
    "build_vla_eval_request",
    "build_worldlines_training_payload",
    "create_vla_eval_record",
    "default_action",
    "create_trajectory_dataset_from_runs",
    "deterministic_run_id",
    "run_robot_episode",
    "stable_hash",
    "train_worldlines_policy",
    "write_json_artifact",
    "write_policy_artifact",
    "write_robot_run_record",
    "write_trajectory_dataset_artifact",
]
