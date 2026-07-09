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
    ROBOT_TRANSPORT_SCHEMA_VERSION,
    WORLD_MODEL_SCHEMA_VERSION,
    PolicyArtifact,
    RobotContractError,
    RobotRunRecord,
    RobotTaskSpec,
    TrajectoryDatasetArtifact,
    TransportSpec,
    WorldModelArtifact,
    stable_hash,
)
from .datasets import create_trajectory_dataset_from_runs
from .env import RobotEnv, StepResult
from .fixture import FixtureRobotEnv
from .locomujoco import GymnasiumRobotEnvAdapter, LocoMuJoCoRobotEnv, RobotBackendError
from .replay import (
    ReplayImportRequest,
    build_replay_import_request,
    validate_policy_for_replay,
)
from .runner import default_action, deterministic_run_id, run_robot_episode
from .task_client import RobotTaskRunResult, RobotTasksClient
from .vla_eval import (
    VLA_EVAL_RECORD_SCHEMA_VERSION,
    VlaEvalRunRecord,
    build_vla_eval_request,
    create_vla_eval_record,
)
from .world_models import (
    build_cosmos_world_model_payload,
    create_synthetic_dataset_from_world_model,
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
    "ROBOT_TRANSPORT_SCHEMA_VERSION",
    "VLA_EVAL_RECORD_SCHEMA_VERSION",
    "WORLD_MODEL_SCHEMA_VERSION",
    "FixtureRobotEnv",
    "GymnasiumRobotEnvAdapter",
    "LocoMuJoCoRobotEnv",
    "PolicyArtifact",
    "ReplayImportRequest",
    "RobotContractError",
    "RobotBackendError",
    "RobotEnv",
    "RobotRunRecord",
    "RobotTaskRunResult",
    "RobotTaskSpec",
    "RobotTasksClient",
    "StepResult",
    "TrajectoryDatasetArtifact",
    "TransportSpec",
    "VlaEvalRunRecord",
    "WorldModelArtifact",
    "WorldlinesAdapterError",
    "WorldlinesModelPlaneAdapter",
    "WorldlinesTrainingConfig",
    "build_replay_import_request",
    "build_vla_eval_request",
    "build_cosmos_world_model_payload",
    "build_worldlines_training_payload",
    "create_synthetic_dataset_from_world_model",
    "create_vla_eval_record",
    "default_action",
    "create_trajectory_dataset_from_runs",
    "deterministic_run_id",
    "run_robot_episode",
    "stable_hash",
    "train_worldlines_policy",
    "validate_policy_for_replay",
    "write_json_artifact",
    "write_policy_artifact",
    "write_robot_run_record",
    "write_trajectory_dataset_artifact",
]
