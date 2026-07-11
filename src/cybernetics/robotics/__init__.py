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
from .client import RobotEvalsClient, RobotEvalsError
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
from .env import ObservationBundle, RobotEnv, StepResult, VectorRobotEnv, VectorStepResult
from .evidence import RoboticsEvidenceError, write_robotics_behavior_bundle
from .fixture import FixtureRobotEnv
from .gymnasium import GymnasiumRobotEnvAdapter, GymnasiumVectorEnvAdapter, RobotBackendError
from .locomujoco import LocoMuJoCoRobotEnv
from .replay import (
    ReplayImportRequest,
    build_replay_import_request,
    validate_policy_for_replay,
)
from .runner import default_action, deterministic_run_id, run_robot_episode
from .runtime_contracts import (
    ARTIFACT_REF_SCHEMA_VERSION,
    ASSET_BUNDLE_REF_SCHEMA_VERSION,
    EPISODE_MANIFEST_SCHEMA_VERSION,
    POLICY_DEPLOYMENT_SCHEMA_VERSION,
    ROBOTICS_JOB_SCHEMA_VERSION,
    SIMULATOR_PACKAGE_SCHEMA_VERSION,
    TASK_PACKAGE_SCHEMA_VERSION,
    ActionChunk,
    ActionSelectionSpec,
    ActionSpec,
    ArtifactRef,
    AssetBundleRef,
    AssetMountSpec,
    EnvironmentFactorySpec,
    EnvironmentReadinessSpec,
    EpisodeManifest,
    EvaluationCheck,
    EvaluationSpec,
    PlacementSpec,
    PolicyDeploymentSpec,
    RecordingSpec,
    RoboticsJobSpec,
    RolloutSpec,
    RuntimeResources,
    SimulatorPackageSpec,
    TaskPackageSpec,
    TensorSpec,
    canonical_runtime_json,
    runtime_contract_hash,
)
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
    "ARTIFACT_REF_SCHEMA_VERSION",
    "ASSET_BUNDLE_REF_SCHEMA_VERSION",
    "EPISODE_MANIFEST_SCHEMA_VERSION",
    "POLICY_DEPLOYMENT_SCHEMA_VERSION",
    "ROBOT_POLICY_SCHEMA_VERSION",
    "ROBOTICS_JOB_SCHEMA_VERSION",
    "ROBOT_RUN_SCHEMA_VERSION",
    "ROBOT_TASK_SCHEMA_VERSION",
    "SIMULATOR_PACKAGE_SCHEMA_VERSION",
    "TASK_PACKAGE_SCHEMA_VERSION",
    "VLA_EVAL_RECORD_SCHEMA_VERSION",
    "WORLD_MODEL_SCHEMA_VERSION",
    "FixtureRobotEnv",
    "GymnasiumRobotEnvAdapter",
    "GymnasiumVectorEnvAdapter",
    "LocoMuJoCoRobotEnv",
    "PolicyArtifact",
    "PolicyDeploymentSpec",
    "ReplayImportRequest",
    "RobotContractError",
    "RobotEvalsClient",
    "RobotEvalsError",
    "RobotBackendError",
    "RobotEnv",
    "RobotRunRecord",
    "RobotTaskSpec",
    "RoboticsEvidenceError",
    "RoboticsJobSpec",
    "RolloutSpec",
    "RuntimeResources",
    "SimulatorPackageSpec",
    "TaskPackageSpec",
    "StepResult",
    "VectorRobotEnv",
    "VectorStepResult",
    "ObservationBundle",
    "TensorSpec",
    "ActionSpec",
    "ActionChunk",
    "ActionSelectionSpec",
    "ArtifactRef",
    "AssetBundleRef",
    "AssetMountSpec",
    "EnvironmentFactorySpec",
    "EnvironmentReadinessSpec",
    "EpisodeManifest",
    "EvaluationCheck",
    "EvaluationSpec",
    "PlacementSpec",
    "RecordingSpec",
    "TrajectoryDatasetArtifact",
    "VlaEvalRunRecord",
    "WorldModelArtifact",
    "WorldlinesAdapterError",
    "WorldlinesModelPlaneAdapter",
    "WorldlinesTrainingConfig",
    "build_replay_import_request",
    "build_vla_eval_request",
    "build_cosmos_world_model_payload",
    "build_worldlines_training_payload",
    "canonical_runtime_json",
    "create_synthetic_dataset_from_world_model",
    "create_vla_eval_record",
    "default_action",
    "create_trajectory_dataset_from_runs",
    "deterministic_run_id",
    "run_robot_episode",
    "runtime_contract_hash",
    "stable_hash",
    "train_worldlines_policy",
    "validate_policy_for_replay",
    "write_json_artifact",
    "write_policy_artifact",
    "write_robot_run_record",
    "write_robotics_behavior_bundle",
    "write_trajectory_dataset_artifact",
]
