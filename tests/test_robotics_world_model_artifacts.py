from __future__ import annotations

import json

import pytest
from test_robotics_contracts import dataset_dict, task_dict

from cybernetics.robotics import (
    ROBOT_DATASET_SCHEMA_VERSION,
    WORLD_MODEL_SCHEMA_VERSION,
    RobotContractError,
    RobotTaskSpec,
    TrajectoryDatasetArtifact,
    WorldModelArtifact,
    build_cosmos_world_model_payload,
    create_synthetic_dataset_from_world_model,
)


def _task() -> RobotTaskSpec:
    return RobotTaskSpec.from_dict(task_dict())


def _dataset() -> TrajectoryDatasetArtifact:
    return TrajectoryDatasetArtifact.from_dict(dataset_dict(_task()))


def _world_model() -> WorldModelArtifact:
    dataset = _dataset()
    return WorldModelArtifact.from_dict(
        {
            "schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "artifact_id": "wm_cosmos_fixture",
            "task_spec_uri": dataset.task_spec_uri,
            "task_spec_hash": dataset.task_spec_hash,
            "source_dataset_artifact_id": dataset.artifact_id,
            "model_family": "cosmos3",
            "model_uri": "worldlines://world-models/cosmos3/fixture",
            "model_role": "world_model",
            "input_schema": dataset.observation_schema,
            "output_schema": {"future_frames": {"dtype": "uint8", "shape": [16, 224, 224, 3]}},
            "horizon": 1.6,
            "dt": 0.1,
            "finetune_dataset_uri": dataset.storage_uri,
            "synthetic_data_policy": {"generated_outputs_are": "synthetic"},
            "calibration_metrics": {"fid_proxy": 0.0},
            "backend_version": "cosmos3-mock/v1",
            "created_by_run_id": "world_model_job_001",
        }
    )


def test_world_model_artifact_round_trips_cosmos_task_dataset_lineage() -> None:
    artifact = _world_model()
    round_trip = WorldModelArtifact.from_dict(artifact.to_dict())

    assert round_trip == artifact
    assert round_trip.model_family == "cosmos3"
    assert round_trip.task_spec_hash == _dataset().task_spec_hash
    assert round_trip.source_dataset_artifact_id == _dataset().artifact_id
    assert "simulator_backend" not in round_trip.to_dict()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_role", "simulator", "model_role"),
        ("horizon", 0, "horizon"),
        ("dt", 0, "dt"),
        ("model_uri", "", "model_uri"),
        ("finetune_dataset_uri", "", "finetune_dataset_uri"),
        ("task_spec_hash", "", "task_spec_hash"),
    ],
)
def test_world_model_artifact_rejects_invalid_role_and_timing(
    field: str, value, match: str
) -> None:
    data = _world_model().to_dict()
    data[field] = value

    with pytest.raises(RobotContractError, match=match):
        WorldModelArtifact.from_dict(data)


def test_cosmos_generated_dataset_defaults_to_synthetic_provenance() -> None:
    source = _dataset()
    world_model = _world_model()

    generated = create_synthetic_dataset_from_world_model(
        world_model,
        source,
        "artifact://datasets/cosmos/generated",
        frame_count=16,
    )

    assert generated.schema_version == ROBOT_DATASET_SCHEMA_VERSION
    assert generated.data_provenance == "synthetic"
    assert generated.source_backend == source.source_backend
    assert generated.source_backend != "cosmos"
    assert generated.task_spec_hash == source.task_spec_hash
    assert generated.source_runs == [world_model.created_by_run_id]


def test_cosmos_generated_dataset_requires_explicit_mixed_provenance_sources() -> None:
    source = _dataset()
    world_model = _world_model()

    with pytest.raises(RobotContractError, match="mixed"):
        create_synthetic_dataset_from_world_model(
            world_model,
            source,
            "artifact://datasets/cosmos/mixed",
            frame_count=8,
            data_provenance="mixed",
        )

    generated = create_synthetic_dataset_from_world_model(
        world_model,
        source,
        "artifact://datasets/cosmos/mixed",
        frame_count=8,
        data_provenance="mixed",
        artifact_refs=[
            {"kind": "generated_video", "data_provenance": "synthetic"},
            {"kind": "source_rollout", "data_provenance": "sim"},
        ],
    )

    assert generated.data_provenance == "mixed"


def test_cosmos_payload_contains_model_plane_fields_only() -> None:
    dataset = _dataset()

    payload = build_cosmos_world_model_payload(
        dataset,
        model_args={"family": "cosmos3", "prompt": "future walking clip"},
    )

    assert set(payload) == {
        "dataset_uri",
        "observation_schema",
        "action_schema",
        "episode_count",
        "frame_count",
        "model_family",
        "data_provenance",
        "model_args",
    }
    assert payload["dataset_uri"] == dataset.storage_uri
    assert payload["model_family"] == "cosmos3"

    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "RobotEnv",
        "reward_spec",
        "success_metric",
        "termination",
        "simulator_handle",
        "callback",
        "eval_protocol",
    ):
        assert forbidden not in encoded


def test_world_model_artifact_does_not_replace_robot_env_evaluation() -> None:
    artifact = _world_model()
    payload = artifact.to_dict()

    assert "status" not in payload
    assert "success_rate" not in payload
    assert "terminated" not in payload
    assert artifact.calibration_metrics == {"fid_proxy": 0.0}
