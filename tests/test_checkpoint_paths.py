from __future__ import annotations

import pytest

from cybernetics.types.checkpoint import ParsedCheckpointCyberneticsPath
from cybernetics.types.weights_info_response import WeightsInfoResponse
from cybernetics.lib.public_interfaces.rest_client import RestClient


@pytest.mark.parametrize(
    ("path", "training_run_id", "checkpoint_type", "checkpoint_id"),
    [
        (
            "worldlines://model_abc123/weights/final",
            "model_abc123",
            "training",
            "weights/final",
        ),
        (
            "worldlines://model_abc123/sampler_weights/public-alpha",
            "model_abc123",
            "sampler",
            "sampler_weights/public-alpha",
        ),
    ],
)
def test_parse_checkpoint_worldlines_path_accepts_backend_paths(
    path: str,
    training_run_id: str,
    checkpoint_type: str,
    checkpoint_id: str,
) -> None:
    parsed = ParsedCheckpointCyberneticsPath.from_worldlines_path(path)

    assert parsed.worldlines_path == path
    assert parsed.training_run_id == training_run_id
    assert parsed.checkpoint_type == checkpoint_type
    assert parsed.checkpoint_id == checkpoint_id


@pytest.mark.parametrize(
    "path",
    [
        "worldlines://model_abc123/archive/final",
        "https://example.com/model_abc123/weights/final",
        "worldlines://model_abc123/weights",
    ],
)
def test_parse_checkpoint_worldlines_path_rejects_invalid_paths(path: str) -> None:
    with pytest.raises(ValueError, match="Invalid worldlines path"):
        ParsedCheckpointCyberneticsPath.from_worldlines_path(path)


def test_checkpoint_route_prefers_resolved_control_plane_run_id() -> None:
    training_run_id, checkpoint_id = RestClient._checkpoint_route_from_worldlines_path(
        "worldlines://model_backend_abc/weights/final",
        WeightsInfoResponse(
            training_run_id="wlm_control_plane_123",
            base_model="dreamzero-droid",
            is_lora=True,
            lora_rank=4,
        ),
    )

    assert training_run_id == "wlm_control_plane_123"
    assert checkpoint_id == "weights/final"


def test_checkpoint_route_falls_back_to_parsed_backend_prefix() -> None:
    training_run_id, checkpoint_id = RestClient._checkpoint_route_from_worldlines_path(
        "worldlines://model_backend_abc/sampler_weights/eval",
        WeightsInfoResponse(
            base_model="dreamzero-droid",
            is_lora=True,
            lora_rank=4,
        ),
    )

    assert training_run_id == "model_backend_abc"
    assert checkpoint_id == "sampler_weights/eval"
