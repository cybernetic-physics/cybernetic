from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from cybernetics import types
from cybernetics._compat import model_dump
from cybernetics._models import construct_type
from cybernetics.lib.public_interfaces.sampling_client import SamplingClient
from cybernetics.resources.service import _model_dump_omit_none


def _tensor(data: list[int] | list[float], dtype: str, shape: list[int]) -> types.TensorData:
    return types.TensorData(data=data, dtype=dtype, shape=shape)


def test_create_sampling_session_omits_null_model_path_for_base_model() -> None:
    request = types.CreateSamplingSessionRequest(
        session_id="wls_123",
        sampling_session_seq_id=0,
        base_model="dreamzero-droid",
        model_path=None,
    )

    body = _model_dump_omit_none(request)

    assert body == {
        "session_id": "wls_123",
        "sampling_session_seq_id": 0,
        "base_model": "dreamzero-droid",
    }


def test_sample_request_carries_continuous_policy_conditioning() -> None:
    request = types.SampleRequest(
        sampling_session_id="wlss_123",
        seq_id=7,
        sampling_params=types.SamplingParams(max_tokens=1),
        policy_mode="native",
        include_predicted_video=True,
        conditioning={
            "images": _tensor([0, 1, 2], "int64", [1, 1, 1, 3]),
            "state": _tensor([0.5, -0.5], "float32", [1, 2]),
            "state_mask": _tensor([1], "int64", [1]),
            "embodiment_id": _tensor([0], "int64", [1]),
        },
        policy_context=types.PolicySessionContext(
            sequence_ids=["episode-000001"],
            step_ids=[7],
            reset_mask=[True],
            seeds=[41],
        ),
    )

    body = model_dump(request, exclude_unset=True, mode="json")

    assert body.get("prompt", {"chunks": []}) == {"chunks": []}
    assert body["conditioning"]["images"]["shape"] == [1, 1, 1, 3]
    assert body["conditioning"]["state"]["data"] == [0.5, -0.5]
    assert body["policy_mode"] == "native"
    assert body["include_predicted_video"] is True
    assert body["policy_context"] == {
        "sequence_ids": ["episode-000001"],
        "step_ids": [7],
        "reset_mask": [True],
        "seeds": [41],
    }


def test_raw_droid_observation_is_typed_and_sample_droid_is_ergonomic() -> None:
    observation = types.DroidObservation.from_numpy(
        exterior_image_0_left=np.zeros((2, 3, 3), dtype=np.uint8),
        exterior_image_1_left=np.ones((2, 3, 3), dtype=np.uint8),
        wrist_image_left=np.full((2, 3, 3), 2, dtype=np.uint8),
        joint_position=np.arange(7, dtype=np.float32),
        gripper_position=0.25,
        instruction="pick up the object",
    )
    observed: dict[str, object] = {}

    class _Client:
        def sample(self, **kwargs):
            observed.update(kwargs)
            return "future"

    result = SamplingClient.sample_droid(  # type: ignore[arg-type]
        _Client(),
        observation,
        include_predicted_video=True,
        seed=7,
    )

    assert result == "future"
    assert observed["prompt"] == types.ModelInput.empty()
    assert observed["droid_observation"] == observation
    assert observed["policy_mode"] == "native"
    assert observed["include_predicted_video"] is True
    assert observed["sampling_params"].seed == 7  # type: ignore[union-attr]
    body = model_dump(observation, mode="json")
    assert body["exterior_image_0_left"]["shape"] == [2, 3, 3]
    assert body["joint_position"]["shape"] == [7]
    assert body["instruction"] == "pick up the object"


def test_sampling_client_carries_policy_mode_and_video_request() -> None:
    observed_requests: list[types.SampleRequest] = []

    class _FakeSamplingAPI:
        async def asample(self, **kwargs):
            observed_requests.append(kwargs["request"])
            return SimpleNamespace(request_id="req-policy")

    class _FakeClient:
        sampling = _FakeSamplingAPI()

    class _FakeHolder:
        async def sample_request_extra_headers(self, *, request_kind="sample"):
            assert request_kind == "sample"
            return {}

        def aclient(self, client_pool_type):
            @contextmanager
            def _ctx():
                yield _FakeClient()

            return _ctx()

    client = object.__new__(SamplingClient)
    client.holder = _FakeHolder()
    client._sampling_session_id = "sample-policy"
    client._request_id_counter = 0
    client._last_queue_state_logged = 0

    result = asyncio.run(
        client._send_asample_request(
            1,
            types.ModelInput.empty(),
            None,
            types.SamplingParams(max_tokens=1),
            False,
            0,
            request_id=17,
            policy_mode="native",
            include_predicted_video=True,
        )
    )

    assert result.request_id == "req-policy"
    assert observed_requests[0].seq_id == 17
    assert observed_requests[0].policy_mode == "native"
    assert observed_requests[0].include_predicted_video is True


def test_policy_session_context_rejects_mismatched_lanes() -> None:
    with pytest.raises(ValidationError, match="must match sequence_ids length"):
        types.PolicySessionContext(
            sequence_ids=["lane-0", "lane-1"],
            step_ids=[0],
            reset_mask=[True, True],
            seeds=[10, 11],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sequence_ids", [], "1..4096 lanes"),
        ("sequence_ids", ["x" * 257], "at most 256 characters"),
        ("step_ids", [-1], "JSON-safe integers"),
        ("seeds", [2**53], "JSON-safe integers"),
    ],
)
def test_policy_session_context_bounds_hosted_state_keys(
    field: str, value: list[object], message: str
) -> None:
    context = {
        "sequence_ids": ["lane-0"],
        "step_ids": [0],
        "reset_mask": [True],
        "seeds": [41],
    }
    context[field] = value

    with pytest.raises(ValidationError, match=message):
        types.PolicySessionContext(**context)


def test_tensor_data_accepts_natural_rgb_and_mask_numpy_dtypes() -> None:
    images = types.TensorData.from_numpy(np.array([0, 127, 255], dtype=np.uint8))
    mask = types.TensorData.from_numpy(np.array([True, False], dtype=bool))

    assert images.dtype == "int64"
    assert images.data == [0, 127, 255]
    assert images.to_numpy().dtype == np.int64
    assert mask.dtype == "int64"
    assert mask.data == [1, 0]


def test_sample_response_preserves_continuous_policy_artifacts() -> None:
    response = types.SampleResponse(
        sequences=[],
        action_chunk=_tensor([0.1, 0.2], "float32", [1, 2]),
        trajectory=[{"x_t": _tensor([0.3], "float32", [1])}],
        video=_tensor([1, 2, 3], "int64", [1, 1, 1, 3]),
        predicted_video=_tensor([4, 5, 6], "int64", [1, 1, 1, 3]),
    )

    assert response.action_chunk is not None
    assert response.action_chunk.shape == [1, 2]
    assert response.action_chunk.data == [0.1, 0.2]
    assert response.trajectory is not None
    assert response.trajectory[0]["x_t"].data == [0.3]
    assert response.video is not None
    assert response.video.shape == [1, 1, 1, 3]
    assert response.predicted_video is not None
    assert response.predicted_video.data == [4, 5, 6]


def test_future_retrieve_response_includes_sample_response_artifacts() -> None:
    response = construct_type(
        type_=types.FutureRetrieveResponse,
        value={
            "type": "sample",
            "sequences": [],
            "action_chunk": {"data": [0.1, 0.2], "dtype": "float32", "shape": [1, 2]},
            "trajectory": [{"x_t": {"data": [0.3], "dtype": "float32", "shape": [1]}}],
            "predicted_video": {"data": [4, 5, 6], "dtype": "int64", "shape": [1, 1, 1, 3]},
        },
    )

    assert isinstance(response, types.SampleResponse)
    assert response.action_chunk is not None
    assert response.action_chunk.shape == [1, 2]
    assert response.trajectory is not None
    assert response.trajectory[0]["x_t"].data == [0.3]
    assert response.predicted_video is not None
    assert response.predicted_video.shape == [1, 1, 1, 3]
