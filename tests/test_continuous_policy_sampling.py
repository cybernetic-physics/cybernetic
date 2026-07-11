from __future__ import annotations

from cybernetics import types
from cybernetics._compat import model_dump
from cybernetics._models import construct_type
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
        conditioning={
            "images": _tensor([0, 1, 2], "int64", [1, 1, 1, 3]),
            "state": _tensor([0.5, -0.5], "float32", [1, 2]),
            "state_mask": _tensor([1], "int64", [1]),
            "embodiment_id": _tensor([0], "int64", [1]),
        },
    )

    body = model_dump(request, exclude_unset=True, mode="json")

    assert body.get("prompt", {"chunks": []}) == {"chunks": []}
    assert body["conditioning"]["images"]["shape"] == [1, 1, 1, 3]
    assert body["conditioning"]["state"]["data"] == [0.5, -0.5]


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
