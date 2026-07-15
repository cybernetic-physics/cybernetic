import asyncio
from concurrent.futures import Future as ConcurrentFuture

import numpy as np
import pytest

from cybernetics import types
from cybernetics.lib.openpi import (
    PI0_DROID_ACTION_SPACE,
    PI0_DROID_BASE_MODEL,
    PI0_DROID_DSRL_ACTION_SHAPE,
    PI0_DROID_INITIAL_FLOW_NOISE_CONTRACT_VERSION,
    PI0_DROID_INITIAL_FLOW_NOISE_SHAPE,
    Pi0DroidDsrlAction,
)
from cybernetics.lib.public_interfaces.sampling_client import SamplingClient


def test_pi0_droid_public_identifiers() -> None:
    assert PI0_DROID_BASE_MODEL == "pi0-droid"
    assert PI0_DROID_ACTION_SPACE == "droid_joint_position"
    assert PI0_DROID_DSRL_ACTION_SHAPE == (32,)
    assert PI0_DROID_INITIAL_FLOW_NOISE_SHAPE == (10, 32)
    assert PI0_DROID_INITIAL_FLOW_NOISE_CONTRACT_VERSION == 1


def _observation(**overrides: object) -> types.DroidObservation:
    values: dict[str, object] = {
        "exterior_image_0_left": np.zeros((2, 3, 3), dtype=np.uint8),
        "exterior_image_1_left": np.ones((2, 3, 3), dtype=np.uint8),
        "wrist_image_left": np.full((2, 3, 3), 2, dtype=np.uint8),
        "joint_position": np.arange(7, dtype=np.float32),
        "gripper_position": np.asarray([0.25], dtype=np.float32),
        "instruction": "pick up the cube",
    }
    values.update(overrides)
    return types.DroidObservation.from_numpy(**values)  # type: ignore[arg-type]


def test_droid_observation_validates_robot_boundary() -> None:
    observation = _observation()
    assert observation.joint_position.shape == [7]
    assert observation.gripper_position.shape == [1]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"exterior_image_0_left": np.zeros((2, 3), dtype=np.uint8)}, "HxWx3"),
        (
            {"exterior_image_1_left": np.full((2, 3, 3), 256, dtype=np.int64)},
            r"\[0, 255\]",
        ),
        ({"joint_position": np.zeros(6, dtype=np.float32)}, r"shape \[7\]"),
        ({"gripper_position": 1.1}, r"in \[0, 1\]"),
        ({"instruction": "   "}, "must not be empty"),
    ],
)
def test_droid_observation_rejects_malformed_values(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _observation(**override)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy_mode": "sde"}, "only policy_mode='native'"),
        ({"include_predicted_video": True}, "does not produce predicted video"),
        ({"seed": 3}, "does not support deterministic seed"),
    ],
)
def test_pi0_sampling_client_rejects_unsupported_options(
    kwargs: dict[str, object], message: str
) -> None:
    class _Client:
        _base_model = "pi0-droid"

        def sample(self, **_kwargs: object) -> object:
            raise AssertionError("unsupported PI0 request must not be sent")

    with pytest.raises(ValueError, match=message):
        SamplingClient.sample_droid(  # type: ignore[arg-type]
            _Client(),
            _observation(),
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy_mode": "sde"}, "only policy_mode='native'"),
        ({"include_predicted_video": True}, "does not produce predicted video"),
        ({"seed": 3}, "does not support deterministic seed"),
    ],
)
def test_typed_dsrl_action_rejects_pi0_options_when_model_is_inferred(
    kwargs: dict[str, object], message: str
) -> None:
    class _Client:
        _base_model = None

        def sample(self, **_kwargs: object) -> object:
            raise AssertionError("unsupported PI0 request must not be sent")

    with pytest.raises(ValueError, match=message):
        SamplingClient.sample_droid(  # type: ignore[arg-type]
            _Client(),
            _observation(),
            dsrl_action=Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32)),
            **kwargs,  # type: ignore[arg-type]
        )


def test_pi0_sampling_client_expands_typed_dsrl_action() -> None:
    observed: dict[str, object] = {}

    class _Client:
        _base_model = "pi0-droid"

        def sample(self, **kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            observed.update(kwargs)
            future: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()
            future.set_result(
                types.SampleResponse(
                    sequences=[],
                    policy_metadata={
                        "pi0_initial_flow_noise": {
                            "contract_version": 1,
                            "applied": True,
                            "dtype": "float32",
                            "shape": [10, 32],
                            "sha256": dsrl_action.initial_flow_noise_sha256(),
                        }
                    },
                )
            )
            return future

    action_values = np.arange(32, dtype=np.float32)
    dsrl_action = Pi0DroidDsrlAction.from_numpy(action_values)
    result = SamplingClient.sample_droid(  # type: ignore[arg-type]
        _Client(),
        _observation(),
        dsrl_action=dsrl_action,
    )

    assert result.result().policy_metadata is not None
    initial_noise = observed["pi0_initial_flow_noise"]
    assert isinstance(initial_noise, types.TensorData)
    assert initial_noise.dtype == "float32"
    assert initial_noise.shape == [10, 32]
    np.testing.assert_array_equal(
        initial_noise.to_numpy(),
        np.repeat(action_values[np.newaxis, :], repeats=10, axis=0),
    )


def test_pi0_sample_request_serializes_validated_wire_noise() -> None:
    noise = Pi0DroidDsrlAction.from_numpy(
        np.arange(32, dtype=np.float32)
    ).to_pi0_initial_flow_noise()
    request = types.SampleRequest(
        sampling_session_id="wlss_pi0",
        seq_id=3,
        sampling_params=types.SamplingParams(max_tokens=1),
        droid_observation=_observation(),
        pi0_initial_flow_noise=noise,
    )

    assert request.pi0_initial_flow_noise is not None
    assert request.pi0_initial_flow_noise.dtype == "float32"
    assert request.pi0_initial_flow_noise.shape == [10, 32]
    assert len(request.pi0_initial_flow_noise.data) == 320


@pytest.mark.parametrize(
    ("noise", "observation", "base_model", "message"),
    [
        (
            types.TensorData(data=[0.0] * 32, dtype="float32", shape=[32]),
            _observation(),
            None,
            "shape must be",
        ),
        (
            types.TensorData(data=[0] * 320, dtype="int64", shape=[10, 32]),
            _observation(),
            None,
            "dtype must be float32",
        ),
        (
            types.TensorData(data=[float("nan")] * 320, dtype="float32", shape=[10, 32]),
            _observation(),
            None,
            "finite",
        ),
        (
            types.TensorData(data=[0.0] * 320, dtype="float32", shape=[10, 32]),
            None,
            None,
            "requires droid_observation",
        ),
        (
            types.TensorData(data=[0.0] * 320, dtype="float32", shape=[10, 32]),
            _observation(),
            "dreamzero-droid",
            "supported only by pi0-droid",
        ),
    ],
)
def test_pi0_sample_request_rejects_invalid_wire_noise(
    noise: types.TensorData,
    observation: types.DroidObservation | None,
    base_model: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        types.SampleRequest(
            base_model=base_model,
            sampling_params=types.SamplingParams(max_tokens=1),
            droid_observation=observation,
            pi0_initial_flow_noise=noise,
        )


def test_pi0_sampling_client_dsrl_future_fails_closed_without_ack() -> None:
    class _Client:
        _base_model = "pi0-droid"

        def sample(self, **_kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            future: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()
            future.set_result(types.SampleResponse(sequences=[], policy_metadata={}))
            return future

    dsrl_action = Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32))
    future = SamplingClient.sample_droid(  # type: ignore[arg-type]
        _Client(),
        _observation(),
        dsrl_action=dsrl_action,
    )

    with pytest.raises(ValueError, match="did not acknowledge"):
        future.result()


def test_pi0_sampling_client_dsrl_async_fails_closed_without_ack() -> None:
    class _Client:
        _base_model = "pi0-droid"
        sample_droid = SamplingClient.sample_droid

        def sample(self, **_kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            future: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()
            future.set_result(types.SampleResponse(sequences=[], policy_metadata={}))
            return future

    dsrl_action = Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32))

    with pytest.raises(ValueError, match="did not acknowledge"):
        asyncio.run(
            SamplingClient.sample_droid_async(  # type: ignore[arg-type]
                _Client(),
                _observation(),
                dsrl_action=dsrl_action,
            )
        )


def test_pi0_sampling_client_snapshots_exact_wire_noise_before_submit() -> None:
    source: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()

    class _Client:
        _base_model = "pi0-droid"

        def sample(self, **_kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            return source

    dsrl_action = Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32))
    expected_sha256 = dsrl_action.initial_flow_noise_sha256()
    verified = SamplingClient.sample_droid(  # type: ignore[arg-type]
        _Client(),
        _observation(),
        dsrl_action=dsrl_action,
    )
    dsrl_action.values.data[0] = 1.0
    assert dsrl_action.initial_flow_noise_sha256() != expected_sha256
    source.set_result(
        types.SampleResponse(
            sequences=[],
            policy_metadata={
                "pi0_initial_flow_noise": {
                    "contract_version": 1,
                    "applied": True,
                    "dtype": "float32",
                    "shape": [10, 32],
                    "sha256": expected_sha256,
                }
            },
        )
    )

    assert verified.result().policy_metadata is not None


def test_pi0_sampling_client_dsrl_cancellation_propagates_both_directions() -> None:
    dsrl_action = Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32))

    source_from_caller: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()

    class _CallerCancelledClient:
        _base_model = "pi0-droid"

        def sample(self, **_kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            return source_from_caller

    verified = SamplingClient.sample_droid(  # type: ignore[arg-type]
        _CallerCancelledClient(),
        _observation(),
        dsrl_action=dsrl_action,
    )
    assert verified.cancel()
    assert source_from_caller.cancelled()

    source_from_server: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()

    class _ServerCancelledClient:
        _base_model = "pi0-droid"

        def sample(self, **_kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            return source_from_server

    verified = SamplingClient.sample_droid(  # type: ignore[arg-type]
        _ServerCancelledClient(),
        _observation(),
        dsrl_action=dsrl_action,
    )
    assert source_from_server.cancel()
    assert verified.cancelled()


@pytest.mark.asyncio
async def test_pi0_sampling_client_async_cancellation_cancels_dependent_source() -> None:
    source: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()

    class _Client:
        _base_model = "pi0-droid"
        sample_droid = SamplingClient.sample_droid

        def sample(self, **_kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            return source

    task = asyncio.create_task(
        SamplingClient.sample_droid_async(  # type: ignore[arg-type]
            _Client(),
            _observation(),
            dsrl_action=Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32)),
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.cancelled()


def test_pi0_sampling_client_keeps_concurrent_acknowledgements_isolated() -> None:
    submitted: list[tuple[ConcurrentFuture[types.SampleResponse], types.TensorData]] = []

    class _Client:
        _base_model = "pi0-droid"

        def sample(self, **kwargs: object) -> ConcurrentFuture[types.SampleResponse]:
            future: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()
            noise = kwargs["pi0_initial_flow_noise"]
            assert isinstance(noise, types.TensorData)
            submitted.append((future, noise))
            return future

    client = _Client()
    first = SamplingClient.sample_droid(  # type: ignore[arg-type]
        client,
        _observation(),
        dsrl_action=Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32)),
    )
    second = SamplingClient.sample_droid(  # type: ignore[arg-type]
        client,
        _observation(),
        dsrl_action=Pi0DroidDsrlAction.from_numpy(np.ones(32, dtype=np.float32)),
    )

    for source, noise in reversed(submitted):
        source.set_result(
            types.SampleResponse(
                sequences=[],
                policy_metadata={
                    "pi0_initial_flow_noise": {
                        "contract_version": 1,
                        "applied": True,
                        "dtype": "float32",
                        "shape": [10, 32],
                        "sha256": Pi0DroidDsrlAction(
                            values=types.TensorData(
                                data=noise.data[:32],
                                dtype="float32",
                                shape=[32],
                            )
                        ).initial_flow_noise_sha256(),
                    }
                },
            )
        )

    assert first.result().policy_metadata is not None
    assert second.result().policy_metadata is not None


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(31, dtype=np.float32),
        np.zeros(32, dtype=np.float64),
        np.full(32, np.nan, dtype=np.float32),
        np.full(32, np.inf, dtype=np.float32),
    ],
)
def test_pi0_dsrl_action_rejects_invalid_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Pi0DroidDsrlAction.from_numpy(values)


def test_pi0_dsrl_action_fails_closed_on_missing_or_mismatched_ack() -> None:
    action = Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32))
    with pytest.raises(ValueError, match="did not acknowledge"):
        action.require_applied_policy_metadata({})
    with pytest.raises(ValueError, match="different initial flow noise"):
        action.require_applied_policy_metadata(
            {
                "pi0_initial_flow_noise": {
                    "contract_version": 1,
                    "applied": True,
                    "dtype": "float32",
                    "shape": [10, 32],
                    "sha256": "wrong",
                }
            }
        )

    action.require_applied_policy_metadata(
        {
            "pi0_initial_flow_noise": {
                "contract_version": 1,
                "applied": True,
                "dtype": "float32",
                "shape": [10, 32],
                "sha256": action.initial_flow_noise_sha256(),
            }
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", True),
        ("applied", 1),
        ("shape", [10.0, 32.0]),
    ],
)
def test_pi0_dsrl_ack_requires_exact_json_types(field: str, value: object) -> None:
    action = Pi0DroidDsrlAction.from_numpy(np.zeros(32, dtype=np.float32))
    acknowledgement: dict[str, object] = {
        "contract_version": 1,
        "applied": True,
        "dtype": "float32",
        "shape": [10, 32],
        "sha256": action.initial_flow_noise_sha256(),
    }
    acknowledgement[field] = value

    with pytest.raises(ValueError, match="different initial flow noise"):
        action.require_applied_policy_metadata({"pi0_initial_flow_noise": acknowledgement})
