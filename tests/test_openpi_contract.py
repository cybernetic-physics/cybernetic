import numpy as np
import pytest

from cybernetics import types
from cybernetics.lib.openpi import PI0_DROID_ACTION_SPACE, PI0_DROID_BASE_MODEL
from cybernetics.lib.public_interfaces.sampling_client import SamplingClient


def test_pi0_droid_public_identifiers() -> None:
    assert PI0_DROID_BASE_MODEL == "pi0-droid"
    assert PI0_DROID_ACTION_SPACE == "droid_joint_position"


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
