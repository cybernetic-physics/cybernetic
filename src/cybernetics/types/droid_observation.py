from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._models import StrictBase
from .tensor_data import TensorData

__all__ = ["DroidObservation"]


class DroidObservation(StrictBase):
    """Raw DROID policy observation owned by the DreamZero backend.

    Images use ``H x W x 3`` RGB arrays. Joint position has seven values and
    gripper position has one. The backend owns frame accumulation, resizing,
    language tokenization, normalization, embodiment selection, and action
    unnormalization.
    """

    exterior_image_0_left: TensorData
    exterior_image_1_left: TensorData
    wrist_image_left: TensorData
    joint_position: TensorData
    gripper_position: TensorData
    instruction: str

    @classmethod
    def from_numpy(
        cls,
        *,
        exterior_image_0_left: npt.NDArray[Any],
        exterior_image_1_left: npt.NDArray[Any],
        wrist_image_left: npt.NDArray[Any],
        joint_position: npt.NDArray[Any],
        gripper_position: npt.NDArray[Any] | float,
        instruction: str,
    ) -> "DroidObservation":
        """Build the wire type from natural simulator/robot NumPy values."""
        gripper = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
        return cls(
            exterior_image_0_left=TensorData.from_numpy(np.asarray(exterior_image_0_left)),
            exterior_image_1_left=TensorData.from_numpy(np.asarray(exterior_image_1_left)),
            wrist_image_left=TensorData.from_numpy(np.asarray(wrist_image_left)),
            joint_position=TensorData.from_numpy(
                np.asarray(joint_position, dtype=np.float32).reshape(-1)
            ),
            gripper_position=TensorData.from_numpy(gripper),
            instruction=instruction,
        )
