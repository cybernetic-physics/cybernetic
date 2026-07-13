from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from .._models import StrictBase
from .tensor_data import TensorData

__all__ = ["DroidObservation"]


class DroidObservation(StrictBase):
    """Raw DROID policy observation transformed by the selected backend.

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

    @model_validator(mode="after")
    def _validate_droid_contract(self) -> "DroidObservation":
        for name in (
            "exterior_image_0_left",
            "exterior_image_1_left",
            "wrist_image_left",
        ):
            tensor = getattr(self, name)
            if tensor.dtype != "int64":
                raise ValueError(f"{name} must contain integer RGB values")
            if (
                tensor.shape is None
                or len(tensor.shape) != 3
                or tensor.shape[-1] != 3
                or any(size < 1 for size in tensor.shape)
            ):
                raise ValueError(f"{name} must have shape HxWx3")
            image = tensor.to_numpy()
            if image.min() < 0 or image.max() > 255:
                raise ValueError(f"{name} RGB values must be in [0, 255]")

        joint_position = self.joint_position.to_numpy()
        if self.joint_position.dtype != "float32" or joint_position.shape != (7,):
            raise ValueError("joint_position must be float32 with shape [7]")
        if not np.isfinite(joint_position).all():
            raise ValueError("joint_position must contain only finite values")

        gripper_position = self.gripper_position.to_numpy()
        if self.gripper_position.dtype != "float32" or gripper_position.shape != (1,):
            raise ValueError("gripper_position must be float32 with shape [1]")
        if not np.isfinite(gripper_position).all():
            raise ValueError("gripper_position must contain only finite values")
        if not 0.0 <= float(gripper_position[0]) <= 1.0:
            raise ValueError("gripper_position must be in [0, 1]")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        return self

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
