from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from .._models import StrictBase
from .tensor_data import TensorData

__all__ = ["Pi0DroidDsrlAction"]

PI0_DROID_DSRL_ACTION_SHAPE = (32,)
PI0_DROID_INITIAL_FLOW_NOISE_SHAPE = (10, 32)
PI0_DROID_INITIAL_FLOW_NOISE_CONTRACT_VERSION = 1


class Pi0DroidDsrlAction(StrictBase):
    """One DSRL SAC action for the frozen PI0-DROID policy.

    DSRL learns a 32-dimensional action. The Worldlines OpenPI boundary consumes
    the same vector repeated across PI0's ten-step flow-noise horizon.
    """

    values: TensorData

    @model_validator(mode="after")
    def _validate_action(self) -> "Pi0DroidDsrlAction":
        _validate_float32_tensor(
            self.values,
            name="Pi0DroidDsrlAction.values",
            shape=PI0_DROID_DSRL_ACTION_SHAPE,
        )
        return self

    @classmethod
    def from_numpy(cls, action: npt.NDArray[Any]) -> "Pi0DroidDsrlAction":
        array = np.asarray(action)
        if array.dtype != np.dtype(np.float32):
            raise ValueError("PI0-DROID DSRL action dtype must be float32")
        if array.shape != PI0_DROID_DSRL_ACTION_SHAPE:
            raise ValueError(f"PI0-DROID DSRL action shape must be [32], got {list(array.shape)}")
        if not np.isfinite(array).all():
            raise ValueError("PI0-DROID DSRL action must contain only finite values")
        return cls(values=TensorData.from_numpy(np.ascontiguousarray(array)))

    def to_pi0_initial_flow_noise(self) -> TensorData:
        action = _validate_float32_tensor(
            self.values,
            name="Pi0DroidDsrlAction.values",
            shape=PI0_DROID_DSRL_ACTION_SHAPE,
        )
        expanded = np.repeat(action[np.newaxis, :], repeats=10, axis=0)
        return TensorData.from_numpy(np.ascontiguousarray(expanded, dtype=np.float32))

    def initial_flow_noise_sha256(self) -> str:
        return pi0_initial_flow_noise_sha256(self.to_pi0_initial_flow_noise())

    def require_applied_policy_metadata(self, policy_metadata: Mapping[str, Any]) -> None:
        require_pi0_initial_flow_noise_ack(
            policy_metadata,
            expected_sha256=self.initial_flow_noise_sha256(),
        )


def validate_pi0_initial_flow_noise(value: TensorData) -> npt.NDArray[np.float32]:
    return _validate_float32_tensor(
        value,
        name="pi0_initial_flow_noise",
        shape=PI0_DROID_INITIAL_FLOW_NOISE_SHAPE,
    )


def pi0_initial_flow_noise_sha256(value: TensorData) -> str:
    """Hash the validated little-endian float32 wire tensor."""
    noise = validate_pi0_initial_flow_noise(value)
    canonical = np.ascontiguousarray(noise, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def require_pi0_initial_flow_noise_ack(
    policy_metadata: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> None:
    """Require a type-exact acknowledgement for one submitted noise tensor."""
    acknowledgement = policy_metadata.get("pi0_initial_flow_noise")
    if not isinstance(acknowledgement, Mapping):
        raise ValueError(
            "PI0 response did not acknowledge pi0_initial_flow_noise; "
            "the server may not support DSRL steering"
        )

    expected = {
        "contract_version": PI0_DROID_INITIAL_FLOW_NOISE_CONTRACT_VERSION,
        "applied": True,
        "dtype": "float32",
        "shape": list(PI0_DROID_INITIAL_FLOW_NOISE_SHAPE),
        "sha256": expected_sha256,
    }
    observed = {key: acknowledgement.get(key) for key in expected}
    shape = observed["shape"]
    matches = (
        type(observed["contract_version"]) is int
        and observed["contract_version"] == expected["contract_version"]
        and observed["applied"] is True
        and type(observed["dtype"]) is str
        and observed["dtype"] == expected["dtype"]
        and isinstance(shape, list)
        and len(shape) == len(PI0_DROID_INITIAL_FLOW_NOISE_SHAPE)
        and all(type(item) is int for item in shape)
        and shape == expected["shape"]
        and type(observed["sha256"]) is str
        and observed["sha256"] == expected["sha256"]
    )
    if not matches:
        raise ValueError(
            "PI0 response acknowledged different initial flow noise: "
            f"expected {expected!r}, got {observed!r}"
        )


def _validate_float32_tensor(
    value: TensorData,
    *,
    name: str,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    if value.dtype != "float32":
        raise ValueError(f"{name} dtype must be float32")
    if value.shape != list(shape):
        raise ValueError(f"{name} shape must be {list(shape)}")
    if len(value.data) != int(np.prod(shape)):
        raise ValueError(f"{name} data length does not match its shape")
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value.data
    ):
        raise ValueError(f"{name} must contain only numeric values")
    try:
        array = np.asarray(value.data, dtype=np.float32).reshape(shape)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid values: {exc}") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)
