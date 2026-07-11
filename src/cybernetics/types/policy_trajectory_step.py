from typing import Dict

from typing_extensions import TypeAlias

from .tensor_data import TensorData

__all__ = ["PolicyTrajectoryStep"]

PolicyTrajectoryStep: TypeAlias = Dict[str, TensorData]
"""One continuous-policy rollout step keyed by runtime-native artifact names."""
