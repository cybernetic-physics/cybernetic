from typing import Dict

from typing_extensions import TypeAlias

from .tensor_data import TensorData

__all__ = ["PolicyConditioning"]

PolicyConditioning: TypeAlias = Dict[str, TensorData]
"""Continuous-policy conditioning tensors keyed by runtime-native names.

DreamZero and other VLA policies are conditioned by RGB/state/mask/embodiment
tensors rather than only by token prompts. The SDK keeps this carrier narrow:
clients send TensorData values under runtime-owned keys such as ``images``,
``state``, ``state_mask``, and ``embodiment_id``.
"""
