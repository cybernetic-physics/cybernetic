from typing import Dict, Union

from typing_extensions import TypeAlias

from .tensor_data import TensorData
from .text_data import TextData

__all__ = ["PolicyConditioning"]

PolicyConditioning: TypeAlias = Dict[str, Union[TensorData, TextData]]
"""Policy conditioning values keyed by runtime-native names.

DreamZero and other VLA policies are conditioned by RGB/state/mask/embodiment
tensors rather than only by token prompts. The SDK keeps this carrier narrow:
clients send numeric tensors and shaped UTF-8 instruction observations under
runtime-owned keys.
"""
