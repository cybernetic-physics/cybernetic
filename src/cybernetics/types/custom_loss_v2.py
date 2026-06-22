from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from .datum import Datum

LossInputName = Literal[
    "target_logprobs",
    "prompt_logprobs",
    "behavior_logprobs",
    "reference_logprobs",
    "candidate_logprobs",
    "values",
    "advantages",
    "returns",
]

Layout = Literal["padded", "ragged"]


@dataclass(slots=True)
class GroupingSpec:
    group_ids: Any | None = None
    pair_ids: Any | None = None
    trajectory_ids: Any | None = None
    turn_ids: Any | None = None


@dataclass(slots=True)
class CustomLossContextV2:
    data: list[Datum]
    target_logprobs: Any
    token_mask: Any | None = None
    seq_lens: Any | None = None
    prompt_logprobs: Any | None = None
    behavior_logprobs: Any | None = None
    reference_logprobs: Any | None = None
    candidate_logprobs: Any | None = None
    values: Any | None = None
    advantages: Any | None = None
    returns: Any | None = None
    grouping: GroupingSpec | None = None
    metadata: dict[str, Any] | None = None
    layout: Layout = "padded"


@dataclass(slots=True)
class CustomLossOutputV2:
    loss: Any
    metrics: dict[str, float]
    grad_wrt_inputs: dict[LossInputName, Any] | None = None


CustomLossFnV2 = Callable[[CustomLossContextV2], CustomLossOutputV2]

RequestedLossInputs = Sequence[LossInputName]
