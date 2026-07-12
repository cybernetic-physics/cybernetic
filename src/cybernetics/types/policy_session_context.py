from typing import Any

import pydantic

from .._compat import PYDANTIC_V2
from .._models import StrictBase

__all__ = ["PolicySessionContext"]

_MAX_LANES = 4096
_MAX_SEQUENCE_ID_LENGTH = 256
_MAX_SAFE_INTEGER = (1 << 53) - 1


def _validate_lane_values(
    sequence_ids: list[str],
    step_ids: list[int],
    reset_mask: list[bool],
    seeds: list[int],
) -> None:
    width = len(sequence_ids)
    if width < 1 or width > _MAX_LANES:
        raise ValueError(f"policy context must contain 1..{_MAX_LANES} lanes")
    if len(set(sequence_ids)) != width:
        raise ValueError("policy sequence_ids must be unique within a request")
    if any(not sequence_id.strip() for sequence_id in sequence_ids):
        raise ValueError("policy sequence_ids must not be empty")
    if any(len(sequence_id) > _MAX_SEQUENCE_ID_LENGTH for sequence_id in sequence_ids):
        raise ValueError(
            f"policy sequence_ids must be at most {_MAX_SEQUENCE_ID_LENGTH} characters"
        )
    for name, lane_values in (
        ("step_ids", step_ids),
        ("reset_mask", reset_mask),
        ("seeds", seeds),
    ):
        if len(lane_values) != width:
            raise ValueError(f"policy {name} must match sequence_ids length")
    if any(step_id < 0 or step_id > _MAX_SAFE_INTEGER for step_id in step_ids):
        raise ValueError("policy step_ids must be nonnegative JSON-safe integers")
    if any(seed < 0 or seed > _MAX_SAFE_INTEGER for seed in seeds):
        raise ValueError("policy seeds must be nonnegative JSON-safe integers")


class PolicySessionContext(StrictBase):
    """Per-lane state metadata for a continuous policy sampling request.

    The authenticated sampling session remains the tenant boundary. Sequence
    ids identify recurrent lanes within that session; reset state is applied by
    the hosted sampler before it evaluates the matching observation batch.
    """

    sequence_ids: list[str]
    step_ids: list[int]
    reset_mask: list[bool]
    seeds: list[int]

    if PYDANTIC_V2:

        @pydantic.model_validator(mode="after")
        def validate_lanes(self) -> "PolicySessionContext":
            _validate_lane_values(
                self.sequence_ids,
                self.step_ids,
                self.reset_mask,
                self.seeds,
            )
            return self

    else:

        @pydantic.root_validator(skip_on_failure=True)
        def validate_lanes(cls, values: dict[str, Any]) -> dict[str, Any]:
            _validate_lane_values(
                values["sequence_ids"],
                values["step_ids"],
                values["reset_mask"],
                values["seeds"],
            )
            return values
