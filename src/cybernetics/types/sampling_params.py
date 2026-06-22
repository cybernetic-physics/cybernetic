from typing import Any, Dict, Optional, Sequence, Union

from pydantic import model_validator

from .._models import BaseModel

__all__ = ["SamplingParams"]


class SamplingParams(BaseModel):
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate"""

    seed: Optional[int] = None
    """Random seed for reproducible generation"""

    stop: Union[str, Sequence[str], Sequence[int], None] = None
    """Stop sequences for generation"""

    temperature: float = 1
    """Sampling temperature"""

    top_k: int = -1
    """Top-k sampling parameter (-1 for no limit)"""

    top_p: float = 1
    """Nucleus sampling probability"""

    json_schema: Optional[Dict[str, Any]] = None
    """Optional JSON schema for grammar-constrained decoding (forwarded to
    sglang's xgrammar backend as `sampling_params.json_schema`). When set, the
    model is forced to emit a token sequence that, when decoded, parses as a
    JSON document conforming to this schema. Eliminates the failure mode where
    long reasoning rollouts run out of tokens before emitting the JSON answer."""

    response_format: Optional[Dict[str, Any]] = None
    """E19: OpenAI-shaped response_format alias. Accepts
        {"type": "json_object"}
        {"type": "json_schema", "json_schema": {"schema": {...}}}
    If `json_schema` is None and a schema is present here, it is auto-extracted.
    Held as a separate field so clients can round-trip the original shape."""

    completion_logprobs: bool = False
    """E18: return per-token logprobs for the completion (each entry in
    `SampledSequence.logprobs`). Only the picked-token logprobs are returned,
    not the full distribution. Off by default to save bandwidth."""

    @model_validator(mode="before")
    @classmethod
    def _unwrap_response_format(cls, data):
        if not isinstance(data, dict):
            return data
        rf = data.get("response_format")
        if rf and data.get("json_schema") is None and isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                inner = rf.get("json_schema", {})
                schema = inner.get("schema") if isinstance(inner, dict) else None
                if isinstance(schema, dict):
                    data["json_schema"] = schema
            elif rf.get("type") == "json_object":
                data["json_schema"] = {"type": "object"}
        return data
