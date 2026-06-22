"""Client-side DreamZero <-> Cybernetics serde and pure transform helpers.

This package is the home of the CLIENT-SIDE SERDE: the driver rebuilds the
``dreamzero_cotrain.collate`` dict locally (umt5 tokenization, relative-action
transform, normalization) and ships it over the wire as a single ``Datum`` per
sample; the runtime decodes it back into the collate dict and calls
``VLA.forward``. No dreamzero dataset is instantiated server-side.

It lives under ``packages/client/src/worldlines/lib/dreamzero/`` so that BOTH the
client (``worldlines`` editable dep) AND the backend import the SAME codec, per
the ``[tool.uv.sources]`` wiring in the backend ``pyproject.toml``.

No ``groot`` and no GPU are required here: the serde is pure ``numpy``/``torch``.
"""

from __future__ import annotations

#: The base_model identifier the runtime self-dispatches on.
DREAMZERO_DROID_BASE_MODEL = "dreamzero-droid"

#: int64 codes written into the ``<name>.__dtype__`` sibling header TensorData so
#: that ``decode_sft_batch`` can restore the ORIGINAL numpy/torch dtype exactly
#: (TensorData.dtype is only ``int64``/``float32`` -- see tensor_dtype.py).
DTYPE_CODE_FLOAT32 = 0
DTYPE_CODE_INT64 = 1
DTYPE_CODE_UINT8 = 2
DTYPE_CODE_BOOL = 3

#: Suffix for the per-key sibling dtype-header TensorData.
DTYPE_HEADER_SUFFIX = "__dtype__"

#: Collate keys whose ORIGINAL semantic dtype is floating point. Everything in
#: SFT_FLOAT_KEYS is carried as float32 TensorData (no header needed, but a
#: header is still emitted for symmetry).
SFT_FLOAT_KEYS = (
    "state",
    "action",
    "lapa_action",
    "segmentation_target",
    "segmentation_target_mask",
)

#: Collate keys whose ORIGINAL semantic dtype is integer / boolean / uint8.
#: These are flattened to int64 on the wire and restored via the dtype header.
SFT_INT_KEYS = (
    "images",
    "text",
    "text_attention_mask",
    "text_negative",
    "text_attention_mask_negative",
    "state_mask",
    "action_mask",
    "lapa_action_mask",
    "embodiment_id",
    "has_real_action",
    "has_lapa_action",
    "is_cotrain_instance",
)

#: RL-trajectory float keys (per-SDE-step latents + scalars), see serde schema.
RL_FLOAT_KEYS = (
    "x_t",
    "x_prev",
    "mu_old",
    "sigma",
    "log_prob_old",
    "t",
    "dt",
    "advantages",
)

#: RL-trajectory int keys.
RL_INT_KEYS = (
    "group_ids",
    "trajectory_ids",
    "token_mask",
)

__all__ = [
    "DREAMZERO_DROID_BASE_MODEL",
    "DTYPE_CODE_FLOAT32",
    "DTYPE_CODE_INT64",
    "DTYPE_CODE_UINT8",
    "DTYPE_CODE_BOOL",
    "DTYPE_HEADER_SUFFIX",
    "SFT_FLOAT_KEYS",
    "SFT_INT_KEYS",
    "RL_FLOAT_KEYS",
    "RL_INT_KEYS",
]
