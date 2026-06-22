"""Client-side openpi pi0.5 <-> Cybernetics serde.

This package is the pi0.5 sibling of ``cybernetics.lib.dreamzero``: the driver
builds a pi0.5 ``collate`` dict locally (image stacks, normalized actions/state,
tokenized prompt) and ships it as ONE ``Datum`` per sample; the runtime decodes
it into an openpi ``Observation`` + actions and calls ``PI0Pytorch.forward``. No
openpi dataset is instantiated server-side.

It lives under ``cybernetics/lib/pi05/`` so BOTH the client (``cybernetics``
editable dep) AND the backend import the SAME codec. No ``openpi`` and no GPU are
required here: the serde is pure ``numpy``/``torch``. The dtype header + tensor
primitives are reused from ``lib.dreamzero.serde`` (the wire-width convention is
identical: float32 / int64 only, with a ``<name>.__dtype__`` header restoring
uint8/bool/int64/float).
"""

from __future__ import annotations

#: The base_model identifier the runtime self-dispatches on.
PI05_BASE_MODEL = "pi0.5"

#: Reused from lib.dreamzero: int64 codes for the ``<name>.__dtype__`` header.
from cybernetics.lib.dreamzero import (  # noqa: E402
    DTYPE_CODE_BOOL,
    DTYPE_CODE_FLOAT32,
    DTYPE_CODE_INT64,
    DTYPE_CODE_UINT8,
    DTYPE_HEADER_SUFFIX,
)

#: pi0.5 collate keys whose ORIGINAL semantic dtype is floating point.
PI05_FLOAT_KEYS = ("state", "actions")

#: pi0.5 collate keys carried as int64/bool/uint8 on the wire (header restores).
PI05_INT_KEYS = (
    "images",
    "image_masks",
    "tokenized_prompt",
    "tokenized_prompt_mask",
)

__all__ = [
    "DTYPE_CODE_BOOL",
    "DTYPE_CODE_FLOAT32",
    "DTYPE_CODE_INT64",
    "DTYPE_CODE_UINT8",
    "DTYPE_HEADER_SUFFIX",
    "PI05_BASE_MODEL",
    "PI05_FLOAT_KEYS",
    "PI05_INT_KEYS",
]
