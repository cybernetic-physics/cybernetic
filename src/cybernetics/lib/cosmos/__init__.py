"""Client-side NVIDIA Cosmos 3 omni (Cosmos3-Nano) <-> Cybernetics serde.

This package is the cosmos sibling of ``cybernetics.lib.dreamzero`` /
``cybernetics.lib.pi05``: a driver builds a cosmos ``collate`` dict locally (a
synthetic video clip + a prompt) and ships it as ONE ``Datum`` per sample;
``CosmosRuntime`` decodes it into a ``(video, prompt)`` pair, runs the pipeline's
proven VAE-encode path to obtain the clean latents, and computes the
rectified-flow velocity MSE on the Cosmos3OmniTransformer. No cosmos dataset is
instantiated server-side.

It lives under ``cybernetics/lib/cosmos/`` so BOTH the client (``cybernetics``
editable dep) AND the backend import the SAME codec. No ``diffusers`` /
``diffusers_cosmos3`` and no GPU are required here: the serde is pure ``numpy``.
The dtype-header + tensor primitives (``encode_tensor`` / ``decode_tensor`` /
``<name>.__dtype__``) are REUSED from ``cybernetics.lib.dreamzero.serde`` (the
wire-width convention is identical: float32 / int64 only, with a
``<name>.__dtype__`` header restoring uint8/bool/int64/float exactly).
"""

from __future__ import annotations

#: The base_model identifier the runtime self-dispatches on.
COSMOS_BASE_MODEL = "cosmos3-nano"

#: Reused from lib.dreamzero: int64 codes for the ``<name>.__dtype__`` header.
from cybernetics.lib.dreamzero import (  # noqa: E402
    DTYPE_CODE_BOOL,
    DTYPE_CODE_FLOAT32,
    DTYPE_CODE_INT64,
    DTYPE_CODE_UINT8,
    DTYPE_HEADER_SUFFIX,
)

#: cosmos collate keys carried as uint8/int64 on the wire (header restores).
COSMOS_INT_KEYS = (
    "video",
    "prompt",
)

__all__ = [
    "COSMOS_BASE_MODEL",
    "COSMOS_INT_KEYS",
    "DTYPE_CODE_BOOL",
    "DTYPE_CODE_FLOAT32",
    "DTYPE_CODE_INT64",
    "DTYPE_CODE_UINT8",
    "DTYPE_HEADER_SUFFIX",
]
