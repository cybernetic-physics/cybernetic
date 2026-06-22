"""The Cosmos 3 omni client-side codec: collate-dict <-> ``Datum``.

DECISION (mirrors dreamzero / pi05): CLIENT-SIDE SERDE. The driver builds the
cosmos collate dict locally (a video clip + a prompt) and ships it as ONE
``Datum`` per sample. ``CosmosRuntime`` decodes it into a ``(video, prompt)``
pair, runs the pipeline's proven VAE-encode path to get the clean latents, and
computes the rectified-flow velocity MSE on the Cosmos3OmniTransformer.

Pure ``numpy`` -- NO ``diffusers`` / ``diffusers_cosmos3``. The dtype-header +
tensor primitives (``encode_tensor`` / ``decode_tensor`` / ``_tensor_to_numpy``)
are REUSED from ``cybernetics.lib.dreamzero.serde`` -- the wire-width convention
is identical (``TensorData.dtype`` is only ``float32`` / ``int64``; a
``<name>.__dtype__`` sibling header restores uint8/bool/int64/float exactly).

WIRE SCHEMA (the cosmos collate dict):

  video                  uint8 [1, T, H, W, 3]   (NTHWC, channels-last, one clip)
  prompt                 int64 [1, L]            (tokenized caption ids)

Each collate value -> a ``TensorData`` under ``loss_fn_inputs`` keyed by the
collate key, PLUS a sibling ``"<name>.__dtype__"`` header. ``model_input.chunks``
carries exactly ONE ``EncodedTextChunk`` whose ``tokens`` are the flattened
``prompt`` ids -- the server chunking heuristic needs a non-zero length; the
authoritative ``prompt`` (with its real ND shape) ALSO rides ``loss_fn_inputs``.
The runtime detokenizes ``prompt`` back to text via the pipeline tokenizer.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from cybernetics.lib.dreamzero.serde import (
    _datum_field,
    _to_numpy,
    decode_tensor,
    encode_tensor,
)
from cybernetics.types.datum import Datum
from cybernetics.types.encoded_text_chunk import EncodedTextChunk
from cybernetics.types.model_input import ModelInput
from cybernetics.types.tensor_data import TensorData

from . import DTYPE_HEADER_SUFFIX

# cosmos default shape contract (mirrors hosted_models.COSMOS_NANO_RECIPE["shape"]).
NUM_FRAMES = 5
HEIGHT = 256
WIDTH = 256
PROMPT_LEN = 32


def _flatten_tokens(value: Any) -> list[int]:
    arr = _to_numpy(value).astype(np.int64, copy=False)
    return arr.flatten().tolist()


def collate_to_datum(
    collate: Mapping[str, Any],
    *,
    extra_loss_inputs: Optional[Mapping[str, Any]] = None,
) -> Datum:
    """Encode a cosmos ``collate`` dict into a single wire ``Datum``.

    Every collate key becomes a ``TensorData`` (+ dtype header) under
    ``loss_fn_inputs``; the ``prompt`` ids ALSO seed the single
    ``EncodedTextChunk`` length signal. ``extra_loss_inputs`` (driver-computed
    scalars) are encoded the same way.
    """
    if "prompt" not in collate:
        raise KeyError("collate dict must contain 'prompt' for the chunk length signal")

    loss_fn_inputs: dict[str, TensorData] = {}
    for key, value in collate.items():
        if value is None:
            continue
        loss_fn_inputs.update(encode_tensor(key, value))

    if extra_loss_inputs:
        for key, value in extra_loss_inputs.items():
            loss_fn_inputs.update(encode_tensor(key, value))

    chunk = EncodedTextChunk(tokens=_flatten_tokens(collate["prompt"]))
    return Datum(loss_fn_inputs=loss_fn_inputs, model_input=ModelInput(chunks=[chunk]))


encode_sft_datum = collate_to_datum


def datum_to_collate(datum: Any) -> dict[str, np.ndarray]:
    """Decode a wire ``Datum`` back into the cosmos ``collate`` dict (numpy).

    Restores every key's ORIGINAL dtype (uint8/int64) and ND shape via the
    ``<name>.__dtype__`` headers. Accepts both the pydantic ``Datum`` (client
    side) AND the plain wire dict the runtime receives.
    """
    inputs = _datum_field(datum, "loss_fn_inputs")
    decoded: dict[str, np.ndarray] = {}
    for key, td in inputs.items():
        if key.endswith(f".{DTYPE_HEADER_SUFFIX}"):
            continue
        header_key = f"{key}.{DTYPE_HEADER_SUFFIX}"
        if header_key not in inputs:
            raise KeyError(f"missing dtype header '{header_key}' for value key '{key}'")
        from cybernetics.lib.dreamzero.serde import _tensor_to_numpy

        code = int(np.asarray(_tensor_to_numpy(inputs[header_key])).reshape(-1)[0])
        decoded[key] = decode_tensor(td, code)
    return decoded


def build_cosmos_collate(
    rng: np.random.Generator,
    *,
    num_frames: int = NUM_FRAMES,
    height: int = HEIGHT,
    width: int = WIDTH,
    prompt_len: int = PROMPT_LEN,
) -> dict[str, Any]:
    """SYNTHETIC, shape-correct B=1 cosmos collate (numpy only, NO diffusers).

    Replace this with your real video loader + the Qwen2 tokenizer. Mirrors
    ``build_pi05_collate``'s structure and the int64-only wire widening (uint8
    video / int64 prompt ride the dtype header). ``video`` is uint8
    ``[1, T, H, W, 3]`` (NTHWC, channels-last); ``prompt`` is int64 ``[1, L]``
    token ids. The runtime normalizes the video to [-1, 1] and detokenizes the
    prompt ids via the pipeline tokenizer.
    """
    video = rng.integers(0, 256, size=(1, num_frames, height, width, 3), dtype=np.uint8)
    prompt = rng.integers(0, 150000, size=(1, prompt_len), dtype=np.int64)
    return {
        "video": video,
        "prompt": prompt,
    }
