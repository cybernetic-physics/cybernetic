"""The openpi pi0.5 client-side codec: collate-dict <-> ``Datum``.

DECISION (mirrors dreamzero): CLIENT-SIDE SERDE. The driver builds the pi0.5
collate dict locally (image stacks, normalized actions/state, tokenized prompt)
and ships it as ONE ``Datum`` per sample. ``Pi05Runtime`` decodes it into an
openpi ``Observation`` + actions and calls ``PI0Pytorch.forward``.

Pure ``numpy``/``torch`` -- NO ``openpi``. The dtype-header + tensor primitives
(``encode_tensor`` / ``decode_tensor`` / ``_tensor_to_numpy``) are REUSED from
``cybernetics.lib.dreamzero.serde`` -- the wire-width convention is identical
(``TensorData.dtype`` is only ``float32`` / ``int64``; a ``<name>.__dtype__``
sibling header restores uint8/bool/int64/float exactly).

WIRE SCHEMA (the pi0.5 collate dict):

  state                  float32 [B, state_dim]           (non-horizoned vector)
  actions                float32 [B, action_horizon, action_dim]
  images                 uint8   [B, V, H, W, 3]           (V cameras, channels-last)
  image_masks            bool    [B, V]
  tokenized_prompt       int64   [B, L]
  tokenized_prompt_mask  bool    [B, L]

Each collate value -> a ``TensorData`` under ``loss_fn_inputs`` keyed by the
collate key, PLUS a sibling ``"<name>.__dtype__"`` header. ``model_input.chunks``
carries exactly ONE ``EncodedTextChunk`` whose ``tokens`` are the flattened
``tokenized_prompt`` ids -- the server chunking heuristic needs a non-zero length;
the authoritative ``tokenized_prompt`` (with its real ND shape) ALSO rides
``loss_fn_inputs``.
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

# pi0.5 default shape contract (mirrors hosted_models.PI05_RECIPE["shape"]).
ACTION_DIM = 32
ACTION_HORIZON = 50
STATE_DIM = 32
MAX_PROMPT_LEN = 200
IMAGE_H = 224
IMAGE_W = 224
NUM_CAMERAS = 3


def _flatten_tokens(value: Any) -> list[int]:
    arr = _to_numpy(value).astype(np.int64, copy=False)
    return arr.flatten().tolist()


def collate_to_datum(
    collate: Mapping[str, Any],
    *,
    extra_loss_inputs: Optional[Mapping[str, Any]] = None,
) -> Datum:
    """Encode a pi0.5 ``collate`` dict into a single wire ``Datum``.

    Every collate key becomes a ``TensorData`` (+ dtype header) under
    ``loss_fn_inputs``; the ``tokenized_prompt`` ids ALSO seed the single
    ``EncodedTextChunk`` length signal. ``extra_loss_inputs`` (driver-computed
    scalars) are encoded the same way.
    """
    if "tokenized_prompt" not in collate:
        raise KeyError("collate dict must contain 'tokenized_prompt' for the chunk length signal")

    loss_fn_inputs: dict[str, TensorData] = {}
    for key, value in collate.items():
        if value is None:
            continue
        loss_fn_inputs.update(encode_tensor(key, value))

    if extra_loss_inputs:
        for key, value in extra_loss_inputs.items():
            loss_fn_inputs.update(encode_tensor(key, value))

    chunk = EncodedTextChunk(tokens=_flatten_tokens(collate["tokenized_prompt"]))
    return Datum(loss_fn_inputs=loss_fn_inputs, model_input=ModelInput(chunks=[chunk]))


encode_sft_datum = collate_to_datum


def datum_to_collate(datum: Any) -> dict[str, np.ndarray]:
    """Decode a wire ``Datum`` back into the pi0.5 ``collate`` dict (numpy).

    Restores every key's ORIGINAL dtype (uint8/bool/int64/float) and ND shape via
    the ``<name>.__dtype__`` headers. Accepts both the pydantic ``Datum`` (client
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


def build_pi05_collate(
    rng: np.random.Generator,
    *,
    action_dim: int = ACTION_DIM,
    action_horizon: int = ACTION_HORIZON,
    state_dim: int = STATE_DIM,
    max_prompt_len: int = MAX_PROMPT_LEN,
    image_h: int = IMAGE_H,
    image_w: int = IMAGE_W,
    num_cameras: int = NUM_CAMERAS,
) -> dict[str, Any]:
    """SYNTHETIC, shape-correct B=1 pi0.5 collate (numpy only, NO openpi).

    Replace this with your real image loader + tokenizer + normalized actions.
    Mirrors ``dreamzero_sft_driver.build_collate``'s structure and the
    int64/float32-only wire widening (uint8 images / bool masks ride the dtype
    header). ``state`` is a single non-horizoned [1, state_dim] vector; ``actions``
    is normalized [-1, 1] [1, action_horizon, action_dim].
    """
    state = rng.standard_normal((1, state_dim)).astype(np.float32)
    actions = np.clip(rng.standard_normal((1, action_horizon, action_dim)), -1.0, 1.0).astype(
        np.float32
    )
    images = rng.integers(0, 256, size=(1, num_cameras, image_h, image_w, 3), dtype=np.uint8)
    image_masks = np.ones((1, num_cameras), dtype=bool)
    tokenized_prompt = rng.integers(0, 32000, size=(1, max_prompt_len), dtype=np.int64)
    tokenized_prompt_mask = np.ones((1, max_prompt_len), dtype=bool)
    return {
        "state": state,
        "actions": actions,
        "images": images,
        "image_masks": image_masks,
        "tokenized_prompt": tokenized_prompt,
        "tokenized_prompt_mask": tokenized_prompt_mask,
    }
