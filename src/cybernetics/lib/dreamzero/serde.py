"""The DreamZero client-side codec: collate-dict <-> ``Datum``.

DECISION LOCKED: CLIENT-SIDE SERDE. The driver builds the
``dreamzero_cotrain.collate`` dict locally (umt5 tokenization, relative-action
transform, normalization) and ships it as ONE ``Datum`` per sample. The runtime
decodes it back into the collate dict and calls ``VLA.forward``. No dreamzero
dataset is instantiated server-side.

Pure ``numpy``/``torch`` -- NO ``groot``. ``torch`` is lazy-imported only inside
the functions that accept / return ``torch.Tensor``; the numpy path needs no
torch, so this module imports cleanly with torch absent.

WIRE SCHEMA (the ~14-key collate dict from ``dreamzero_cotrain.collate`` +
``apply_single``):

Each collate value -> a ``TensorData`` under ``loss_fn_inputs`` keyed by the
collate key, PLUS a sibling ``"<name>.__dtype__"`` int64-scalar header TensorData
carrying the ORIGINAL dtype code (0=float32, 1=int64, 2=uint8, 3=bool) so decode
restores uint8/bool/int64/float exactly -- ``TensorData.dtype`` is only
``int64``/``float32`` (tensor_dtype.py), so uint8 images and bool masks are
flattened to int64 losslessly and the header reconstructs them.

``model_input.chunks`` carries exactly ONE ``EncodedTextChunk`` whose ``tokens``
are the flattened umt5 ``text`` input_ids -- this gives the server chunking
heuristic a non-zero length. The authoritative ``text`` tensor (with its real ND
shape) ALSO rides ``loss_fn_inputs`` because the chunk only carries a flat token
list.

``world_model_only=True`` OMITS the ``action``/``action_mask`` keys (and their
headers) entirely; the head then sees ``actions.numel() == 0`` and computes the
dynamics loss only.

Shapes ride ``TensorData.shape`` verbatim: variable ``num_chunks`` is honored
(video ``8m+1`` frames, action ``24n``, state ``n``; action padded to 32, state
to 44). NO hardcoded ``(24, 8)`` / ``33``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from cybernetics.types.datum import Datum
from cybernetics.types.encoded_text_chunk import EncodedTextChunk
from cybernetics.types.model_input import ModelInput
from cybernetics.types.tensor_data import TensorData

from . import (
    DTYPE_CODE_BOOL,
    DTYPE_CODE_FLOAT32,
    DTYPE_CODE_INT64,
    DTYPE_CODE_UINT8,
    DTYPE_HEADER_SUFFIX,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch

DTYPE_HEADER_KEY = DTYPE_HEADER_SUFFIX
ORIG_DTYPE_HEADER_KEY = DTYPE_HEADER_SUFFIX

# Default dreamzero DROID shape contract (overridable via encode/decode kwargs).
_DEFAULT_ACTION_HORIZON = 24
_DEFAULT_NUM_FRAME_PER_BLOCK = 2

_ACTION_KEYS = ("action", "action_mask")

# numpy-dtype-kind -> (wire TensorDtype, original dtype code).
_NUMPY_TO_CODE = {
    "f": (np.float32, DTYPE_CODE_FLOAT32),
    "i": (np.int64, DTYPE_CODE_INT64),
    "u": (np.int64, DTYPE_CODE_UINT8),
    "b": (np.int64, DTYPE_CODE_BOOL),
}

_CODE_TO_NUMPY = {
    DTYPE_CODE_FLOAT32: np.float32,
    DTYPE_CODE_INT64: np.int64,
    DTYPE_CODE_UINT8: np.uint8,
    DTYPE_CODE_BOOL: np.bool_,
}


def _to_numpy(value: Any) -> np.ndarray:
    """Coerce a torch.Tensor / numpy array / python scalar-or-list to ndarray.

    torch is imported lazily so the numpy path needs no torch.
    """
    if isinstance(value, np.ndarray):
        return value
    # Duck-type torch.Tensor without importing torch eagerly.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# Wire dtype string -> numpy dtype (the ONLY two TensorDtype values, tensor_dtype.py).
_WIRE_DTYPE_TO_NUMPY = {"float32": np.float32, "int64": np.int64}


def _datum_field(datum: Any, name: str) -> Any:
    """Read ``datum.<name>`` whether ``datum`` is a pydantic ``Datum`` or a plain
    wire dict.

    At the runtime boundary the datum is a JSON-deserialized PLAIN DICT (the
    backend wire ``Datum`` is a ``TypedDict`` and ``app.py`` passes
    ``forward_backward_input['data']`` to the runtime with no pydantic coercion),
    so attribute access on it would raise ``AttributeError``. The build/client
    side, by contrast, constructs a pydantic ``Datum``. Support both.
    """
    if isinstance(datum, Mapping):
        return datum[name]
    return getattr(datum, name)


def _tensor_to_numpy(td: Any) -> np.ndarray:
    """Materialize a numpy array from a ``TensorData`` (pydantic model OR the plain
    wire dict ``{data, dtype, shape}``) WITHOUT relying on ``TensorData.to_numpy``.

    The wire ``TensorData`` is a ``{data, dtype, shape}`` dict at the runtime
    boundary; the client side holds the pydantic model. Decode both symmetrically.
    """
    if isinstance(td, Mapping):
        data = td["data"]
        dtype = td["dtype"]
        shape = td.get("shape")
    else:
        data = td.data
        dtype = td.dtype
        shape = td.shape
    np_dtype = _WIRE_DTYPE_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported wire TensorDtype {dtype!r}")
    arr = np.asarray(data, dtype=np_dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def encode_tensor(name: str, array_or_tensor: Any) -> dict[str, TensorData]:
    """Encode one named tensor into its value + ``<name>.__dtype__`` header pair.

    uint8/bool -> int64 wire (lossless), floats -> float32, ints -> int64. The
    header records the ORIGINAL dtype code so :func:`decode_tensor` restores it.
    Returns ``{name: TensorData, "<name>.__dtype__": TensorData}``.
    """
    arr = _to_numpy(array_or_tensor)
    kind = arr.dtype.kind
    if kind not in _NUMPY_TO_CODE:
        raise TypeError(f"{name}: unsupported dtype {arr.dtype} (kind {kind!r})")
    wire_np_dtype, code = _NUMPY_TO_CODE[kind]
    wire = arr.astype(wire_np_dtype, copy=False)
    value_td = TensorData(
        data=wire.flatten().tolist(),
        dtype="float32" if code == DTYPE_CODE_FLOAT32 else "int64",
        shape=list(arr.shape),
    )
    header_td = TensorData(data=[code], dtype="int64", shape=[])
    return {name: value_td, f"{name}.{DTYPE_HEADER_SUFFIX}": header_td}


def decode_tensor(tensor_data: Any, code: int) -> np.ndarray:
    """Restore the ORIGINAL-dtype numpy array from a wire ``TensorData`` + code.

    ``tensor_data`` may be a pydantic ``TensorData`` OR the plain wire dict
    ``{data, dtype, shape}`` the runtime actually receives.
    """
    if code not in _CODE_TO_NUMPY:
        raise ValueError(f"unknown dtype code {code}")
    arr = _tensor_to_numpy(tensor_data)
    return arr.astype(_CODE_TO_NUMPY[code], copy=False)


def _flatten_tokens(text_value: Any) -> list[int]:
    arr = _to_numpy(text_value).astype(np.int64, copy=False)
    return arr.flatten().tolist()


def collate_to_datum(
    collate: Mapping[str, Any],
    *,
    world_model_only: bool = False,
    extra_loss_inputs: Optional[Mapping[str, Any]] = None,
) -> Datum:
    """Encode a dreamzero ``collate`` dict into a single wire ``Datum``.

    Every collate key becomes a ``TensorData`` (+ dtype header) under
    ``loss_fn_inputs``; the ``text`` input_ids ALSO seed the single
    ``EncodedTextChunk`` length signal. ``world_model_only=True`` omits
    ``action``/``action_mask`` so the head runs dynamics-only.
    ``extra_loss_inputs`` (e.g. driver-computed scalars) are encoded the same way.
    """
    if "text" not in collate:
        raise KeyError(
            "collate dict must contain 'text' (umt5 input_ids) for the chunk length signal"
        )

    loss_fn_inputs: dict[str, TensorData] = {}
    for key, value in collate.items():
        if world_model_only and key in _ACTION_KEYS:
            continue
        if value is None:
            continue
        loss_fn_inputs.update(encode_tensor(key, value))

    if extra_loss_inputs:
        for key, value in extra_loss_inputs.items():
            loss_fn_inputs.update(encode_tensor(key, value))

    chunk = EncodedTextChunk(tokens=_flatten_tokens(collate["text"]))
    return Datum(loss_fn_inputs=loss_fn_inputs, model_input=ModelInput(chunks=[chunk]))


# Build-stage public alias.
encode_sft_datum = collate_to_datum


def _validate_chunk_divisibility(
    decoded: Mapping[str, np.ndarray],
    *,
    action_horizon: int,
    num_frames: Optional[int] = None,
    num_frame_per_block: int = _DEFAULT_NUM_FRAME_PER_BLOCK,
) -> None:
    """Assert variable-num_chunks structural invariants (no hardcoded 24/8/33).

    The action token count must be a multiple of ``action_horizon`` (24n). The
    raw video carries ``8m+1`` frames (e.g. the canonical 33 = 8*4+1), so neither
    the raw frame count nor the VAE latent frame count is a clean multiple of
    ``num_frame_per_block`` -- the causal per-block invariant lives on the interior
    latent frames and only the server-side VAE+DiT can check it. We therefore
    validate ONLY the client-checkable action-token invariant here. ``num_frames``
    / ``num_frame_per_block`` are accepted for API symmetry but are NOT asserted
    against each other (the old ``num_frames % num_frame_per_block`` check wrongly
    rejected the canonical 33/2 recipe).
    """
    del num_frames, num_frame_per_block  # accepted for API symmetry; see docstring
    if "action" in decoded:
        # Token axis is shape[-2] for BOTH the per-sample [T, action_dim] and the
        # batched [B, T, action_dim] convention. Producers ship the B=1-stacked
        # collate dict that dreamzero_cotrain.collate's np.stack yields, so a
        # leading batch axis is present; validating shape[0] would wrongly reject
        # batched action ([1, 24, 32] -> 1 % 24 != 0).
        action_len = decoded["action"].shape[-2]
        if action_len % action_horizon != 0:
            raise ValueError(
                f"action length {action_len} not divisible by action_horizon {action_horizon}"
            )


def datum_to_collate(
    datum: Any,
    *,
    action_horizon: int = _DEFAULT_ACTION_HORIZON,
    num_frames: Optional[int] = None,
    num_frame_per_block: int = _DEFAULT_NUM_FRAME_PER_BLOCK,
    validate: bool = True,
) -> dict[str, np.ndarray]:
    """Decode a wire ``Datum`` back into the dreamzero ``collate`` dict (numpy).

    Restores every key's ORIGINAL dtype (uint8/bool/int64/float) and ND shape via
    the ``<name>.__dtype__`` headers. The action keys are simply absent for a
    world-model-only datum.

    Accepts both the pydantic ``Datum`` (client/build side) AND the plain wire
    dict the runtime receives -- at the runtime boundary ``loss_fn_inputs`` is a
    plain ``dict[str, {data, dtype, shape}]``, not a pydantic model.
    """
    inputs = _datum_field(datum, "loss_fn_inputs")
    decoded: dict[str, np.ndarray] = {}
    for key, td in inputs.items():
        if key.endswith(f".{DTYPE_HEADER_SUFFIX}"):
            continue
        header_key = f"{key}.{DTYPE_HEADER_SUFFIX}"
        if header_key not in inputs:
            raise KeyError(f"missing dtype header '{header_key}' for value key '{key}'")
        code = int(np.asarray(_tensor_to_numpy(inputs[header_key])).reshape(-1)[0])
        decoded[key] = decode_tensor(td, code)

    if validate:
        _validate_chunk_divisibility(
            decoded,
            action_horizon=action_horizon,
            num_frames=num_frames,
            num_frame_per_block=num_frame_per_block,
        )
    return decoded


def decode_sft_batch(
    datum: Any,
    *,
    action_horizon: int = _DEFAULT_ACTION_HORIZON,
    num_frames: Optional[int] = None,
    num_frame_per_block: int = _DEFAULT_NUM_FRAME_PER_BLOCK,
    validate: bool = True,
) -> dict[str, "torch.Tensor"]:
    """Decode a wire ``Datum`` into a ``{key: torch.Tensor}`` collate batch.

    Same as :func:`datum_to_collate` but returns torch tensors with the original
    dtypes (uint8/bool/long/float32), ready to feed ``VLA.forward``. torch is
    imported lazily here. Accepts both the pydantic ``Datum`` and the plain wire
    dict the runtime receives.

    BATCH AXIS. serde is shape-transparent: it round-trips ``TensorData.shape``
    verbatim and adds NO axis. The producer (driver / test fixture) ships the
    already-batched ``B == 1`` collate dict that ``dreamzero_cotrain.collate``
    yields -- ``np.stack`` (dreamzero_cotrain.py:162-163) prepends a leading axis-0
    of size ``B`` over the per-sample ``apply_single`` arrays, and the umt5
    tokenizer emits ``text`` / ``text_attention_mask`` already as ``[B, seq]``. So
    the wire carries ``action [1, action_horizon, max_action_dim]``, ``state [1,
    state_horizon, max_state_dim]``, ``images [1, num_frames, image_h, image_w, 3]``,
    the scalar flags as ``[1]``, and ``text [1, seq]``; this decoder returns exactly
    those batched ranks, ready for ``VLA.forward`` / ``WANPolicyHead.forward``. The
    client adds the batch axis AFTER the per-sample ``relative_actions.to_relative``
    / ``normalize`` (which require 2-D action, relative_actions.py:122-125), i.e. as
    the final ``np.stack`` step -- never inside this RL-shared codec. The
    action-token divisibility check tolerates both the per-sample ``[T, dim]`` and
    batched ``[B, T, dim]`` ranks (token axis ``-2``).
    """
    import torch  # lazy: numpy path (datum_to_collate) needs no torch

    np_batch = datum_to_collate(
        datum,
        action_horizon=action_horizon,
        num_frames=num_frames,
        num_frame_per_block=num_frame_per_block,
        validate=validate,
    )
    out: dict[str, torch.Tensor] = {}
    for key, arr in np_batch.items():
        # ascontiguousarray promotes 0-d -> (1,); reshape back to keep scalar rank.
        out[key] = torch.from_numpy(np.ascontiguousarray(arr).reshape(arr.shape))
    return out
