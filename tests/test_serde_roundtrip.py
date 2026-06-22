"""Shared-contract test: the dual-homed DreamZero serde encodes Datums that the
backend codec can decode. ``serde.py`` ships in the SDK *and* runs in the hosted
backend, so an encode -> decode round-trip pins the wire contract (dtype + shape +
values + the ``world_model_only`` action-omission path).
"""

from __future__ import annotations

import numpy as np

from cybernetics.lib.dreamzero import serde


def _collate() -> dict[str, np.ndarray]:
    return {
        "text": np.array([1, 2, 3, 4], dtype=np.int64),  # umt5 input_ids -> chunk length signal
        "state": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "images": np.zeros((1, 2, 2), dtype=np.uint8),
        "action": np.zeros((1, 24, 4), dtype=np.float32),  # 24 == default action_horizon
        "action_mask": np.array([[True, False]], dtype=np.bool_),
    }


def test_encode_decode_round_trip_preserves_dtype_shape_values() -> None:
    collate = _collate()
    datum = serde.collate_to_datum(collate)

    # Exactly one placeholder EncodedTextChunk (a length signal), not multiple chunks.
    assert len(datum.model_input.chunks) == 1

    decoded = serde.datum_to_collate(datum)
    for key in ("state", "images", "action", "action_mask"):
        assert decoded[key].dtype == collate[key].dtype, key
        assert decoded[key].shape == collate[key].shape, key
        assert np.array_equal(decoded[key], collate[key]), key


def test_world_model_only_omits_action_keys() -> None:
    datum = serde.collate_to_datum(_collate(), world_model_only=True)
    decoded = serde.datum_to_collate(datum, validate=False)
    assert "action" not in decoded
    assert "action_mask" not in decoded
    assert "state" in decoded  # non-action tensors still ride


def test_each_value_key_has_a_dtype_header() -> None:
    datum = serde.collate_to_datum(_collate())
    keys = set(datum.loss_fn_inputs)
    for value_key in ("state", "images", "action"):
        assert f"{value_key}.{serde.DTYPE_HEADER_SUFFIX}" in keys, value_key
