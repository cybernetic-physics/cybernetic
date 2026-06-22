"""Pure-numpy ``droid_relative`` action transform + q99 normalization.

This is the client-side reproduction of the dreamzero relative-action pipeline.
Both halves are pure functions that take the stats dict in -- no ``groot``, no
GPU, no torch required.

WHY THIS MODULE EXISTS (MUST-NOT-REGRESS):
A naive serde that ships ABSOLUTE actions normalized by the absolute stats trains
garbage that still "runs". The dreamzero DROID recipe (``droid_relative.yaml``)
trains on RELATIVE deltas ``action - state[anchor]`` per ``action_horizon`` chunk
on the ``joint_position`` slice, normalized by the SEPARATE
``meta/relative_stats_dreamzero.json`` (NOT the absolute stats). This module owns
that transform so the driver applies it BEFORE ``encode_sft_datum``.

Relative-delta math (authoritative source:
``scripts/data/convert_lerobot_to_gear.py:compute_relative_stats`` and
``groot/vla/data/dataset/lerobot.py:_calculate_relative_stats_for_key``)::

    for each chunk starting at index i (step action_horizon):
        ref_state = state[i]                 # the anchor proprio
        relative  = action[i : i+H] - ref_state

q99 normalization (authoritative source:
``groot/vla/data/transform/state_action.py`` ``mode == "q99"``)::

    forward : n = clamp(2 * (x - q01) / (q99 - q01) - 1, -1, 1)   (q01==q99 -> x)
    inverse : x = (n + 1) / 2 * (q99 - q01) + q01
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# The q99 stats ship inside the package at this location when available. The data
# itself is produced by the hosted DreamZero data pipeline and is dropped in here
# at build/release time; the SDK never fabricates statistics.
RELATIVE_STATS_FILENAME = "relative_stats_dreamzero.json"
_RELATIVE_STATS_PACKAGE = "cybernetics.lib.dreamzero.meta"


class RelativeStatsUnavailable(FileNotFoundError):
    """The bundled ``relative_stats_dreamzero.json`` is not present.

    The q99 normalize/unnormalize path needs this data; pass an explicit ``path``
    to :func:`load_relative_stats`, or install a release whose wheel bundles
    ``cybernetics/lib/dreamzero/meta/relative_stats_dreamzero.json``.
    """


def default_relative_stats_path() -> Path | None:
    """Return the bundled stats path if it ships in this install, else ``None``."""

    try:
        resource = resources.files(_RELATIVE_STATS_PACKAGE) / RELATIVE_STATS_FILENAME
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    path = Path(str(resource))
    return path if path.is_file() else None


@dataclass(frozen=True)
class RelativeStats:
    """Per-key relative-action statistics (one entry per joint dim).

    Mirrors the schema of ``meta/relative_stats_dreamzero.json``: a dict keyed by
    relative-action key (e.g. ``"joint_position"``) whose value carries
    ``max/min/mean/std/q01/q99`` lists of length ``= len(joint_slice)``.
    """

    q01: np.ndarray
    q99: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    @classmethod
    def from_mapping(cls, stats: Mapping[str, Any]) -> "RelativeStats":
        def _arr(name: str) -> np.ndarray:
            return np.asarray(stats[name], dtype=np.float64)

        return cls(
            q01=_arr("q01"),
            q99=_arr("q99"),
            mean=_arr("mean"),
            std=_arr("std"),
            min=_arr("min"),
            max=_arr("max"),
        )


def load_relative_stats(path: str | Path | None = None) -> dict[str, RelativeStats]:
    """Load ``relative_stats_dreamzero.json`` into ``{key: RelativeStats}``.

    With ``path=None`` the bundled stats are used; raises
    :class:`RelativeStatsUnavailable` if neither an explicit path nor a bundled
    file is present (the q99 path has no data to work with).
    """
    resolved = Path(path) if path is not None else default_relative_stats_path()
    if resolved is None:
        raise RelativeStatsUnavailable(
            f"no {RELATIVE_STATS_FILENAME} available: pass an explicit path or install a "
            f"release that bundles {_RELATIVE_STATS_PACKAGE.replace('.', '/')}/{RELATIVE_STATS_FILENAME}"
        )
    with open(resolved) as fh:
        raw = json.load(fh)
    return {key: RelativeStats.from_mapping(val) for key, val in raw.items()}


def _resolve_slice(joint_slice: Any, n_dims: int) -> slice:
    if joint_slice is None:
        return slice(0, n_dims)
    if isinstance(joint_slice, slice):
        return joint_slice
    start, stop = joint_slice
    return slice(start, stop)


def _anchor_index(c: int, anchor_index: int, state_per_chunk: int, n_state: int) -> int:
    """Map chunk ``c`` to its anchor row in the per-chunk ``state`` array.

    The collate ``state`` holds ``state_per_chunk`` proprio tokens per action
    chunk (``num_state_per_block``, default 1), so chunk ``c``'s anchor is
    ``c * state_per_chunk + anchor_index``.
    """
    ref_idx = c * state_per_chunk + anchor_index
    if ref_idx >= n_state:
        raise IndexError(f"anchor state index {ref_idx} out of range for state of length {n_state}")
    return ref_idx


def to_relative(
    action: np.ndarray,
    state: np.ndarray,
    *,
    anchor_index: int = 0,
    joint_slice: Any = None,
    action_horizon: int = 24,
    state_per_chunk: int = 1,
) -> np.ndarray:
    """Convert absolute actions to per-chunk relative deltas on ``joint_slice``.

    ``action`` is ``[H_total, action_dim]`` with ``H_total % action_horizon == 0``.
    ``state`` is the per-chunk proprio ``[n_state, state_dim]`` with
    ``state_per_chunk`` tokens per action chunk (``num_state_per_block``). For each
    ``action_horizon``-frame chunk, subtract the chunk's anchor state's
    ``joint_slice`` columns from the corresponding action columns. Channels outside
    ``joint_slice`` are passed through unchanged (e.g. the gripper). The returned
    array is a copy; ``action`` is not mutated.
    """
    action = np.asarray(action)
    state = np.asarray(state)
    if action.ndim != 2:
        raise ValueError(f"action must be [H_total, action_dim], got {action.shape}")
    if state.ndim != 2:
        raise ValueError(f"state must be [n_state, state_dim], got {state.shape}")
    h_total = action.shape[0]
    if h_total % action_horizon != 0:
        raise ValueError(
            f"action length {h_total} not divisible by action_horizon {action_horizon}"
        )

    sl = _resolve_slice(joint_slice, action.shape[1])
    rel = action.astype(np.float64, copy=True)
    n_chunks = h_total // action_horizon
    for c in range(n_chunks):
        ref_idx = _anchor_index(c, anchor_index, state_per_chunk, state.shape[0])
        ref = state[ref_idx, sl]
        lo = c * action_horizon
        hi = lo + action_horizon
        rel[lo:hi, sl] = action[lo:hi, sl] - ref
    return rel.astype(action.dtype, copy=False) if action.dtype.kind == "f" else rel


def from_relative(
    rel_action: np.ndarray,
    state: np.ndarray,
    *,
    anchor_index: int = 0,
    joint_slice: Any = None,
    action_horizon: int = 24,
    state_per_chunk: int = 1,
) -> np.ndarray:
    """Invert :func:`to_relative`: add the chunk anchor state back per chunk.

    ``to_relative`` then ``from_relative`` with the same anchor/slice/state is the
    identity (within float round-off).
    """
    rel_action = np.asarray(rel_action)
    state = np.asarray(state)
    if rel_action.ndim != 2:
        raise ValueError(f"rel_action must be [H_total, action_dim], got {rel_action.shape}")
    h_total = rel_action.shape[0]
    if h_total % action_horizon != 0:
        raise ValueError(
            f"action length {h_total} not divisible by action_horizon {action_horizon}"
        )

    sl = _resolve_slice(joint_slice, rel_action.shape[1])
    out = rel_action.astype(np.float64, copy=True)
    n_chunks = h_total // action_horizon
    for c in range(n_chunks):
        ref_idx = _anchor_index(c, anchor_index, state_per_chunk, state.shape[0])
        ref = state[ref_idx, sl]
        lo = c * action_horizon
        hi = lo + action_horizon
        out[lo:hi, sl] = rel_action[lo:hi, sl] + ref
    return out


def normalize(action: np.ndarray, stats: RelativeStats) -> np.ndarray:
    """q99-normalize ``action`` to [-1, 1] over the columns covered by ``stats``.

    Only the leading ``len(stats.q01)`` channels are normalized (the relative
    joint slice); trailing channels (padding / non-relative dims) pass through.
    Matches ``state_action.py`` ``mode == "q99"``: ``2*(x-q01)/(q99-q01)-1`` with
    the ``q01 == q99`` columns left as the identity, then clamped to [-1, 1].
    """
    action = np.asarray(action, dtype=np.float64)
    n = stats.q01.shape[0]
    out = action.copy()
    x = action[..., :n]
    q01 = stats.q01
    q99 = stats.q99
    span = q99 - q01
    mask = span != 0
    norm = np.where(mask, (x - q01) / np.where(mask, span, 1.0), x)
    norm = np.where(mask, 2.0 * norm - 1.0, x)
    norm = np.clip(norm, -1.0, 1.0)
    out[..., :n] = norm
    return out


def unnormalize(action: np.ndarray, stats: RelativeStats) -> np.ndarray:
    """Inverse of :func:`normalize`: ``(x+1)/2*(q99-q01)+q01`` on the joint slice.

    NOTE: q99 normalization clamps to [-1, 1], so values whose true magnitude
    exceeded the q01/q99 band are NOT recoverable; round-trip identity holds only
    for inputs that lay inside the band before normalization.
    """
    action = np.asarray(action, dtype=np.float64)
    n = stats.q01.shape[0]
    out = action.copy()
    x = action[..., :n]
    q01 = stats.q01
    q99 = stats.q99
    out[..., :n] = (x + 1.0) / 2.0 * (q99 - q01) + q01
    return out
