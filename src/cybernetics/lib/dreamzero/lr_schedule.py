"""Driver-side math the runtime CANNOT carry over the wire.

``AdamParams`` has no LR-schedule field, so the DRIVER computes the per-step
learning rate (cosine + warmup) client-side and ships it via
``adam_params.learning_rate`` on every ``optim_step``. Likewise the RL advantage
and RWR-weight reductions run client-side and ride ``loss_fn_inputs`` (NOT
``loss_fn_config``, which is ``Dict[str, float]``-only).

Everything here is pure ``numpy``/``math`` -- NO ``groot``, NO ``torch``. The
advantage/RWR helpers are numpy ports of
``groot/vla/rl/rollouts.py`` (``compute_standardized_advantages`` /
``compute_group_advantages`` / ``compute_rwr_weights``) so the driver reproduces
the dreamzero reductions without importing groot.

LR contract (the dreamzero DROID recipe):
    base_lr = 1e-4, max_steps = 100, warmup_ratio = 0.05  -> warmup = ceil(5) = 5

    step < warmup : lr = base_lr * step / warmup            (linear ramp 0 -> base)
    step >= warmup: lr = base_lr * 0.5 * (1 + cos(pi * prog))  (cosine -> 0)
                    prog = (step - warmup) / (max_steps - warmup)

This matches HF ``get_cosine_schedule_with_warmup`` with the default half-cosine
(num_cycles = 0.5). ``step`` is 1-indexed at the optim_step boundary: step 0
returns 0.0, step == warmup returns ``base_lr``, step == max_steps returns ~0.
"""

from __future__ import annotations

import math

import numpy as np


def warmup_steps(max_steps: int, warmup_ratio: float) -> int:
    """``ceil(warmup_ratio * max_steps)`` -- the integer warmup length.

    Matches the HF ``int(warmup_ratio * num_training_steps)`` intent but uses
    ``ceil`` so a tiny ratio still yields at least one ramp step when the product
    is non-zero (warmup_ratio=0.05, max_steps=100 -> 5).
    """
    return int(math.ceil(warmup_ratio * max_steps))


def cosine_with_warmup_lr(
    step: int,
    *,
    base_lr: float = 1e-4,
    max_steps: int = 100,
    warmup_ratio: float = 0.05,
) -> float:
    """Per-step learning rate: linear warmup then half-cosine decay to ~0.

    ``step`` is the (1-indexed) optimizer step about to be taken. Clamps to
    ``[0, base_lr]`` over the warmup phase and to ``[0, base_lr]`` over the decay
    phase (never negative past ``max_steps``).
    """
    if base_lr <= 0.0:
        return 0.0
    warmup = warmup_steps(max_steps, warmup_ratio)
    if step <= 0:
        return 0.0
    if warmup > 0 and step < warmup:
        return base_lr * float(step) / float(warmup)
    denom = max(1, max_steps - warmup)
    progress = float(step - warmup) / float(denom)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def standardized_advantages(
    rewards: np.ndarray,
    *,
    clip: float | None = None,
) -> np.ndarray:
    """z-score of per-trajectory scalar rewards (MC return baseline).

    numpy port of ``RolloutBuffer.compute_standardized_advantages``: population
    std (``ddof=0``) clamped to 1e-6, optional symmetric clip. Returns float32.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    mean = r.mean()
    std = max(float(r.std(ddof=0)), 1e-6)
    adv = (r - mean) / std
    if clip is not None:
        adv = np.clip(adv, -clip, clip)
    return adv.astype(np.float32, copy=False)


def group_advantages(
    rewards: np.ndarray,
    group_size: int,
    *,
    clip: float | None = None,
) -> np.ndarray:
    """GRPO-style within-group z-score (G rollouts per conditioning state).

    numpy port of ``RolloutBuffer.compute_group_advantages``: reshape to
    ``[-1, group_size]``, z-score within each row (population std clamped to
    1e-6), flatten back. Requires ``len(rewards) % group_size == 0``.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    n = r.shape[0]
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if n % group_size != 0:
        raise ValueError(
            f"N={n} not divisible by group_size={group_size}; submit rollouts in groups."
        )
    grouped = r.reshape(-1, group_size)
    mean = grouped.mean(axis=1, keepdims=True)
    std = np.maximum(grouped.std(axis=1, ddof=0, keepdims=True), 1e-6)
    adv = ((grouped - mean) / std).reshape(-1)
    if clip is not None:
        adv = np.clip(adv, -clip, clip)
    return adv.astype(np.float32, copy=False)


def rwr_weights(
    rewards: np.ndarray,
    *,
    beta: float = 1.0,
    max_weight: float = 20.0,
    baseline: str = "mean",
) -> np.ndarray:
    """Exponential-of-advantage RWR/AWR weights, clipped at ``max_weight``.

    numpy port of ``RolloutBuffer.compute_rwr_weights``:
    ``w_i = min(exp((R_i - base) / beta), max_weight)`` with
    ``base in {mean, median, 0.0}``. Returns float32.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if baseline == "mean":
        base = float(r.mean())
    elif baseline == "median":
        base = float(np.median(r))
    elif baseline == "none":
        base = 0.0
    else:
        raise ValueError(f"unknown baseline {baseline!r}")
    adv = r - base
    w = np.exp(adv / max(beta, 1e-6))
    w = np.minimum(w, max_weight)
    return w.astype(np.float32, copy=False)


__all__ = [
    "warmup_steps",
    "cosine_with_warmup_lr",
    "standardized_advantages",
    "group_advantages",
    "rwr_weights",
]
