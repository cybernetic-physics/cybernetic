#!/usr/bin/env python3
"""DreamZero flow-RWR smoke example for the Cybernetics SDK.

Default mode is local-only: it builds a tiny synthetic DreamZero conditioning
batch plus per-step RL trajectory tensors and prints the wire keys. Use
``--remote-run`` only when you are ready to create a Worldlines session/model
and spend GPU time on the configured control plane. Remote runs cancel their SDK
session on exit by default so successful smokes do not leave paid compute
running; pass ``--keep-lease`` for debugging.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
from dreamzero_sft_smoke import (
    ACTION_HORIZON,
    DZ_BETA1,
    DZ_BETA2,
    DZ_EPS,
    DZ_GRAD_CLIP,
    DZ_WEIGHT_DECAY,
    MAX_ACTION_DIM,
    build_synthetic_droid_collate,
    cleanup_remote_session,
)

import cybernetics
from cybernetics import types
from cybernetics.lib.dreamzero import DREAMZERO_DROID_BASE_MODEL, serde

DEFAULT_NUM_FRAMES = 9
DEFAULT_TRAJECTORY_STEPS = 2


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    collate = build_synthetic_droid_collate(args.num_frames, rng)
    trajectory = build_synthetic_flow_trajectory(
        steps=args.trajectory_steps,
        rng=rng,
    )
    datum = serde.collate_to_datum(
        collate,
        world_model_only=True,
        extra_loss_inputs=trajectory,
    )
    loss_keys = sorted(datum.loss_fn_inputs)
    print(
        f"built_rl_datum=true base_model={DREAMZERO_DROID_BASE_MODEL} "
        f"loss_keys={len(loss_keys)} trajectory_steps={args.trajectory_steps}"
    )
    print("sample_loss_keys=" + ",".join(loss_keys[:12]))
    print("sample_rl_keys=" + ",".join(sorted(trajectory.keys())[:12]))

    if not args.remote_run:
        print("remote_run=false")
        return

    client: cybernetics.ServiceClient | None = None
    try:
        client = cybernetics.ServiceClient(
            base_url=args.base_url,
            project_id=args.project_id,
            user_metadata={"example": "dreamzero_rl_smoke"},
        )
        capabilities = client.get_server_capabilities()
        loss_families = set(capabilities.loss_families or [])
        if args.loss_fn not in loss_families:
            advertised = ",".join(sorted(loss_families)) or "<none>"
            raise SystemExit(
                f"backend does not advertise {args.loss_fn}; loss_families={advertised}"
            )

        training = client.create_lora_training_client(
            base_model=DREAMZERO_DROID_BASE_MODEL,
            rank=args.rank,
            user_metadata={"lora_alpha": str(args.rank), "example": "dreamzero_rl_smoke"},
            timeout=args.timeout,
        )
        fb = training.forward_backward([datum], args.loss_fn).result(timeout=args.timeout)
        print(f"forward_backward_done=true model_id={training.model_id} result={fb}")

        adam = types.AdamParams(
            learning_rate=args.learning_rate,
            beta1=DZ_BETA1,
            beta2=DZ_BETA2,
            eps=DZ_EPS,
            weight_decay=DZ_WEIGHT_DECAY,
            grad_clip_norm=DZ_GRAD_CLIP,
        )
        training.optim_step(adam).result(timeout=args.timeout)
        checkpoint_name = args.checkpoint_name or f"dreamzero-rl-smoke-{int(time.time())}"
        checkpoint = training.save_state(checkpoint_name).result(timeout=args.timeout)
        print(f"checkpoint={checkpoint}")
    finally:
        cleanup_remote_session(
            client,
            keep_lease=args.keep_lease,
            timeout=args.cleanup_timeout,
        )


def build_synthetic_flow_trajectory(
    *,
    steps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build a tiny per-SDE-step trajectory for the DreamZero RL loss path."""

    if steps < 1:
        raise ValueError("steps must be >= 1")

    x_t = rng.standard_normal((steps, 1, ACTION_HORIZON, MAX_ACTION_DIM)).astype(np.float32)
    noise = 0.05 * rng.standard_normal((steps, 1, ACTION_HORIZON, MAX_ACTION_DIM)).astype(np.float32)
    x_prev = x_t - noise
    return {
        "x_t": x_t,
        "x_prev": x_prev,
        "mu_old": x_prev.copy(),
        "sigma": np.full((steps, 1), 0.1, dtype=np.float32),
        "log_prob_old": np.zeros((steps, 1), dtype=np.float32),
        "t": np.linspace(1.0, 0.1, steps, dtype=np.float32),
        "dt": np.full((steps,), 1.0 / steps, dtype=np.float32),
        "advantages": np.array([1.0], dtype=np.float32),
        "rwr_weights": np.array([1.0], dtype=np.float32),
        "group_ids": np.zeros((1,), dtype=np.int64),
        "trajectory_ids": np.zeros((1,), dtype=np.int64),
        "token_mask": np.ones((steps,), dtype=np.int64),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run", action="store_true", help="Create remote GPU work.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Cybernetics API base URL. Defaults to CYBERNETICS_BASE_URL, CP_API_BASE, or stored login.",
    )
    parser.add_argument("--project-id", default=os.environ.get("CYBERNETICS_PROJECT_ID"))
    parser.add_argument("--loss-fn", default="flow_rwr", choices=("flow_rwr", "ppo", "importance_sampling"))
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--trajectory-steps", type=int, default=DEFAULT_TRAJECTORY_STEPS)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    parser.add_argument("--keep-lease", action="store_true", help="Leave remote compute running after the example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
