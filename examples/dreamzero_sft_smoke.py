#!/usr/bin/env python3
"""DreamZero LoRA-SFT smoke example for the Cybernetics SDK.

Default mode is local-only: it builds the same shape-correct synthetic DROID
batch that a real loader would encode, converts it into a Cybernetics Datum, and
prints the wire keys. Use ``--remote-run`` only when you are ready to create a
Worldlines session/model and spend GPU time on the configured control plane.
Remote runs cancel their SDK session on exit by default so successful smokes do
not leave paid compute running; pass ``--keep-lease`` for debugging.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

import cybernetics
from cybernetics import types
from cybernetics.lib.dreamzero import DREAMZERO_DROID_BASE_MODEL, relative_actions, serde

ACTION_HORIZON = 24
MAX_ACTION_DIM = 32
MAX_STATE_DIM = 64
NUM_STATE_PER_BLOCK = 1
NUM_FRAME_PER_BLOCK = 2
DEFAULT_NUM_FRAMES = 9
IMAGE_H = 176
IMAGE_W = 320
TEXT_SEQ_LEN = 512
JOINT_SLICE = (0, 8)

DZ_BETA1, DZ_BETA2 = 0.95, 0.999
DZ_EPS = 1e-8
DZ_WEIGHT_DECAY = 1e-5
DZ_GRAD_CLIP = 1.0


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    collate = build_synthetic_droid_collate(args.num_frames, rng)
    datum = serde.collate_to_datum(collate)
    loss_keys = sorted(datum.loss_fn_inputs)
    print(
        f"built_datum=true base_model={DREAMZERO_DROID_BASE_MODEL} "
        f"loss_keys={len(loss_keys)} num_frames={args.num_frames}"
    )
    print("sample_loss_keys=" + ",".join(loss_keys[:8]))

    if not args.remote_run:
        print("remote_run=false")
        return

    client: cybernetics.ServiceClient | None = None
    try:
        client = cybernetics.ServiceClient(
            base_url=args.base_url,
            project_id=args.project_id,
            user_metadata={"example": "dreamzero_sft_smoke"},
        )
        capabilities = client.get_server_capabilities()
        loss_families = set(capabilities.loss_families or [])
        if "cross_entropy" not in loss_families:
            advertised = ",".join(sorted(loss_families)) or "<none>"
            raise SystemExit(f"backend does not advertise cross_entropy; loss_families={advertised}")

        training = client.create_lora_training_client(
            base_model=DREAMZERO_DROID_BASE_MODEL,
            rank=args.rank,
            user_metadata={"lora_alpha": str(args.rank)},
            timeout=args.timeout,
        )
        fb = training.forward_backward([datum], "cross_entropy").result(timeout=args.timeout)
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
        checkpoint_name = args.checkpoint_name or f"dreamzero-sft-smoke-{int(time.time())}"
        checkpoint = training.save_state(checkpoint_name).result(timeout=args.timeout)
        print(f"checkpoint={checkpoint}")
    finally:
        cleanup_remote_session(
            client,
            keep_lease=args.keep_lease,
            timeout=args.cleanup_timeout,
        )


def build_synthetic_droid_collate(num_frames: int, rng: np.random.Generator) -> dict[str, Any]:
    """Build a synthetic B=1 DROID collate dict with DreamZero-compatible shapes."""

    if (num_frames - 1) % 8 != 0:
        raise ValueError("num_frames must be in the 8m+1 family, e.g. 9, 17, 25, 33")

    n_latent = 1 + (num_frames - 1) // 4
    n_blocks = (n_latent - 1) // NUM_FRAME_PER_BLOCK
    h_total = n_blocks * ACTION_HORIZON
    n_state = n_blocks * NUM_STATE_PER_BLOCK

    state = rng.standard_normal((n_state, MAX_STATE_DIM)).astype(np.float32)
    absolute_action = rng.standard_normal((h_total, MAX_ACTION_DIM)).astype(np.float32)
    relative_action = relative_actions.to_relative(
        absolute_action,
        state,
        anchor_index=0,
        joint_slice=JOINT_SLICE,
        action_horizon=ACTION_HORIZON,
        state_per_chunk=NUM_STATE_PER_BLOCK,
    )
    stats = relative_actions.RelativeStats(
        q01=np.full(JOINT_SLICE[1] - JOINT_SLICE[0], -1.0),
        q99=np.full(JOINT_SLICE[1] - JOINT_SLICE[0], 1.0),
        mean=np.zeros(JOINT_SLICE[1] - JOINT_SLICE[0]),
        std=np.ones(JOINT_SLICE[1] - JOINT_SLICE[0]),
        min=np.full(JOINT_SLICE[1] - JOINT_SLICE[0], -1.0),
        max=np.full(JOINT_SLICE[1] - JOINT_SLICE[0], 1.0),
    )
    action = np.clip(relative_actions.normalize(relative_action, stats), -1.0, 1.0)

    return {
        "text": rng.integers(0, 32000, size=(1, TEXT_SEQ_LEN), dtype=np.int64),
        "text_attention_mask": np.ones((1, TEXT_SEQ_LEN), dtype=bool),
        "state": state[None],
        "state_mask": np.ones((1, n_state, MAX_STATE_DIM), dtype=bool),
        "images": rng.integers(
            0,
            256,
            size=(1, num_frames, 2 * IMAGE_H, 2 * IMAGE_W, 3),
            dtype=np.uint8,
        ),
        "embodiment_id": np.array([0], dtype=np.int64),
        "has_real_action": np.array([True], dtype=bool),
        "has_lapa_action": np.array([False], dtype=bool),
        "is_cotrain_instance": np.array([False], dtype=bool),
        "action": action.astype(np.float32)[None],
        "action_mask": np.ones((1, h_total, MAX_ACTION_DIM), dtype=bool),
    }


def cleanup_remote_session(
    client: cybernetics.ServiceClient | None,
    *,
    keep_lease: bool,
    timeout: float,
) -> None:
    """Best-effort cleanup for paid remote example runs."""

    if client is None:
        return
    if keep_lease:
        print(f"cleanup_skipped=true reason=keep_lease session_id={client.session_id}")
        return
    try:
        response = client.create_rest_client().cancel_session(client.session_id).result(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - cleanup must not hide the real example failure.
        print(f"cleanup_error=true session_id={client.session_id} error={exc}")
        return
    stopped = ",".join(response.stopped_lease_ids) if response.stopped_lease_ids else "<none>"
    print(f"cleanup_session_cancelled=true session_id={client.session_id} stopped_lease_ids={stopped}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run", action="store_true", help="Create remote GPU work.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Cybernetics API base URL. Defaults to CYBERNETICS_BASE_URL, CP_API_BASE, or stored login.",
    )
    parser.add_argument("--project-id", default=os.environ.get("CYBERNETICS_PROJECT_ID"))
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
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
