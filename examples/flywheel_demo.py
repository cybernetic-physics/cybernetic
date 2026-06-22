#!/usr/bin/env python3
"""Synthetic-data flywheel demo for the Cybernetics SDK.

THE FLYWHEEL: a world model dreams robot-manipulation video, an inverse-dynamics
model (IDM) labels the dreamed frames with pseudo-actions, and a VLA trains on
``(dreamed video, pseudo-actions)`` -- closing the loop where generated data
improves the policy with no new teleop. This demo wires one slice of that loop:

    Cosmos3-Nano dream-video
        -> placeholder-IDM pseudo-actions
        -> a dreamzero/groot collate (images = the dreamed frames)
        -> forward_backward on groot-n1.5

The dreamed frames must match the dreamzero/groot collate ``images`` geometry
([1, num_frames, 2*IMAGE_H, 2*IMAGE_W, 3] uint8, i.e. the 2x2 view grid the head
reads), so the Cosmos clip is generated at 352x640 (IMAGE_H=176, IMAGE_W=320).

THE IDM HERE IS A PLACEHOLDER. It is a crude frame-diff motion proxy, NOT a real
inverse-dynamics model. The real IDM is NVIDIA GR00T-Dreams ``idm_training.py``,
which predicts the action sequence that produced a video. We keep the synthetic
DROID action from the collate fixture and only PRINT the motion proxy, so the
forward_backward path stays shape-correct while making the IDM seam explicit.

MODES:
  * local (default): synthesize stand-in frames (no GPU, no cosmos venv) and only
    build the collate + datum + print wire keys. Nothing is shipped.
  * --remote-run: ship the flywheel datum to groot-n1.5 forward_backward on the
    configured control plane. Cosmos generation still needs the cosmos venv: pass
    ``--cosmos-checkpoint`` to dream a real clip locally (requires diffusers /
    diffusers_cosmos3 + a GPU + the Cosmos3-Nano checkpoint); without it the
    stand-in frames are used so the loop is exercisable end-to-end.

The remote run cancels its SDK session on exit by default so successful runs do
not leave paid compute running; pass ``--keep-lease`` for debugging.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from dreamzero_sft_smoke import (
    DZ_BETA1,
    DZ_BETA2,
    DZ_EPS,
    DZ_GRAD_CLIP,
    DZ_WEIGHT_DECAY,
    IMAGE_H,
    IMAGE_W,
    build_synthetic_droid_collate,
    cleanup_remote_session,
)
from groot_sft_smoke import GROOT_BASE_MODEL, _resolve_api_key

import cybernetics
from cybernetics import types
from cybernetics.lib.dreamzero import serde

DEFAULT_NUM_FRAMES = 9
# Cosmos dreams at the dreamzero collate image geometry: the 2x2 view grid.
DREAM_H = 2 * IMAGE_H
DREAM_W = 2 * IMAGE_W
DREAM_PROMPT = (
    "a franka robot arm picks up a red cube from a table and places it on a "
    "plate, third-person view, smooth motion"
)


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    dream = _dream_video(args, rng)
    print(f"dream_frames={tuple(dream.shape)} source={_dream_source(args)}")

    collate = build_synthetic_droid_collate(args.num_frames, rng)
    # --- FLYWHEEL BRIDGE: the dreamed video becomes the VLA's training images ---
    if collate["images"].shape != dream[None].shape:
        raise ValueError(
            f"dream geometry {dream[None].shape} != collate images {collate['images'].shape}; "
            "regenerate the dream at the dreamzero 2x2 grid resolution"
        )
    collate["images"] = dream[None].astype(np.uint8)

    # --- placeholder IDM: a crude motion proxy from the dream frame-diffs. ---
    # REAL = NVIDIA GR00T-Dreams idm_training.py (predicts the action sequence
    # that produced the video). Here we keep the collate's synthetic DROID action
    # and only surface the motion energy, so the IDM seam is explicit.
    motion = np.abs(np.diff(dream.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    print("placeholder_idm_motion_proxy=" + ",".join(f"{m:.1f}" for m in motion))

    datum = serde.collate_to_datum(collate)
    loss_keys = sorted(datum.loss_fn_inputs)
    print(
        f"built_flywheel_datum=true base_model={GROOT_BASE_MODEL} "
        f"loss_keys={len(loss_keys)} num_frames={args.num_frames}"
    )
    print("sample_loss_keys=" + ",".join(loss_keys[:8]))

    if not args.remote_run:
        print("remote_run=false")
        return

    api_key = _resolve_api_key(args)
    client: cybernetics.ServiceClient | None = None
    try:
        client = cybernetics.ServiceClient(
            base_url=args.base_url,
            api_key=api_key,
            project_id=args.project_id,
            default_headers={"X-API-Key": api_key} if api_key else {},
            user_metadata={"example": "flywheel_demo"},
        )
        training = client.create_lora_training_client(
            base_model=GROOT_BASE_MODEL,
            rank=args.rank,
            user_metadata={"lora_alpha": str(args.rank), "example": "flywheel_demo"},
            timeout=args.timeout,
        )
        fb = training.forward_backward([datum], "cross_entropy").result(timeout=args.timeout)
        print(f"flywheel_forward_backward_done=true model_id={training.model_id} result={fb}")

        adam = types.AdamParams(
            learning_rate=args.learning_rate,
            beta1=DZ_BETA1,
            beta2=DZ_BETA2,
            eps=DZ_EPS,
            weight_decay=DZ_WEIGHT_DECAY,
            grad_clip_norm=DZ_GRAD_CLIP,
        )
        training.optim_step(adam).result(timeout=args.timeout)
        print("flywheel_loop_ok=true note=cosmos_dream->idm_stub->datum->groot_forward_backward")
    finally:
        cleanup_remote_session(
            client,
            keep_lease=args.keep_lease,
            timeout=args.cleanup_timeout,
        )


def _dream_source(args: argparse.Namespace) -> str:
    if args.dream_npy:
        return "npy"
    if args.cosmos_checkpoint:
        return "cosmos"
    return "synthetic"


def _dream_video(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Return the dreamed clip as uint8 [num_frames, DREAM_H, DREAM_W, 3].

    Precedence: a pre-dreamed ``--dream-npy`` file -> a live Cosmos generation via
    ``--cosmos-checkpoint`` (needs the cosmos venv + GPU) -> synthetic stand-in
    frames (default, no GPU). The synthetic path keeps the demo runnable anywhere.
    """
    if args.dream_npy:
        dream = np.load(args.dream_npy)
        return dream.astype(np.uint8)
    if args.cosmos_checkpoint:
        return _cosmos_dream(args)
    # Synthetic stand-in: a smooth moving gradient so the motion proxy is nonzero.
    frames = np.empty((args.num_frames, DREAM_H, DREAM_W, 3), dtype=np.uint8)
    base = rng.integers(0, 256, size=(DREAM_H, DREAM_W, 3), dtype=np.uint8).astype(np.int16)
    for t in range(args.num_frames):
        shifted = np.roll(base, shift=4 * t, axis=1)
        frames[t] = np.clip(shifted, 0, 255).astype(np.uint8)
    return frames


def _cosmos_dream(args: argparse.Namespace) -> np.ndarray:
    """Dream a clip with the Cosmos3-Nano pipeline (requires the cosmos venv + GPU).

    Mirrors flywheel_dream.py: generate at the dreamzero 2x2 grid resolution so the
    frames drop straight into the groot collate ``images`` slot.
    """
    try:
        import torch  # noqa: F401
        import diffusers_cosmos3  # noqa: F401
        from diffusers import Cosmos3OmniDiffusersPipeline
    except ImportError as exc:  # pragma: no cover - needs the cosmos venv.
        raise SystemExit(
            "cosmos dream requires the cosmos venv (diffusers + diffusers_cosmos3 + torch + a GPU); "
            f"install it or drop --cosmos-checkpoint to use synthetic frames. ({exc})"
        ) from exc

    import torch

    pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
        args.cosmos_checkpoint, torch_dtype=torch.bfloat16
    ).to("cuda")
    out = pipe(
        prompt=DREAM_PROMPT,
        num_frames=args.num_frames,
        height=DREAM_H,
        width=DREAM_W,
        fps=12,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    frames = out[0] if isinstance(out, (list, tuple)) else out  # [3, T, H, W] in [0, 1]
    video = frames.float().clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]
    return (video * 255).round().astype(np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run", action="store_true", help="Create remote GPU work.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Cybernetics API base URL. Defaults to CYBERNETICS_BASE_URL, CP_API_BASE, or stored login.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to CYBERNETICS_API_KEY then WORLDLINES_API_KEY.",
    )
    parser.add_argument("--project-id", default=os.environ.get("CYBERNETICS_PROJECT_ID"))
    parser.add_argument(
        "--cosmos-checkpoint",
        default=None,
        help="Path to the Cosmos3-Nano checkpoint to dream a real clip (needs the cosmos venv + GPU).",
    )
    parser.add_argument(
        "--dream-npy",
        default=None,
        help="Path to a pre-dreamed uint8 [num_frames, H, W, 3] .npy clip (skips Cosmos generation).",
    )
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    parser.add_argument(
        "--keep-lease", action="store_true", help="Leave remote compute running after the example."
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
