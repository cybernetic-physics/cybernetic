#!/usr/bin/env python3
"""TEMPLATE: finetune a robot foundation model on a REAL LeRobot dataset.

The ``*_smoke`` examples ship a SYNTHETIC collate; this template shows the seam
where real data plugs in. It loads a ``LeRobotDataset`` (e.g. an SO-101 set),
pulls one episode's (image frames, state, action), maps them onto the
dreamzero/groot collate contract (resize images to the 2H x 2W view grid,
normalize the action), encodes a ``Datum`` with the dreamzero serde, and runs
``forward_backward`` / ``optim_step`` / ``save_state`` on ``groot-n1.5``.

It is runnable-shaped, but the exact dataset + the state/action <-> collate
mapping MUST be confirmed against your dataset's feature schema (joint count, image
keys, fps). See the inline ``# TODO`` notes. The ``--dataset`` default is a
PLACEHOLDER -- set it to your real LeRobot dataset id.

Local-only by default: it builds the collate + datum and prints the wire keys.
Use ``--remote-run`` to ship one real-data step to the control plane (cancels the
SDK session on exit unless ``--keep-lease``).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from dreamzero_sft_smoke import (
    ACTION_HORIZON,
    DZ_BETA1,
    DZ_BETA2,
    DZ_EPS,
    DZ_GRAD_CLIP,
    DZ_WEIGHT_DECAY,
    IMAGE_H,
    IMAGE_W,
    JOINT_SLICE,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    NUM_STATE_PER_BLOCK,
    TEXT_SEQ_LEN,
    cleanup_remote_session,
)
from groot_sft_smoke import GROOT_BASE_MODEL, _resolve_api_key

import cybernetics
from cybernetics import types
from cybernetics.lib.dreamzero import relative_actions, serde

# TODO: confirm dataset id. This is a PLACEHOLDER SO-101 LeRobot dataset id; set
# --dataset to the real one you have access to (check the LeRobot hub / your own).
DEFAULT_DATASET = "lerobot/svla_so101_pickplace"
DEFAULT_NUM_FRAMES = 9
DREAM_H = 2 * IMAGE_H
DREAM_W = 2 * IMAGE_W


def main() -> None:
    args = _parse_args()
    dataset = _load_lerobot_dataset(args.dataset)
    episode = _read_one_episode(dataset, args.episode, args.num_frames)
    collate = _episode_to_collate(episode, args.num_frames)
    datum = serde.collate_to_datum(collate)
    loss_keys = sorted(datum.loss_fn_inputs)
    print(
        f"built_datum=true base_model={GROOT_BASE_MODEL} dataset={args.dataset} "
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
            user_metadata={"example": "finetune_on_real_data"},
        )
        training = client.create_lora_training_client(
            base_model=GROOT_BASE_MODEL,
            rank=args.rank,
            user_metadata={"lora_alpha": str(args.rank), "example": "finetune_on_real_data"},
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
        checkpoint_name = args.checkpoint_name or f"groot-real-data-{int(time.time())}"
        checkpoint = training.save_state(checkpoint_name).result(timeout=args.timeout)
        print(f"checkpoint={checkpoint}")
    finally:
        cleanup_remote_session(
            client,
            keep_lease=args.keep_lease,
            timeout=args.cleanup_timeout,
        )


def _load_lerobot_dataset(dataset_id: str):
    """Load a LeRobotDataset, with a helpful message if lerobot is not installed."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            # Newer lerobot moved the module path.
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "lerobot is required for this template: pip install lerobot "
                "(see https://github.com/huggingface/lerobot). "
                f"({exc})"
            ) from exc
    return LeRobotDataset(dataset_id)


def _read_one_episode(dataset, episode_index: int, num_frames: int) -> dict:
    """Pull (image frames, state, action) for ``num_frames`` steps of one episode.

    TODO: confirm the feature keys against your dataset. LeRobot datasets vary in
    their image key ("observation.image" vs "observation.images.<cam>"), state key
    ("observation.state"), and action key ("action"). This reads the first
    matching keys and the first ``num_frames`` frames of the requested episode.
    """
    from_idx, to_idx = _episode_bounds(dataset, episode_index)
    image_key = _first_image_key(dataset)
    state_key = "observation.state"
    action_key = "action"

    images, states, actions = [], [], []
    for frame_idx in range(from_idx, min(from_idx + num_frames, to_idx)):
        frame = dataset[frame_idx]
        images.append(_to_uint8_hwc(frame[image_key]))
        states.append(np.asarray(frame[state_key], dtype=np.float32).reshape(-1))
        actions.append(np.asarray(frame[action_key], dtype=np.float32).reshape(-1))
    return {
        "images": np.stack(images),  # [T, h, w, 3] uint8
        "state": np.stack(states),  # [T, state_dim_raw]
        "action": np.stack(actions),  # [T, action_dim_raw]
    }


def _episode_bounds(dataset, episode_index: int) -> tuple[int, int]:
    # LeRobotDataset exposes per-episode frame ranges; fall back to whole dataset.
    try:
        from_idx = int(dataset.episode_data_index["from"][episode_index].item())
        to_idx = int(dataset.episode_data_index["to"][episode_index].item())
        return from_idx, to_idx
    except (AttributeError, KeyError, IndexError, TypeError):
        return 0, len(dataset)


def _first_image_key(dataset) -> str:
    # TODO: confirm the camera key you want; this picks the first image feature.
    features = getattr(dataset, "features", {}) or {}
    for key in features:
        if "image" in key:
            return key
    return "observation.image"


def _to_uint8_hwc(value) -> np.ndarray:
    """Coerce a LeRobot image (torch CHW float [0,1] or numpy HWC uint8) to HWC uint8."""
    arr = np.asarray(getattr(value, "numpy", lambda: value)())
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _resize_to_grid(images: np.ndarray) -> np.ndarray:
    """Nearest-neighbor resize each [h, w, 3] frame to the [DREAM_H, DREAM_W, 3] grid.

    A real loader composites num_views cameras into a 2x2 grid then resizes; here we
    just resize a single camera into the same target geometry so the wire shape is
    correct. TODO: swap for your real multi-view compositing + a quality resize.
    """
    t, _, _, c = images.shape
    out = np.empty((t, DREAM_H, DREAM_W, c), dtype=np.uint8)
    for i in range(t):
        out[i] = _nn_resize(images[i], DREAM_H, DREAM_W)
    return out


def _nn_resize(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    ys = (np.arange(out_h) * h // out_h).clip(0, h - 1)
    xs = (np.arange(out_w) * w // out_w).clip(0, w - 1)
    return frame[ys[:, None], xs[None, :]]


def _episode_to_collate(episode: dict, num_frames: int) -> dict:
    """Map one episode onto the dreamzero/groot collate contract.

    Action token count is TIED to the frame count by the head block structure
    (see dreamzero_sft_smoke.build_synthetic_droid_collate): num_frames (8m+1) ->
    n_latent -> n_blocks -> h_total actions, n_state states. We pad/clip the real
    state/action dims onto MAX_STATE_DIM / MAX_ACTION_DIM, run the SAME client-side
    relative + normalize path, and resize the images into the 2x2 grid.
    """
    if (num_frames - 1) % 8 != 0:
        raise ValueError("num_frames must be in the 8m+1 family, e.g. 9, 17, 25, 33")

    n_latent = 1 + (num_frames - 1) // 4
    n_blocks = (n_latent - 1) // 2
    h_total = n_blocks * ACTION_HORIZON
    n_state = n_blocks * NUM_STATE_PER_BLOCK

    # TODO: confirm the state/action dim mapping for your robot (joint order,
    # gripper channel). Here we pad/truncate the raw dims onto the wire widths.
    state = _pad_to(episode["state"], n_state, MAX_STATE_DIM)
    raw_action = _pad_to(episode["action"], h_total, MAX_ACTION_DIM)

    rel_action = relative_actions.to_relative(
        raw_action,
        state,
        anchor_index=0,
        joint_slice=JOINT_SLICE,
        action_horizon=ACTION_HORIZON,
        state_per_chunk=NUM_STATE_PER_BLOCK,
    )
    joint_n = JOINT_SLICE[1] - JOINT_SLICE[0]
    # TODO: load your dataset's real q01/q99 normalization stats instead of this
    # neutral identity-ish band (matches the synthetic fixture).
    stats = relative_actions.RelativeStats(
        q01=np.full(joint_n, -1.0),
        q99=np.full(joint_n, 1.0),
        mean=np.zeros(joint_n),
        std=np.ones(joint_n),
        min=np.full(joint_n, -1.0),
        max=np.full(joint_n, 1.0),
    )
    action = np.clip(relative_actions.normalize(rel_action, stats), -1.0, 1.0).astype(np.float32)

    images = _resize_to_grid(episode["images"])  # [num_frames, DREAM_H, DREAM_W, 3]
    if images.shape[0] != num_frames:
        # TODO: handle short episodes (pad/repeat last frame) for your data.
        raise ValueError(
            f"episode yielded {images.shape[0]} frames but num_frames={num_frames}; "
            "pick a longer episode or lower --num-frames"
        )

    return {
        # TODO: tokenize the real task instruction with the umt5 tokenizer; a
        # placeholder token stream keeps the shape correct (umt5 max_length=512).
        "text": np.zeros((1, TEXT_SEQ_LEN), dtype=np.int64),
        "text_attention_mask": np.ones((1, TEXT_SEQ_LEN), dtype=bool),
        "state": state[None],
        "state_mask": np.ones((1, n_state, MAX_STATE_DIM), dtype=bool),
        "images": images[None],
        "embodiment_id": np.array([0], dtype=np.int64),
        "has_real_action": np.array([True], dtype=bool),
        "has_lapa_action": np.array([False], dtype=bool),
        "is_cotrain_instance": np.array([False], dtype=bool),
        "action": action[None],
        "action_mask": np.ones((1, h_total, MAX_ACTION_DIM), dtype=bool),
    }


def _pad_to(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Pad/truncate a [T, D] array to exactly [rows, cols] (zero-pad, repeat last row)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    t, d = arr.shape
    out = np.zeros((rows, cols), dtype=np.float32)
    use_d = min(d, cols)
    for i in range(rows):
        src = arr[min(i, t - 1)]  # repeat last row if too short
        out[i, :use_d] = src[:use_d]
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run", action="store_true", help="Create remote GPU work.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="LeRobot dataset id (PLACEHOLDER default; confirm before training for real).",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to read.")
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
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    parser.add_argument(
        "--keep-lease", action="store_true", help="Leave remote compute running after the example."
    )
    parser.add_argument("--checkpoint-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
