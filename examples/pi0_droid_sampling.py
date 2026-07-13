#!/usr/bin/env python3
"""Sample the hosted PI0 DROID policy through the authenticated Cybernetics SDK.

The input is an ``.npz`` file containing the three RGB observations and robot
state listed by ``DroidObservation``. Use ``--validate-only`` to check the file
without creating a hosted Worldlines session.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import cybernetics
from cybernetics import types

OBSERVATION_KEYS = (
    "exterior_image_0_left",
    "exterior_image_1_left",
    "wrist_image_left",
    "joint_position",
    "gripper_position",
)


def main() -> None:
    args = _parse_args()
    observation = load_observation(args.observation, instruction=args.instruction)
    print(f"observation_valid=true path={args.observation}")

    if args.validate_only:
        print("remote_run=false")
        return

    client = cybernetics.ServiceClient(
        base_url=args.base_url,
        project_id=args.project_id,
        user_metadata={"example": "pi0_droid_sampling"},
    )
    try:
        sampler = client.create_sampling_client(
            base_model="pi0-droid",
            timeout=args.timeout,
        )
        response = sampler.sample_droid(observation).result(timeout=args.timeout)
        actions = _require_action_chunk(response)
        np.save(args.output, actions, allow_pickle=False)
        print(
            f"sample_complete=true action_space=droid_joint_position "
            f"shape={list(actions.shape)} output={args.output}"
        )
    finally:
        if args.keep_session:
            print(f"cleanup_skipped=true session_id={client.session_id}")
        else:
            _cancel_session(client, timeout=args.cleanup_timeout)


def load_observation(path: Path, *, instruction: str) -> types.DroidObservation:
    """Load and validate one raw DROID observation from an NPZ file."""

    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in OBSERVATION_KEYS if key not in payload]
        if missing:
            raise ValueError(f"observation is missing NPZ keys: {', '.join(missing)}")

        images = {
            key: _require_rgb(key, payload[key]) for key in OBSERVATION_KEYS if "image" in key
        }
        joints = np.asarray(payload["joint_position"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(payload["gripper_position"], dtype=np.float32).reshape(-1)

    if joints.shape != (7,):
        raise ValueError(f"joint_position must contain 7 values, got shape {joints.shape}")
    if gripper.shape != (1,):
        raise ValueError(f"gripper_position must contain 1 value, got shape {gripper.shape}")
    if not np.isfinite(joints).all() or not np.isfinite(gripper).all():
        raise ValueError("joint_position and gripper_position must be finite")

    return types.DroidObservation.from_numpy(
        exterior_image_0_left=images["exterior_image_0_left"],
        exterior_image_1_left=images["exterior_image_1_left"],
        wrist_image_left=images["wrist_image_left"],
        joint_position=joints,
        gripper_position=gripper,
        instruction=instruction,
    )


def _require_rgb(name: str, image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"{name} must be an H x W x 3 uint8 RGB image, got {image.shape} {image.dtype}"
        )
    return image


def _require_action_chunk(response: types.SampleResponse) -> np.ndarray:
    if response.action_chunk is None:
        raise RuntimeError("PI0 response did not contain an action_chunk")
    actions = response.action_chunk.to_numpy().astype(np.float32, copy=False)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 8:
        raise RuntimeError(f"PI0 action_chunk must have shape [H, 8], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("PI0 action_chunk contains non-finite values")
    return actions


def _cancel_session(client: cybernetics.ServiceClient, *, timeout: float) -> None:
    try:
        client.create_rest_client().cancel_session(client.session_id).result(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - cleanup must not hide the sampling result.
        print(f"cleanup_error=true session_id={client.session_id} error={exc}", file=sys.stderr)
        return
    print(f"cleanup_session_cancelled=true session_id={client.session_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path, help="NPZ file containing one DROID observation.")
    parser.add_argument("--instruction", required=True, help="Natural-language robot instruction.")
    parser.add_argument("--output", type=Path, default=Path("pi0-droid-actions.npy"))
    parser.add_argument(
        "--base-url",
        default=None,
        help="Defaults to CYBERNETICS_BASE_URL, CP_API_BASE, or the stored login.",
    )
    parser.add_argument("--project-id", default=os.environ.get("CYBERNETICS_PROJECT_ID"))
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate input without API work."
    )
    parser.add_argument(
        "--keep-session", action="store_true", help="Keep the Worldlines session open."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
