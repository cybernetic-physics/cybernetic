#!/usr/bin/env python3
"""GR00T N1.5 LoRA-SFT smoke example for the Cybernetics SDK.

GR00T N1.5 on Worldlines is the Wan-lineage groot: the ``groot-n1.5`` backend
consumes the SAME flow-matching VLA datum as ``dreamzero-droid`` (the
``dreamzero_cotrain.collate`` dict), so this example reuses
``cybernetics.lib.dreamzero.serde`` and the SAME shape-correct synthetic DROID
collate that ``dreamzero_sft_smoke`` builds. The only difference on the wire is
the ``base_model`` string; everything else is identical.

Default mode is local-only: it builds the synthetic DROID batch a real loader
would encode, converts it into a Cybernetics Datum, and prints the wire keys.
Use ``--remote-run`` only when you are ready to create a Worldlines
session/model and spend GPU time on the configured control plane (the runtime
builds the VLA on first contact, which can take minutes). Remote runs cancel
their SDK session on exit by default so successful smokes do not leave paid
compute running; pass ``--keep-lease`` for debugging.
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
    build_synthetic_droid_collate,
    cleanup_remote_session,
)

import cybernetics
from cybernetics import types
from cybernetics.lib.dreamzero import serde

# GR00T N1.5 is the Wan-lineage groot; same flow-matching VLA datum as
# dreamzero-droid, dispatched on a different base_model string.
GROOT_BASE_MODEL = "groot-n1.5"
DEFAULT_NUM_FRAMES = 9


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    collate = build_synthetic_droid_collate(args.num_frames, rng)
    datum = serde.collate_to_datum(collate)
    loss_keys = sorted(datum.loss_fn_inputs)
    print(
        f"built_datum=true base_model={GROOT_BASE_MODEL} "
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
            user_metadata={"example": "groot_sft_smoke"},
        )
        capabilities = client.get_server_capabilities()
        loss_families = set(capabilities.loss_families or [])
        if "cross_entropy" not in loss_families:
            advertised = ",".join(sorted(loss_families)) or "<none>"
            raise SystemExit(
                f"backend does not advertise cross_entropy; loss_families={advertised}"
            )

        # Building the GR00T VLA on first contact can take minutes.
        training = client.create_lora_training_client(
            base_model=GROOT_BASE_MODEL,
            rank=args.rank,
            user_metadata={"lora_alpha": str(args.rank), "example": "groot_sft_smoke"},
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
        checkpoint_name = args.checkpoint_name or f"groot-sft-smoke-{int(time.time())}"
        checkpoint = training.save_state(checkpoint_name).result(timeout=args.timeout)
        print(f"checkpoint={checkpoint}")
    finally:
        cleanup_remote_session(
            client,
            keep_lease=args.keep_lease,
            timeout=args.cleanup_timeout,
        )


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    return (
        args.api_key
        or os.environ.get("CYBERNETICS_API_KEY")
        or os.environ.get("WORLDLINES_API_KEY")
    )


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
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    parser.add_argument(
        "--keep-lease", action="store_true", help="Leave remote compute running after the example."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
