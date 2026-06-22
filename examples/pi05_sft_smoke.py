#!/usr/bin/env python3
"""openpi pi0.5 full-finetune SFT smoke example for the Cybernetics SDK.

This is the CLIENT side of the pi0.5 integration. Per the client-side serde
decision it builds a pi0.5 ``collate`` dict locally (image stack, normalized
actions/state, tokenized prompt) with ``cybernetics.lib.pi05.build_pi05_collate``,
encodes it into a single ``Datum`` with ``serde.collate_to_datum``, and ships it
over the wire. ``Pi05Runtime`` decodes it into an openpi ``Observation`` +
actions and calls ``PI0Pytorch.forward``; NO openpi dataset is instantiated
server-side.

pi0.5 is FULL FINETUNE: ``create_lora_training_client`` is used for the
registration handshake but ``rank`` is IGNORED by the runtime (every parameter
trains). The ``loss_fn`` shipped on ``forward_backward`` is ALSO ignored
server-side -- the loss is the model-internal flow-matching velocity MSE; we send
the wire literal ``"cross_entropy"`` (the wire LossFnType Literal has no
flow-matching name and the runtime ignores the value regardless).

Default mode is local-only: it builds the synthetic pi0.5 batch, converts it into
a Cybernetics Datum, and prints the wire keys. Use ``--remote-run`` only when you
are ready to create a Worldlines session/model and spend GPU time on the
configured control plane. Remote runs cancel their SDK session on exit by default
so successful smokes do not leave paid compute running; pass ``--keep-lease`` for
debugging.

NOTE: a REAL collate dict needs an image loader + the PaliGemma tokenizer + the
checkpoint norm_stats applied to actions/state. ``build_pi05_collate`` is a
clearly-marked SYNTHETIC fixture (numpy only, no openpi) producing a
shape-correct collate so the serde / forward-backward / optim / save path is
exercisable without GPU/data.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from dreamzero_sft_smoke import cleanup_remote_session
from groot_sft_smoke import _resolve_api_key

import cybernetics
from cybernetics import types
from cybernetics.lib.pi05 import PI05_BASE_MODEL, serde

# pi0.5 optimizer recipe (mirrors hosted_models.PI05_RECIPE["optimizer"]).
PI05_BETA1, PI05_BETA2 = 0.9, 0.95
PI05_EPS = 1e-8
PI05_WEIGHT_DECAY = 1e-5
PI05_GRAD_CLIP = 1.0


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    collate = serde.build_pi05_collate(rng)
    datum = serde.collate_to_datum(collate)
    loss_keys = sorted(datum.loss_fn_inputs)
    print(
        f"built_datum=true base_model={PI05_BASE_MODEL} "
        f"loss_keys={len(loss_keys)} prompt_len={collate['tokenized_prompt'].shape[-1]}"
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
            user_metadata={"example": "pi05_sft_smoke"},
        )
        # Full-FT: rank is ignored by Pi05Runtime (every parameter trains).
        training = client.create_lora_training_client(
            base_model=PI05_BASE_MODEL,
            rank=args.rank,
            timeout=args.timeout,
        )
        print(
            f"created_training_client=true model_id={training.model_id} "
            f"base={PI05_BASE_MODEL} mode=full-finetune"
        )

        # loss_fn is ignored server-side (the loss is the model-internal
        # flow-matching MSE); ship "cross_entropy".
        fb = training.forward_backward([datum], "cross_entropy").result(timeout=args.timeout)
        print(f"forward_backward_done=true model_id={training.model_id} result={fb}")

        adam = types.AdamParams(
            learning_rate=args.learning_rate,
            beta1=PI05_BETA1,
            beta2=PI05_BETA2,
            eps=PI05_EPS,
            weight_decay=PI05_WEIGHT_DECAY,
            grad_clip_norm=PI05_GRAD_CLIP,
        )
        training.optim_step(adam).result(timeout=args.timeout)
        checkpoint_name = args.checkpoint_name or f"pi05-sft-smoke-{int(time.time())}"
        checkpoint = training.save_state(checkpoint_name).result(timeout=args.timeout)
        print(f"checkpoint={checkpoint}")
    finally:
        cleanup_remote_session(
            client,
            keep_lease=args.keep_lease,
            timeout=args.cleanup_timeout,
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
    parser.add_argument(
        "--rank", type=int, default=1, help="Ignored server-side (pi0.5 is full-finetune)."
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    parser.add_argument(
        "--keep-lease", action="store_true", help="Leave remote compute running after the example."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
