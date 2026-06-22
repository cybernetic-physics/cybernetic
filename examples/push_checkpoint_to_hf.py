#!/usr/bin/env python3
"""Push a Worldlines checkpoint to a HuggingFace repo.

A Worldlines ``save_state`` writes its artifact to the backend's on-disk
``artifact_root`` as ``/data/<run>/weights/<name>/`` containing:

    model.safetensors   the trained weights (LoRA adapters or full-FT params)
    config.json         the model/recipe config
    metadata.pt         training metadata (step, base_model, optimizer state ref)
    COMPLETE            a sentinel written last; its presence means the dir is done

This helper uploads such a directory to a HuggingFace repo via ``huggingface_hub``
(``HfApi.create_repo(exist_ok=True)`` + ``upload_folder``). The HF WRITE token is
read from the ``HF_TOKEN`` env var -- it is NEVER taken on the command line and
NEVER printed.

USAGE:
    export HF_TOKEN=hf_...            # a WRITE token; never hardcode it
    python push_checkpoint_to_hf.py \
        --checkpoint-dir /data/<run>/weights/<name> \
        --repo-id your-org/your-model \
        --private

The ``--checkpoint-dir`` is the on-disk weights dir described above. If your
checkpoint is only reachable through the SDK/REST archive endpoint, download it
first (e.g. ``rest_client`` weight export) into a local dir and point
``--checkpoint-dir`` at the extracted folder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXPECTED_FILES = ("model.safetensors", "config.json")
SENTINEL = "COMPLETE"


def main() -> None:
    args = _parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    _validate_checkpoint(checkpoint_dir)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN env var is required (a HuggingFace WRITE token); refusing to proceed."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo_ready=true repo_id={args.repo_id} private={args.private}", flush=True)

    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        commit_message=args.commit_message or f"Upload worldlines checkpoint {checkpoint_dir.name}",
    )
    print(f"upload_done=true repo_id={args.repo_id} commit={commit}", flush=True)


def _validate_checkpoint(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint dir not found or not a directory: {checkpoint_dir}")
    if not (checkpoint_dir / SENTINEL).exists():
        print(
            f"warning: no {SENTINEL} sentinel in {checkpoint_dir}; the save may be incomplete.",
            file=sys.stderr,
            flush=True,
        )
    missing = [name for name in EXPECTED_FILES if not (checkpoint_dir / name).exists()]
    if missing:
        print(
            f"warning: missing expected files {missing} in {checkpoint_dir}; uploading anyway.",
            file=sys.stderr,
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="On-disk weights dir, e.g. /data/<run>/weights/<name>.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo id, e.g. your-org/your-model.",
    )
    parser.add_argument("--private", action="store_true", help="Create/keep the HF repo private.")
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message for the upload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
