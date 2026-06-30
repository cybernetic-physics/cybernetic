"""Task integrity lock: sha256 digests of every judge file, with an optional ed25519 signature.

The sha256 digests give fast, secrets-free drift/tamper detection (a policy PR that edits a
judge file fails ``verify-task`` with EXIT_CONTRACT). The optional ed25519 signature lets a
task OWNER attest the lock so a forged lock (recomputed digests) is also rejected -- useful for
cross-repo / multi-owner trust. Neither is the load-bearing anchor: that is the authoritative
``pull_request_target`` check, which grades against the canonical task bytes on the protected
base ref regardless of anything in the PR. These layers are fast feedback + defense-in-depth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas import ContractError

TASKLOCK_SCHEMA_VERSION = "behavior-ci-tasklock/v2"

try:  # optional asymmetric signing
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    _HAVE_ED25519 = True
except Exception:  # pragma: no cover - crypto is optional
    _HAVE_ED25519 = False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TaskLock:
    """Integrity record for an in-repo task pack (taskkit v2)."""

    task_id: str
    task_version: str
    grader_entrypoint: str
    digests: Dict[str, str]  # repo-relative-path -> sha256
    signature: Optional[Dict[str, str]] = None  # {alg, key_id, sig(hex)}
    schema_version: str = TASKLOCK_SCHEMA_VERSION
    candidate_copies: Dict[str, str] = field(
        default_factory=dict
    )  # back-compat (unused for repo tasks)

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskLock":
        return cls(
            task_id=data.get("task_id", ""),
            task_version=str(data.get("task_version", "0")),
            grader_entrypoint=data.get("grader_entrypoint", "behavior_ci_run_trial"),
            digests=dict(data.get("digests", {})),
            signature=data.get("signature"),
            schema_version=data.get("schema_version", TASKLOCK_SCHEMA_VERSION),
            candidate_copies=dict(data.get("candidate_copies", {})),
        )

    def to_dict(self) -> Dict:
        d = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "grader_entrypoint": self.grader_entrypoint,
            "digests": self.digests,
        }
        if self.signature:
            d["signature"] = self.signature
        return d


def _signed_payload(task_id: str, task_version: str, digests: Dict[str, str]) -> bytes:
    return json.dumps(
        {"task_id": task_id, "task_version": task_version, "digests": digests},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_digests(task_dir: Path, files: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in sorted(files):
        out[rel] = sha256_bytes((Path(task_dir) / rel).read_bytes())
    return out


def sign_lock(lock: TaskLock, private_key_pem: bytes, key_id: str) -> TaskLock:
    """Attest a lock with an ed25519 private key (task-owner only)."""
    if not _HAVE_ED25519:  # pragma: no cover
        raise ContractError("ed25519 signing needs the 'cryptography' package")
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ContractError("signing key is not an ed25519 private key")
    sig = key.sign(_signed_payload(lock.task_id, lock.task_version, lock.digests)).hex()
    return TaskLock(
        task_id=lock.task_id,
        task_version=lock.task_version,
        grader_entrypoint=lock.grader_entrypoint,
        digests=lock.digests,
        signature={"alg": "ed25519", "key_id": key_id, "sig": sig},
    )


def verify_lock(task_dir: Path, lock: TaskLock, pubkey_pem: Optional[bytes] = None) -> None:
    """Recompute every judge-file digest == lock, then (if signed + pubkey given) the signature.

    Raises ContractError on ANY mismatch. A missing signature is allowed (sha256 still enforced);
    a present signature with no/failed pubkey verification is rejected when a pubkey is provided.
    """
    task_dir = Path(task_dir)
    mismatches = []
    for rel, expected in lock.digests.items():
        p = task_dir / rel
        if not p.exists():
            mismatches.append(f"{rel}: missing")
            continue
        actual = sha256_bytes(p.read_bytes())
        if actual != expected:
            mismatches.append(f"{rel}: {expected[:12]}!= {actual[:12]}")
    if mismatches:
        raise ContractError(
            "task lock digest mismatch (a judge file was edited):\n  - " + "\n  - ".join(mismatches)
        )
    if lock.signature and pubkey_pem is not None:
        if not _HAVE_ED25519:  # pragma: no cover
            raise ContractError("task lock is signed but 'cryptography' is unavailable to verify")
        from cryptography.hazmat.primitives import serialization

        pub = serialization.load_pem_public_key(pubkey_pem)
        if not isinstance(pub, Ed25519PublicKey):
            raise ContractError("task pubkey is not an ed25519 public key")
        try:
            pub.verify(
                bytes.fromhex(lock.signature["sig"]),
                _signed_payload(lock.task_id, lock.task_version, lock.digests),
            )
        except InvalidSignature:
            raise ContractError("task lock signature is invalid")
