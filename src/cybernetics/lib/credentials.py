"""Cybernetics API-key resolution and the local login credential store.

This module owns one invariant: the on-disk CLI login credential at
``$XDG_CONFIG_HOME/cybernetics/auth.json`` (``0700`` directory, ``0600`` file,
written atomically) and the canonical key-resolution order used by both the SDK
client and the ``cybernetics`` CLI:

    1. an explicit ``api_key=`` argument,
    2. the ``CYBERNETICS_API_KEY`` environment variable,
    3. the short ``CP_API_KEY`` environment variable used in Cybernetic Physics
       examples and dev-infra runbooks,
    4. the deprecated ``WORLDLINES_API_KEY`` fallback for one release,
    5. the stored login file written by ``cybernetics auth login``.

It depends only on the standard library (no ``click`` / ``httpx`` / ``rich``) so
it can be imported from the low-level client without pulling in CLI machinery.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

API_KEY_ENV = "CYBERNETICS_API_KEY"
CP_API_KEY_ENV = "CP_API_KEY"
# Deprecated: read for one release so existing exports keep working.
LEGACY_API_KEY_ENV = "WORLDLINES_API_KEY"
BASE_URL_ENV = "CYBERNETICS_BASE_URL"
CP_BASE_URL_ENV = "CP_API_BASE"

_DIR_MODE = 0o700
_FILE_MODE = 0o600


@dataclass(frozen=True)
class StoredCredentials:
    """A login credential persisted by ``cybernetics auth login``."""

    api_key: str
    base_url: str | None = None
    user: str | None = None
    workspace: str | None = None
    key_id: str | None = None
    saved_at: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "user": self.user,
            "workspace": self.workspace,
            "key_id": self.key_id,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "StoredCredentials":
        api_key = data.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise CredentialsError("stored credential file is missing an 'api_key'")
        return cls(
            api_key=api_key,
            base_url=_opt_str(data.get("base_url")),
            user=_opt_str(data.get("user")),
            workspace=_opt_str(data.get("workspace")),
            key_id=_opt_str(data.get("key_id")),
            saved_at=data.get("saved_at")
            if isinstance(data.get("saved_at"), (int, float))
            else None,
        )


class CredentialsError(Exception):
    """The local credential file could not be read, parsed, or written."""


def config_dir() -> Path:
    """Return the Cybernetics config directory (honoring ``XDG_CONFIG_HOME``)."""

    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "cybernetics"


def credentials_path() -> Path:
    """Return the path to the login credential file."""

    return config_dir() / "auth.json"


def load_credentials() -> StoredCredentials | None:
    """Load the stored login credential, or ``None`` if there is no login file.

    Warns (but does not fail) when the file permissions are looser than ``0600``
    so a misconfigured secret is visible to the operator.
    """

    path = credentials_path()
    if not path.exists():
        return None
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            warnings.warn(
                f"Credential file {path} is group/other-accessible (mode {oct(mode)}); "
                f"run 'chmod 600 {path}'.",
                stacklevel=2,
            )
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CredentialsError(f"could not read credential file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CredentialsError(f"credential file {path} is not a JSON object")
    return StoredCredentials.from_json(data)


def save_credentials(credentials: StoredCredentials) -> Path:
    """Atomically persist ``credentials`` with ``0700`` dir / ``0600`` file modes."""

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, _DIR_MODE)
    path = credentials_path()

    payload = dict(credentials.to_json())
    if payload.get("saved_at") is None:
        payload["saved_at"] = time.time()

    # Atomic write: temp file in the same dir + os.replace, with 0600 from creation.
    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=".auth-", suffix=".json")
    try:
        os.fchmod(fd, _FILE_MODE)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except OSError as exc:
        _silent_unlink(tmp_name)
        raise CredentialsError(f"could not write credential file {path}: {exc}") from exc
    os.chmod(path, _FILE_MODE)
    return path


def delete_credentials() -> bool:
    """Delete the login credential file. Returns ``True`` if a file was removed."""

    path = credentials_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CredentialsError(f"could not delete credential file {path}: {exc}") from exc


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the Cybernetics API key by precedence; ``None`` if unset everywhere.

    Order: explicit arg -> ``CYBERNETICS_API_KEY`` -> ``CP_API_KEY`` ->
    deprecated ``WORLDLINES_API_KEY`` -> the stored login file.
    """

    if explicit:
        return explicit
    env = os.environ.get(API_KEY_ENV)
    if env:
        return env
    cp_env = os.environ.get(CP_API_KEY_ENV)
    if cp_env:
        return cp_env
    legacy = os.environ.get(LEGACY_API_KEY_ENV)
    if legacy:
        warnings.warn(
            f"{LEGACY_API_KEY_ENV} is deprecated; set {API_KEY_ENV} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    stored = _safe_load()
    return stored.api_key if stored else None


def resolve_base_url(explicit: str | None = None) -> str | None:
    """Resolve the control-plane base URL: explicit -> env -> stored login file."""

    if explicit:
        return explicit
    env = os.environ.get(BASE_URL_ENV)
    if env:
        return env
    cp_env = os.environ.get(CP_BASE_URL_ENV)
    if cp_env:
        return cp_env
    stored = _safe_load()
    return stored.base_url if stored else None


def _safe_load() -> StoredCredentials | None:
    try:
        return load_credentials()
    except CredentialsError:
        return None


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _silent_unlink(name: str) -> None:
    try:
        os.unlink(name)
    except OSError:
        # Best-effort cleanup for temp files; callers should not fail on already-removed paths.
        pass
