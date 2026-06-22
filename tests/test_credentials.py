"""Contract: the local credential store is owner-only, atomic, and resolves keys
in the documented order. This is the SDK/CLI shared secret boundary.
"""

from __future__ import annotations

import json
import stat
import warnings
from pathlib import Path

import pytest

from cybernetics.lib import credentials
from cybernetics.lib.credentials import StoredCredentials


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for var in (
        "CYBERNETICS_API_KEY",
        "CP_API_KEY",
        "WORLDLINES_API_KEY",
        "CYBERNETICS_BASE_URL",
        "CP_API_BASE",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_save_then_load_round_trip() -> None:
    creds = StoredCredentials(
        api_key="cp_live_abc",
        base_url="https://api.example.com/v1",
        user="alice",
        workspace="ws_1",
        key_id="key_1",
    )
    path = credentials.save_credentials(creds)
    loaded = credentials.load_credentials()
    assert loaded is not None
    assert loaded.api_key == "cp_live_abc"
    assert loaded.base_url == "https://api.example.com/v1"
    assert loaded.key_id == "key_1"
    assert json.loads(path.read_text())["saved_at"] is not None  # stamped on save


def test_file_and_dir_modes_are_owner_only() -> None:
    path = credentials.save_credentials(StoredCredentials(api_key="cp_live_x"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_load_warns_on_loose_permissions() -> None:
    path = credentials.save_credentials(StoredCredentials(api_key="cp_live_x"))
    path.chmod(0o644)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        credentials.load_credentials()
    assert any("accessible" in str(w.message) for w in caught)


def test_delete_credentials() -> None:
    credentials.save_credentials(StoredCredentials(api_key="cp_live_x"))
    assert credentials.delete_credentials() is True
    assert credentials.delete_credentials() is False  # idempotent
    assert credentials.load_credentials() is None


def test_resolve_api_key_falls_back_to_stored_file() -> None:
    assert credentials.resolve_api_key() is None  # nothing set anywhere
    credentials.save_credentials(
        StoredCredentials(api_key="cp_live_stored", base_url="https://api/v1")
    )
    assert credentials.resolve_api_key() == "cp_live_stored"
    assert credentials.resolve_base_url() == "https://api/v1"
    assert credentials.resolve_api_key("explicit") == "explicit"  # explicit still wins


def test_resolve_api_key_accepts_cp_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CP_API_KEY", "cp_live_alias")
    monkeypatch.setenv("CP_API_BASE", "https://luc-api.cyberneticphysics.com")

    assert credentials.resolve_api_key() == "cp_live_alias"
    assert credentials.resolve_base_url() == "https://luc-api.cyberneticphysics.com"


def test_cybernetics_env_wins_over_cp_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_canonical")
    monkeypatch.setenv("CP_API_KEY", "cp_live_alias")
    monkeypatch.setenv("CYBERNETICS_BASE_URL", "https://api.cyberneticphysics.com")
    monkeypatch.setenv("CP_API_BASE", "https://luc-api.cyberneticphysics.com")

    assert credentials.resolve_api_key() == "cp_live_canonical"
    assert credentials.resolve_base_url() == "https://api.cyberneticphysics.com"


def test_save_is_atomic_no_partial_files(_isolated_config: Path) -> None:
    credentials.save_credentials(StoredCredentials(api_key="cp_live_x"))
    # No leftover temp files in the config dir (atomic temp+replace).
    leftovers = list((_isolated_config / "cybernetics").glob(".auth-*"))
    assert leftovers == []
