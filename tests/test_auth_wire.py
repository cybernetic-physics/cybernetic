"""Contract: the SDK authenticates with ``Authorization: Bearer`` and never ships
a stray ``X-API-Key`` (the three mandatory wire changes + the double-header guard).
"""

from __future__ import annotations

import warnings

import pytest

CP_LIVE_KEY = "cp_live_" + "a" * 64


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CYBERNETICS_API_KEY",
        "WORLDLINES_API_KEY",
        "CYBERNETICS_BASE_URL",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(var, raising=False)


def test_client_accepts_cp_live_key_and_emits_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYBERNETICS_API_KEY", CP_LIVE_KEY)
    import cybernetics

    # No 'wld-' assertion: a cp_live_ customer key constructs cleanly.
    client = cybernetics.AsyncCybernetics()
    assert client.auth_headers == {"Authorization": f"Bearer {CP_LIVE_KEY}"}
    assert "X-API-Key" not in client.auth_headers


def test_default_headers_carry_no_x_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYBERNETICS_API_KEY", CP_LIVE_KEY)
    import cybernetics

    headers = cybernetics.AsyncCybernetics().default_headers
    assert any(k == "Authorization" for k in headers)
    assert not any(k.lower() == "x-api-key" for k in headers)


def test_service_client_default_headers_use_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYBERNETICS_API_KEY", CP_LIVE_KEY)
    from cybernetics.lib.public_interfaces.service_client import _get_default_headers

    headers = _get_default_headers()
    assert headers.get("Authorization") == f"Bearer {CP_LIVE_KEY}"
    assert "X-API-Key" not in headers


def test_resolve_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from cybernetics.lib.credentials import resolve_api_key

    monkeypatch.setenv("CYBERNETICS_API_KEY", "from_env")
    assert resolve_api_key("explicit") == "explicit"
    assert resolve_api_key() == "from_env"


def test_resolve_api_key_deprecated_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from cybernetics.lib.credentials import resolve_api_key

    monkeypatch.delenv("CYBERNETICS_API_KEY", raising=False)
    monkeypatch.setenv("WORLDLINES_API_KEY", "cp_live_legacy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_api_key() == "cp_live_legacy"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_get_tokenizer_raises_install_on_demand_error() -> None:
    from cybernetics.lib.public_interfaces.sampling_client import _load_tokenizer_from_model_info

    with pytest.raises(ImportError, match=r"cybernetic-physics\[tokenizers\]"):
        _load_tokenizer_from_model_info("Qwen/Qwen3-8B")
