"""Read-only Cybernetics control-plane readiness checks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import click

from ..context import CLIContext
from ..exceptions import CyberneticsCliError
from ..output import OutputBase
from .auth import DEFAULT_BASE_URL


@dataclass(frozen=True)
class _ResolvedValue:
    value: str | None
    source: str


@dataclass(frozen=True)
class _HealthCheck:
    ok: bool
    path: str | None
    status_code: int | None
    detail: str | None = None


@dataclass(frozen=True)
class _Readiness:
    supports_training: bool | None
    supports_sampling: bool | None
    loss_families: list[str]
    dreamzero_sft_ready: bool
    dreamzero_rl_ready: bool | None
    dreamzero_rl_unavailable_reason: str | None


class DoctorOutput(OutputBase):
    def __init__(
        self,
        *,
        base_url: str,
        base_url_source: str,
        auth_source: str,
        health: _HealthCheck,
        capabilities: dict[str, Any],
        readiness: _Readiness,
        rl_loss: str,
    ) -> None:
        self.base_url = base_url
        self.base_url_source = base_url_source
        self.auth_source = auth_source
        self.health = health
        self.capabilities = capabilities
        self.readiness = readiness
        self.rl_loss = rl_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "base_url_source": self.base_url_source,
            "auth_source": self.auth_source,
            "health": {
                "ok": self.health.ok,
                "path": self.health.path,
                "status_code": self.health.status_code,
                "detail": self.health.detail,
            },
            "runtime": self.capabilities.get("runtime"),
            "scheduler_mode": self.capabilities.get("scheduler_mode"),
            "supports_training": self.readiness.supports_training,
            "supports_sampling": self.readiness.supports_sampling,
            "loss_families": self.readiness.loss_families,
            "dreamzero_sft_ready": self.readiness.dreamzero_sft_ready,
            "dreamzero_rl_ready": self.readiness.dreamzero_rl_ready,
            "dreamzero_rl_loss": self.rl_loss,
            "dreamzero_rl_unavailable_reason": (
                self.readiness.dreamzero_rl_unavailable_reason
            ),
        }

    def get_title(self) -> str | None:
        return "Cybernetics Doctor"

    def get_table_columns(self) -> list[str]:
        return ["Check", "Status", "Detail"]

    def get_table_rows(self) -> list[list[str]]:
        health_detail = (
            f"{self.health.path} HTTP {self.health.status_code}"
            if self.health.path and self.health.status_code
            else self.health.detail or "not checked"
        )
        return [
            ["API", "ready", f"{self.base_url} ({self.base_url_source})"],
            ["Auth", "ready", self.auth_source],
            ["Health", _status(self.health.ok), health_detail],
            ["Training", _status(self.readiness.supports_training), ""],
            ["Sampling", _status(self.readiness.supports_sampling), ""],
            ["DreamZero SFT", _status(self.readiness.dreamzero_sft_ready), "cross_entropy"],
            [
                "DreamZero RL",
                _status(self.readiness.dreamzero_rl_ready),
                self._rl_detail(),
            ],
        ]

    def _rl_detail(self) -> str:
        if self.readiness.dreamzero_rl_ready:
            return self.rl_loss
        if self.readiness.dreamzero_rl_unavailable_reason:
            return self.readiness.dreamzero_rl_unavailable_reason
        if self.rl_loss not in self.readiness.loss_families:
            return f"{self.rl_loss} not advertised by this backend"
        return "capability unknown"


@click.command()
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.option(
    "--api-key",
    default=None,
    help="API key override. Prefer CYBERNETICS_API_KEY or cybernetics auth login.",
)
@click.option(
    "--rl-loss",
    default="flow_rwr",
    show_default=True,
    help="DreamZero RL loss family to require or report.",
)
@click.option(
    "--require-rl",
    is_flag=True,
    help="Exit nonzero unless the backend advertises DreamZero RL readiness.",
)
@click.pass_obj
def cli(
    ctx: CLIContext | None,
    base_url: str | None,
    api_key: str | None,
    rl_loss: str,
    require_rl: bool,
) -> None:
    """Check API/auth/DreamZero readiness without creating a session or lease."""

    resolved_base = _resolve_base_url_with_source(base_url)
    resolved_key = _resolve_api_key_with_source(api_key)
    if not resolved_key.value:
        raise CyberneticsCliError(
            "No API key found.",
            "Run 'cybernetics auth login' or set CYBERNETICS_API_KEY.",
        )

    base = resolved_base.value or DEFAULT_BASE_URL
    health = _fetch_health(base)
    capabilities = _fetch_capabilities(base, resolved_key.value)
    readiness = _summarize_capabilities(capabilities, rl_loss)
    output = DoctorOutput(
        base_url=base,
        base_url_source=resolved_base.source,
        auth_source=resolved_key.source,
        health=health,
        capabilities=capabilities,
        readiness=readiness,
        rl_loss=rl_loss,
    )
    output.print((ctx or CLIContext()).format)

    if require_rl and readiness.dreamzero_rl_ready is not True:
        sys.exit(1)


def _resolve_api_key_with_source(explicit: str | None = None) -> _ResolvedValue:
    from cybernetics.lib.credentials import (
        API_KEY_ENV,
        CP_API_KEY_ENV,
        LEGACY_API_KEY_ENV,
        CredentialsError,
        load_credentials,
    )

    if explicit:
        return _ResolvedValue(explicit, "--api-key")
    if os.environ.get(API_KEY_ENV):
        return _ResolvedValue(os.environ[API_KEY_ENV], API_KEY_ENV)
    if os.environ.get(CP_API_KEY_ENV):
        return _ResolvedValue(os.environ[CP_API_KEY_ENV], CP_API_KEY_ENV)
    if os.environ.get(LEGACY_API_KEY_ENV):
        return _ResolvedValue(os.environ[LEGACY_API_KEY_ENV], LEGACY_API_KEY_ENV)
    try:
        stored = load_credentials()
    except CredentialsError:
        stored = None
    if stored and stored.api_key:
        return _ResolvedValue(stored.api_key, "stored login")
    return _ResolvedValue(None, "unset")


def _resolve_base_url_with_source(explicit: str | None = None) -> _ResolvedValue:
    from cybernetics.lib.credentials import (
        BASE_URL_ENV,
        CP_BASE_URL_ENV,
        CredentialsError,
        load_credentials,
    )

    if explicit:
        return _ResolvedValue(explicit.rstrip("/"), "--base-url")
    if os.environ.get(BASE_URL_ENV):
        return _ResolvedValue(os.environ[BASE_URL_ENV].rstrip("/"), BASE_URL_ENV)
    if os.environ.get(CP_BASE_URL_ENV):
        return _ResolvedValue(os.environ[CP_BASE_URL_ENV].rstrip("/"), CP_BASE_URL_ENV)
    try:
        stored = load_credentials()
    except CredentialsError:
        stored = None
    if stored and stored.base_url:
        return _ResolvedValue(stored.base_url.rstrip("/"), "stored login")
    return _ResolvedValue(DEFAULT_BASE_URL, "default")


def _fetch_health(base_url: str) -> _HealthCheck:
    import httpx

    last_detail: str | None = None
    with httpx.Client(base_url=base_url, timeout=10.0) as http:
        for path in ("/health", "/api/v1/healthz"):
            try:
                resp = http.get(path)
            except httpx.HTTPError as exc:
                last_detail = str(exc)
                continue
            if resp.status_code == 200:
                return _HealthCheck(ok=True, path=path, status_code=resp.status_code)
            last_detail = _safe_error_detail(resp) or f"HTTP {resp.status_code}"
    return _HealthCheck(ok=False, path=None, status_code=None, detail=last_detail)


def _fetch_capabilities(base_url: str, api_key: str) -> dict[str, Any]:
    import httpx

    try:
        with httpx.Client(base_url=base_url, timeout=15.0) as http:
            resp = http.get(
                "/api/v1/get_server_capabilities",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        raise CyberneticsCliError(f"Could not reach the control plane: {exc}") from exc
    if resp.status_code != 200:
        raise CyberneticsCliError(
            f"Capability check failed (HTTP {resp.status_code}).",
            _safe_error_detail(resp),
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise CyberneticsCliError("Capability check returned an invalid response.")
    return body


def _summarize_capabilities(capabilities: dict[str, Any], rl_loss: str) -> _Readiness:
    supports_training = _optional_bool(capabilities.get("supports_training"))
    supports_sampling = _optional_bool(capabilities.get("supports_sampling"))
    loss_families = [
        str(loss)
        for loss in capabilities.get("loss_families") or []
        if isinstance(loss, str)
    ]
    dreamzero_rl_available = _optional_bool(capabilities.get("dreamzero_rl_available"))
    dreamzero_rl_reason = _optional_str(capabilities.get("dreamzero_rl_unavailable_reason"))
    training_ready = supports_training is not False
    dreamzero_sft_ready = training_ready and "cross_entropy" in loss_families

    if dreamzero_rl_available is False:
        dreamzero_rl_ready = False
    elif rl_loss not in loss_families:
        dreamzero_rl_ready = False if loss_families else None
    elif dreamzero_rl_available is None:
        dreamzero_rl_ready = None
    else:
        dreamzero_rl_ready = True

    return _Readiness(
        supports_training=supports_training,
        supports_sampling=supports_sampling,
        loss_families=loss_families,
        dreamzero_sft_ready=dreamzero_sft_ready,
        dreamzero_rl_ready=dreamzero_rl_ready,
        dreamzero_rl_unavailable_reason=dreamzero_rl_reason,
    )


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _status(value: bool | None) -> str:
    if value is True:
        return "ready"
    if value is False:
        return "unavailable"
    return "unknown"


def _safe_error_detail(resp: object) -> str | None:
    try:
        return str(resp.json())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
