"""The ``cybernetics auth`` command group — browser login + key management.

``login`` runs an OAuth 2.0 Device Authorization Grant (RFC 8628) against the
control plane and stores the resulting long-lived ``cp_live_`` key at
``~/.config/cybernetics/auth.json`` (see :mod:`cybernetics.lib.credentials`).
``logout`` revokes + removes it, ``status`` shows who you are, and ``token``
prints the resolved key for ``export CYBERNETICS_API_KEY=$(cybernetics auth token)``.

All heavy imports (``httpx``, ``webbrowser``, ``time``) are lazy so the top-level
CLI stays fast.
"""

import sys

import click

from ..exceptions import CyberneticsCliError

# The control-plane API origin (NOT the site origin). Override with
# CYBERNETICS_BASE_URL, a stored login's base_url, or --base-url.
DEFAULT_BASE_URL = "https://api.cyberneticphysics.com"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


def _resolve_base_url(explicit: str | None) -> str:
    from cybernetics.lib.credentials import resolve_base_url

    return (resolve_base_url(explicit) or DEFAULT_BASE_URL).rstrip("/")


@click.group()
def cli() -> None:
    """Manage Cybernetics authentication."""


@cli.command()
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.option("--client-name", default="cybernetics-cli", help="Label for the minted key.")
@click.option("--no-browser", is_flag=True, help="Do not open a browser automatically.")
def login(base_url: str | None, client_name: str, no_browser: bool) -> None:
    """Log in via the browser (device authorization grant)."""
    import time
    import webbrowser

    import httpx

    from cybernetics.lib.credentials import StoredCredentials, save_credentials

    base = _resolve_base_url(base_url)
    with httpx.Client(base_url=base, timeout=30.0) as http:
        start = _request_device_code(http, client_name)
        _print_login_prompt(start["user_code"], start["verification_uri"])
        if not no_browser:
            try:
                click.pause(info="Press Enter to open the browser…", err=True)
            except Exception:  # noqa: BLE001 -- non-interactive stdin
                pass
            webbrowser.open(start["verification_uri"])

        key_payload = _poll_for_token(
            http,
            device_code=start["device_code"],
            interval=start["interval"],
            expires_in=start["expires_in"],
            sleep=time.sleep,
            monotonic=time.monotonic,
        )

    credentials = StoredCredentials(
        api_key=key_payload["access_token"],
        base_url=base,
        key_id=key_payload.get("api_key_id"),
        workspace=key_payload.get("workspace_id"),
    )
    me = _whoami(base, credentials.api_key)
    credentials = StoredCredentials(
        api_key=credentials.api_key,
        base_url=base,
        user=(me.get("user") or {}).get("login") or (me.get("user") or {}).get("email"),
        workspace=me.get("activeWorkspaceId") or credentials.workspace,
        key_id=credentials.key_id,
    )
    save_credentials(credentials)
    user = credentials.user or "you"
    workspace = credentials.workspace or "default"
    click.echo(f"✓ Logged in as {user} (workspace {workspace})")


@cli.command()
def logout() -> None:
    """Revoke the stored key and delete the local credential file."""
    import httpx

    from cybernetics.lib.credentials import delete_credentials, load_credentials

    stored = load_credentials()
    if stored is None:
        click.echo("Not logged in.")
        return
    if stored.key_id and stored.base_url:
        try:
            with httpx.Client(base_url=stored.base_url, timeout=15.0) as http:
                http.delete(
                    f"/v1/api-keys/{stored.key_id}",
                    headers={"Authorization": f"Bearer {stored.api_key}"},
                )
        except httpx.HTTPError:
            # Best-effort revocation; still drop the local file so we're logged out.
            pass
    delete_credentials()
    click.echo("✓ Logged out.")


@cli.command()
def status() -> None:
    """Show the current login (whoami)."""
    from cybernetics.lib.credentials import load_credentials, resolve_api_key

    key = resolve_api_key()
    if not key:
        click.echo("Not logged in. Run 'cybernetics auth login'.")
        sys.exit(1)
    stored = load_credentials()
    base = _resolve_base_url(stored.base_url if stored else None)
    me = _whoami(base, key)
    source = "stored login" if stored and stored.api_key == key else "environment"
    prefix = key[:16]
    me_user = me.get("user") or {}
    click.echo("Logged in")
    click.echo(f"  User:      {me_user.get('login') or me_user.get('email') or 'unknown'}")
    click.echo(
        f"  Workspace: {me.get('activeWorkspaceId') or (stored.workspace if stored else 'unknown')}"
    )
    click.echo(f"  Key:       {prefix}…")
    click.echo(f"  Source:    {source}")


@cli.command()
def token() -> None:
    """Print the resolved API key to stdout."""
    from cybernetics.lib.credentials import resolve_api_key

    key = resolve_api_key()
    if not key:
        raise CyberneticsCliError(
            "No API key found.",
            "Run 'cybernetics auth login' or set CYBERNETICS_API_KEY.",
        )
    click.echo(key)


# ---------------------------------------------------------------------------
# Helpers (small, testable units)
# ---------------------------------------------------------------------------


def _request_device_code(http: object, client_name: str) -> dict:
    resp = http.post("/v1/auth/device/code", json={"client_name": client_name})  # type: ignore[attr-defined]
    if resp.status_code != 200:
        raise CyberneticsCliError(
            f"Could not start device login (HTTP {resp.status_code}).",
            _safe_error_detail(resp),
        )
    data = resp.json()
    return {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": data.get("verification_uri_complete") or data["verification_uri"],
        "interval": int(data.get("interval", 5)),
        "expires_in": int(data.get("expires_in", 900)),
    }


def _print_login_prompt(user_code: str, verification_uri: str) -> None:
    click.echo("", err=True)
    click.secho(f"  First copy your one-time code: {user_code}", bold=True, err=True)
    click.echo(f"  Then open: {verification_uri}", err=True)
    click.echo("", err=True)


def _poll_for_token(
    http: object,
    *,
    device_code: str,
    interval: int,
    expires_in: int,
    sleep,
    monotonic,
) -> dict:
    """Poll the token endpoint until approved, honoring slow_down + the deadline."""
    deadline = monotonic() + expires_in
    wait = interval
    while monotonic() < deadline:
        sleep(wait)
        resp = http.post(  # type: ignore[attr-defined]
            "/v1/auth/device/token",
            json={"grant_type": DEVICE_GRANT_TYPE, "device_code": device_code},
        )
        body = resp.json()
        if resp.status_code == 200:
            return body
        error = body.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            wait += 5
            continue
        if error == "expired_token":
            raise CyberneticsCliError("The login code expired. Run 'cybernetics auth login' again.")
        raise CyberneticsCliError(f"Login failed: {error or 'unknown error'}")
    raise CyberneticsCliError("Login timed out before approval.")


def _whoami(base_url: str, api_key: str) -> dict:
    import httpx

    try:
        with httpx.Client(base_url=base_url, timeout=15.0) as http:
            resp = http.get("/v1/me", headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        raise CyberneticsCliError(f"Could not reach the control plane: {exc}")
    if resp.status_code != 200:
        raise CyberneticsCliError(
            f"Authentication check failed (HTTP {resp.status_code}).",
            _safe_error_detail(resp),
        )
    return resp.json()


def _safe_error_detail(resp: object) -> str | None:
    try:
        return str(resp.json())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
