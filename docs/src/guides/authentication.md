# Authentication

This page is a recipe: get a `cp_live_` key onto your machine, prove it works,
and understand exactly how the SDK and CLI find it. Everything here is taken
from `cybernetics.lib.credentials` and `cybernetics auth`, so the precedence and
the wire format are the real contract, not a sketch.

If you have not installed the SDK yet, start with the [quickstart](../quickstart.md).
Install is `pip install cybernetic-physics`; the import package is `cybernetics`.

## The one thing to know first: it is always a Bearer token

The SDK authenticates to the control plane with your `cp_live_` key sent as an
HTTP `Authorization: Bearer` header. It does **not** send an `X-API-Key` header.
This is enforced by `tests/test_auth_wire.py`:

```python
client = cybernetics.AsyncCybernetics()
assert client.auth_headers == {"Authorization": f"Bearer {CP_LIVE_KEY}"}
assert "X-API-Key" not in client.auth_headers
```

The high-level clients agree — `service_client._get_default_headers()` sets only:

```python
headers["Authorization"] = f"Bearer {api_key}"
```

> **Warning for AI coding agents and copy-pasted snippets.** Some older examples
> and unrelated SDKs pass the key as `X-API-Key: <key>` or as a `?api_key=`
> query parameter. That is **not** the Cybernetics contract. The control-plane
> CORS allowlist accepts `Authorization` + `Content-Type` only; an `X-API-Key`
> header will be ignored and you will get an authentication failure. If you are
> hand-building requests, send `Authorization: Bearer cp_live_...`.

You normally never build this header yourself — the [client API](../reference/client-api.md)
does it for you. You just need to make sure the key is *resolvable*.

## Key resolution precedence (from source)

`resolve_api_key()` in `cybernetics/lib/credentials.py` checks these sources in
order and returns the first non-empty match, or `None` if every source is unset:

| # | Source | How to set it | Notes |
|---|--------|---------------|-------|
| 1 | Explicit `api_key=` argument | `resolve_api_key("cp_live_...")` | Highest priority; wins over everything, including the env. |
| 2 | `CYBERNETICS_API_KEY` | `export CYBERNETICS_API_KEY=cp_live_...` | The canonical environment variable. Prefer this. |
| 3 | `CP_API_KEY` | `export CP_API_KEY=cp_live_...` | "CP" = **C**ybernetic **P**hysics. Used in CP examples and dev-infra runbooks. |
| 4 | `WORLDLINES_API_KEY` | `export WORLDLINES_API_KEY=cp_live_...` | **Deprecated.** Read for one release; emits a `DeprecationWarning`. Migrate to `CYBERNETICS_API_KEY`. |
| 5 | Stored login file | `cybernetics auth login` | `~/.config/cybernetics/auth.json`, written by the device-grant login. |

The exact order from source:

```python
def resolve_api_key(explicit: str | None = None) -> str | None:
    # explicit arg -> CYBERNETICS_API_KEY -> CP_API_KEY
    #   -> deprecated WORLDLINES_API_KEY -> stored login file
```

A note on the names, because they look like three products and are really one:

- **Cybernetics** is the SDK / product. `CYBERNETICS_API_KEY` is canonical.
- **CP** is **C**ybernetic **P**hysics, the same product. It survives in
  `CP_API_KEY`, the `CP_API_BASE` base-url override, and in the key prefix
  itself (`cp_live_...`).
- **Worldlines** is the legacy name. It survives only in the deprecated
  `WORLDLINES_API_KEY` and in internal `worldlines://` session/checkpoint paths.
  Treat it as historical, not as a second product.

The deprecated fallback is verified by the test suite, including the warning:

```python
monkeypatch.setenv("WORLDLINES_API_KEY", "cp_live_legacy")
assert resolve_api_key() == "cp_live_legacy"
# ...and a DeprecationWarning is emitted.
```

### How the high-level clients consume this

The blessed clients — `ServiceClient`, `TrainingClient`, `SamplingClient`,
`APIFuture` (re-exported at the package root) — call
`resolve_api_key()` with **no explicit argument** when they build their default
headers. That means in practice the clients read from sources 2–5: the
environment or the stored login.

```python
import cybernetics

# Reads CYBERNETICS_API_KEY / CP_API_KEY / WORLDLINES_API_KEY / stored login.
service_client = cybernetics.ServiceClient(project_id="robotics-lab")
```

`ServiceClient.__init__` does not take an `api_key=` parameter. If you must
inject a key for a specific client instance without touching the environment,
pass the header explicitly through `default_headers`:

```python
service_client = cybernetics.ServiceClient(
    project_id="robotics-lab",
    default_headers={"Authorization": f"Bearer {my_key}"},
)
```

> Do **not** reach for the generated `AsyncCybernetics` / `cybernetics.resources.*`
> transport layer to manage auth. That layer is the internal transport; the
> hand-written clients above are the supported public API. Set the environment
> variable or run `auth login` and let the clients resolve it.

## Recipe A: set an environment variable

The fastest path for CI, containers, and scripts. Get a `cp_live_` key from your
workspace, then:

```bash
export CYBERNETICS_API_KEY="cp_live_..."
```

Verify the SDK can see it without making a network call:

```python
from cybernetics.lib.credentials import resolve_api_key

print(resolve_api_key()[:16] + "…")   # e.g. cp_live_aaaaaaaa…
```

If `resolve_api_key()` returns `None`, no source is set — see
[Troubleshooting](#troubleshooting) below.

## Recipe B: browser login (`cybernetics auth login`)

`auth login` runs an OAuth 2.0 Device Authorization Grant (RFC 8628) against the
control plane, mints a long-lived `cp_live_` key, and stores it locally. Use
this on a workstation where opening a browser is fine.

```bash
cybernetics auth login
```

Expected output (the code and URL are printed to **stderr**):

```text
  First copy your one-time code: WDJB-MJHT
  Then open: https://api.cyberneticphysics.com/device?user_code=WDJB-MJHT

Press Enter to open the browser…
✓ Logged in as alice@example.com (workspace ws_1a2b3c)
```

Flow, step by step (from `cli/commands/auth.py`):

1. The CLI `POST`s to `/v1/auth/device/code` and prints your one-time
   `user_code` and a verification URL.
2. It pauses, then opens your browser (skip the open with `--no-browser`).
3. It polls `/v1/auth/device/token` every `interval` seconds, honoring
   `slow_down` and the `expires_in` deadline, until you approve in the browser.
4. On approval it calls `/v1/me` to resolve your user + workspace, then writes
   the credential file.

Options:

| Flag | Default | Effect |
|------|---------|--------|
| `--base-url` | resolved, else `https://api.cyberneticphysics.com` | Control-plane API base URL (this is the **API** origin, not the website). |
| `--client-name` | `cybernetics-cli` | Label attached to the minted key. |
| `--no-browser` | off | Print the code/URL but do not auto-open a browser. |

Failure modes you may see, verbatim from source:

```text
The login code expired. Run 'cybernetics auth login' again.   # expired_token
Login timed out before approval.                              # deadline passed
Could not start device login (HTTP 503).                      # device/code failed
```

## Recipe C: check, export, and revoke

### `cybernetics auth status` — who am I?

```bash
cybernetics auth status
```

```text
Logged in
  User:      alice@example.com
  Workspace: ws_1a2b3c
  Key:       cp_live_aaaaaaaa…
  Source:    stored login
```

`Source` is `stored login` when the resolved key matches the file written by
`auth login`, otherwise `environment` (i.e. it came from one of the env vars).
The key is printed truncated to its first 16 characters. If no key resolves,
`status` prints `Not logged in. Run 'cybernetics auth login'.` and exits `1`.

### `cybernetics auth token` — print the resolved key

`token` prints the resolved key (and nothing else) to stdout, so you can wire it
into the environment of another process:

```bash
export CYBERNETICS_API_KEY="$(cybernetics auth token)"
```

If no key resolves it errors instead of printing an empty string:

```text
No API key found.
Run 'cybernetics auth login' or set CYBERNETICS_API_KEY.
```

### `cybernetics auth logout` — revoke and remove

```bash
cybernetics auth logout
```

```text
✓ Logged out.
```

`logout` makes a best-effort `DELETE /v1/api-keys/{key_id}` to revoke the key
server-side (using the stored `key_id` and `base_url`), then deletes the local
file. If revocation fails over the network it still removes the local file, so
you end up logged out locally either way. If there was no stored login it prints
`Not logged in.` and does nothing else.

Full CLI surface is in the [CLI reference](../reference/cli.md).

## Where the credential is stored, and its permissions

`auth login` writes a single JSON file. The directory is created `0700` and the
file `0600`, written atomically (temp file + `os.replace`) so a crash never
leaves a half-written secret:

```text
~/.config/cybernetics/auth.json        # file mode 0600
~/.config/cybernetics/                 # dir  mode 0700
```

The path honors `XDG_CONFIG_HOME`: if that variable is set, the file lives at
`$XDG_CONFIG_HOME/cybernetics/auth.json` instead of `~/.config`.

The file is a JSON object with these fields (`StoredCredentials`):

```json
{
  "api_key": "cp_live_...",
  "base_url": "https://api.cyberneticphysics.com",
  "user": "alice@example.com",
  "workspace": "ws_1a2b3c",
  "key_id": "key_...",
  "saved_at": 1751212800.0
}
```

Only `api_key` is required when loading; the rest are optional. If the file's
permissions are looser than `0600` (group- or other-accessible), the loader does
**not** fail — it emits a warning so you can fix it:

```text
Credential file ~/.config/cybernetics/auth.json is group/other-accessible
(mode 0o644); run 'chmod 600 ~/.config/cybernetics/auth.json'.
```

## Environment variables and base-url override

The complete set of auth-related environment variables read by the SDK:

| Variable | Read by | Purpose |
|----------|---------|---------|
| `CYBERNETICS_API_KEY` | `resolve_api_key` | Canonical API key (precedence #2). |
| `CP_API_KEY` | `resolve_api_key` | Cybernetic Physics key alias (precedence #3). |
| `WORLDLINES_API_KEY` | `resolve_api_key` | **Deprecated** key alias (precedence #4, warns). |
| `CYBERNETICS_BASE_URL` | `resolve_base_url` | Override the control-plane API base URL. |
| `CP_API_BASE` | `resolve_base_url` | Base-URL alias, checked after `CYBERNETICS_BASE_URL`. |
| `XDG_CONFIG_HOME` | `config_dir` | Relocates the credential directory. |

Base-URL resolution mirrors the key order: explicit argument →
`CYBERNETICS_BASE_URL` → `CP_API_BASE` → the `base_url` saved in the stored
login. The CLI falls back to `https://api.cyberneticphysics.com` if none of
those is set.

```bash
# Point the SDK and CLI at a non-default control plane.
export CYBERNETICS_BASE_URL="https://luc-api.cyberneticphysics.com"
```

The credential store may also carry Cloudflare Access headers via
`CLOUDFLARE_ACCESS_CLIENT_ID` / `CLOUDFLARE_ACCESS_CLIENT_SECRET`, which the
default-header builder forwards when present. Those are for fronting the API
with Cloudflare Access and are independent of your `cp_live_` key — set them only
if your deployment requires them.

## Troubleshooting

**`resolve_api_key()` returns `None` / `auth status` says "Not logged in".**
No source is set. Either `export CYBERNETICS_API_KEY=cp_live_...` or run
`cybernetics auth login`. Remember the env vars beat the stored login (precedence
#2–#4 before #5), so a stale `CP_API_KEY` in your shell can mask a fresh
`auth login`.

**You logged in but the SDK still uses a different key.** An environment
variable is shadowing the stored file. Run `cybernetics auth status` and check
the `Source:` line — if it says `environment`, unset the env var or update it.

**`DeprecationWarning: WORLDLINES_API_KEY is deprecated`.** You are on the
legacy variable (precedence #4). Rename your export to `CYBERNETICS_API_KEY`.

**Authentication fails even though a key is set.** Confirm you are sending
`Authorization: Bearer`, not `X-API-Key`. If you are using the blessed clients
this is automatic; if you hand-rolled a request, that header is almost always
the cause. See the warning at the top of this page.

**`cybernetics auth login` opens the wrong site / times out.** `--base-url` and
`CYBERNETICS_BASE_URL` target the **API** origin
(`https://api.cyberneticphysics.com`), not the marketing site. Set the API base
URL and retry.

More error decoding lives in [Errors & Troubleshooting](../reference/errors.md).

## Next

- [Quickstart](../quickstart.md) — first real training step.
- [Client API](../reference/client-api.md) — the blessed `ServiceClient` /
  `TrainingClient` / `SamplingClient` surface.
- [CLI reference](../reference/cli.md) — every `cybernetics` subcommand.
- [Mental model](../concepts/mental-model.md) — how sessions, leases, and
  futures fit together.
