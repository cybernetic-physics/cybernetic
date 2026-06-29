# Errors & Troubleshooting

This page is the lookup table for every exception the Cybernetics SDK can raise,
plus recipes for the failure modes you are most likely to hit first: a bad API
key, a capability the backend has not advertised, a cold-start timeout, and a
missing optional dependency.

Everything documented here is grounded in the exception classes exported at the
package root (`cybernetics/_exceptions.py`, re-exported through
`cybernetics/__init__.py`). If a behavior is not described here, do not assume it
exists.

For where keys come from, see [Authentication](../guides/authentication.md). For
the supported client surface, see the [Client API](./client-api.md). For the
`cybernetics doctor` command, see the [CLI reference](./cli.md).

## The exception hierarchy

All SDK errors derive from a single base, `CyberneticsError`. Catch that if you
want one `except` clause to cover everything the SDK can throw:

```python
import cybernetics

try:
    capabilities = service_client.get_server_capabilities()
except cybernetics.CyberneticsError as exc:
    # Covers every SDK-specific error: transport, status, async-request, sidecar.
    log.error("Cybernetics call failed: %s", exc)
    raise
```

The tree below shows the full hierarchy. Every name is importable from the
package root (`from cybernetics import NotFoundError`); the package rewrites each
class's `__module__` to `"cybernetics"`, so a traceback reads
`cybernetics.NotFoundError`, not `cybernetics._exceptions.NotFoundError`.

```text
Exception
└── CyberneticsError                     # base for everything below
    ├── APIError                         # base for HTTP-transport errors; has .request, .body
    │   ├── APIResponseValidationError   # response body did not match the expected schema
    │   ├── APIConnectionError           # could not establish/complete the connection
    │   │   └── APITimeoutError          # the HTTP request itself timed out
    │   └── APIStatusError               # server returned a 4xx/5xx; has .response, .status_code
    │       ├── BadRequestError          # 400
    │       ├── AuthenticationError      # 401
    │       ├── PermissionDeniedError    # 403
    │       ├── NotFoundError            # 404
    │       ├── ConflictError            # 409
    │       ├── UnprocessableEntityError # 422
    │       ├── RateLimitError           # 429
    │       └── InternalServerError      # 500+
    ├── RequestFailedError               # an async request finished in a failed state
    └── SidecarError                     # subprocess sidecar failures
        ├── SidecarStartupError          # sidecar failed to start or timed out
        ├── SidecarDiedError             # sidecar exited while requests were pending
        └── SidecarIPCError              # IPC with the sidecar failed
```

Three families, three different things going wrong:

- **`APIError` and its subclasses** are HTTP-transport errors. They are raised
  synchronously by the underlying transport when an HTTP request fails or the
  server returns a 4xx/5xx. They are what you catch around calls that talk to
  the control plane directly (for example `service_client.get_server_capabilities()`
  or `cybernetics doctor`'s capability probe).
- **`RequestFailedError`** is raised when an *asynchronous* request — one you
  submitted and are awaiting through an [`APIFuture`](./client-api.md) — comes
  back in a failed state. The HTTP call succeeded; the work it represented did
  not.
- **`SidecarError` and its subclasses** come from the in-process subprocess
  ("sidecar") used by some local operations. They have nothing to do with HTTP.

> The Stainless-generated `AsyncCybernetics` / `cybernetics.resources.*` transport
> is what actually raises the `APIError` subclasses internally. Do not program
> against that layer directly — use the hand-written clients (`ServiceClient`,
> `TrainingClient`, `SamplingClient`, `APIFuture`) re-exported at
> the package root. The exception types, however, are part of the public surface
> and are safe to catch.

## HTTP status code → exception

`APIStatusError` is raised for any 4xx/5xx. The SDK maps the common codes to
specific subclasses so you can branch on the error type instead of inspecting
`status_code` by hand. Each subclass carries the `.response` (an
`httpx.Response`), `.status_code`, `.message`, and `.body` (the decoded JSON
body, the raw response, or `None`).

| Status | Exception | Likely cause | What to do |
| ------ | --------- | ------------ | ---------- |
| 400 | `BadRequestError` | Malformed request: bad field types, missing required fields, an unparseable payload. | Re-check the request against the [Data Contracts](./data-contracts.md). Read `exc.body` for the server's detail. |
| 401 | `AuthenticationError` | API key missing, empty, expired, or not a valid `cp_live_` key. | See [401 — authentication](#401--authentication) below and the [Authentication guide](../guides/authentication.md). |
| 403 | `PermissionDeniedError` | Key is valid but the workspace lacks access to the resource, model, or feature. | Confirm the key's workspace. A capability you can authenticate to may still be gated — run `cybernetics doctor`. |
| 404 | `NotFoundError` | The resource ID does not exist: a stale `worldlines://` checkpoint path, a deleted run, a typo'd model ID. | Verify the ID. For checkpoint paths see [Checkpoints](../guides/checkpoints.md). |
| 409 | `ConflictError` | Request conflicts with current state — e.g. a lock/lease contention. The transport auto-retries 409 (see [retries](#retries-and-timeouts)). | Usually transient. If it persists, another client may hold the resource. |
| 422 | `UnprocessableEntityError` | Request was well-formed but semantically invalid: an unsupported loss family, an out-of-range hyperparameter, an incompatible tensor shape. | Fix the values. For loss families, check `loss_families` from `get_server_capabilities` — see [capability not advertised](#capability-not-advertised). |
| 429 | `RateLimitError` | Too many requests; rate limit exceeded. The transport auto-retries 429. | Back off and reduce concurrency. If it persists after retries, lower request rate. |
| 500+ | `InternalServerError` | Server-side error. The transport auto-retries 5xx. | Transient errors clear on retry. Persistent 5xx is a backend issue you cannot fix client-side; capture `exc.request` and `exc.body`. |

Other `APIError` subclasses that are *not* status-code-keyed:

| Exception | When it is raised | Notes |
| --------- | ----------------- | ----- |
| `APIConnectionError` | The connection could not be established or completed (DNS failure, refused connection, dropped socket). | Default message: `"Connection error."`. `exc.body` is `None`. Check `base_url` and network reachability. |
| `APITimeoutError` | The HTTP request timed out. Subclass of `APIConnectionError`. | Default message: `"Request timed out."`. This is the *transport* timeout, distinct from a cold-start wait — see [cold-start timeouts](#cold-start-and-lease-timeouts). |
| `APIResponseValidationError` | The server replied 2xx but the body did not match the expected schema. | Carries `.response` and `.status_code`. Default message: `"Data returned by API invalid for expected schema."`. Usually signals a client/server version skew. |

### Branching on the error type

```python
import cybernetics

try:
    result = service_client.get_server_capabilities()
except cybernetics.AuthenticationError:
    raise SystemExit("Set CYBERNETICS_API_KEY or run `cybernetics auth login`.")
except cybernetics.PermissionDeniedError as exc:
    raise SystemExit(f"Key valid but not authorized: {exc.body}")
except cybernetics.RateLimitError:
    # Transport already retried 429 with backoff; if we still land here, slow down.
    raise
except cybernetics.APIStatusError as exc:
    # Catch-all for any other 4xx/5xx.
    raise SystemExit(f"API error {exc.status_code}: {exc.message}")
except cybernetics.APIConnectionError as exc:
    # Covers APITimeoutError too (it is a subclass).
    raise SystemExit(f"Could not reach the control plane: {exc}")
```

Order matters: catch the specific subclasses before `APIStatusError`, and
`APIStatusError` before the broader `APIError` / `CyberneticsError`.

## Async-request failures: `RequestFailedError`

When you submit work and await it through an `APIFuture` (returned by, e.g.,
`training_client.forward_backward(...)`), a successful HTTP round-trip does not
guarantee the work succeeded. If the request resolves into a failed state, the
future raises `RequestFailedError` from `.result()` / `.result_async()` /
`await`.

`RequestFailedError` carries three fields:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `message` | `str` | Human-readable failure message, prefixed `Request failed: ...`. |
| `request_id` | `str` | The server-side request ID — quote this when reporting a problem. |
| `category` | `RequestErrorCategory` | A `StrEnum` with members `Unknown`, `Server`, `User`. |

`category` tells you who is at fault:

- `RequestErrorCategory.User` — the request was malformed or invalid; retrying
  the same input will fail again. Fix the input.
- `RequestErrorCategory.Server` — a server-side failure; a retry may succeed.
- `RequestErrorCategory.Unknown` — the server did not classify it (this is the
  default when no category is reported).

```python
import cybernetics
from cybernetics.types import RequestErrorCategory

future = training_client.forward_backward(data, "cross_entropy")
try:
    output = future.result()
except cybernetics.RequestFailedError as exc:
    if exc.category is RequestErrorCategory.User:
        # Deterministic — do not retry the same payload.
        raise SystemExit(f"Invalid request {exc.request_id}: {exc.message}")
    # Server/Unknown — a retry might help.
    log.warning("Request %s failed (%s); retrying", exc.request_id, exc.category)
```

`APIFuture.result(timeout=...)` and `result_async(timeout=...)` also raise the
standard library `TimeoutError` if the timeout elapses before the result is
ready — that is a `TimeoutError`, not a `RequestFailedError`. See
[cold-start timeouts](#cold-start-and-lease-timeouts).

## Sidecar errors

Some local operations run inside a subprocess "sidecar." Its failures surface as
`SidecarError` subclasses (all under `CyberneticsError`, none related to HTTP):

| Exception | When it is raised | Example message |
| --------- | ----------------- | --------------- |
| `SidecarStartupError` | The subprocess failed to start or did not complete its startup handshake in time. | `Sidecar subprocess failed to start within <N>s`; `Sidecar subprocess died before startup (exit code: ...)` |
| `SidecarDiedError` | The subprocess exited unexpectedly while requests were still pending. | `Sidecar subprocess is not running (exit code: ...)` |
| `SidecarIPCError` | Inter-process communication with the subprocess failed. | — |

These are environment/process problems rather than request problems. If you hit
them repeatedly, capture the full traceback and the exit code from the message.

## Troubleshooting

### 401 — authentication

`AuthenticationError` (HTTP 401) means the control plane rejected your
credentials. The SDK resolves an API key in this fixed order (from
`cybernetics/lib/credentials.py`):

1. an explicit `api_key=` argument,
2. the `CYBERNETICS_API_KEY` environment variable,
3. the `CP_API_KEY` environment variable (the short name used in Cybernetic
   Physics examples and dev-infra runbooks),
4. the **deprecated** `WORLDLINES_API_KEY` fallback — still read for one release,
   but it emits a `DeprecationWarning` telling you to switch to
   `CYBERNETICS_API_KEY`,
5. the stored login file written by `cybernetics auth login` (at
   `$XDG_CONFIG_HOME/cybernetics/auth.json`, falling back to
   `~/.config/cybernetics/auth.json`).

> **Brand-name reconciliation.** The product/SDK is **Cybernetics**; the import
> package is `cybernetics`; you install it with `pip install cybernetic-physics`.
> **Cybernetic Physics** ("CP") shows up in `CP_API_KEY`, `CP_API_BASE`, and the
> `cp_live_` API-key prefix. **Worldlines** is the legacy name that survives in
> `worldlines://` checkpoint paths and the deprecated `WORLDLINES_API_KEY`. These
> all refer to the same system.

Checklist for a 401:

- Confirm a key is actually set. `cybernetics doctor` prints the resolved auth
  **source** (`CYBERNETICS_API_KEY`, `CP_API_KEY`, `WORLDLINES_API_KEY`, or
  `stored login`); if it reports `No API key found`, nothing in the chain above
  resolved.
- Confirm the value is a real `cp_live_` key and not truncated by your shell.
- If a stored login is being used, re-run `cybernetics auth login`. If the
  credential file warns that it is "group/other-accessible," run
  `chmod 600 ~/.config/cybernetics/auth.json` as the warning instructs.
- If you set `WORLDLINES_API_KEY` and see a `DeprecationWarning`, migrate to
  `CYBERNETICS_API_KEY` — the legacy variable is read for only one more release.

A valid key that is rejected for a *specific resource* returns 403
(`PermissionDeniedError`), not 401. A 403 means "authenticated, but not
authorized."

### Capability not advertised

A backend may not advertise every capability. Asking for one it has not
advertised typically surfaces as a 422 (`UnprocessableEntityError`) — for
example requesting a loss family the server does not support. Before you submit
work, ask the server what it supports rather than discovering the gap mid-run.

**From the CLI**, `cybernetics doctor` probes `/api/v1/get_server_capabilities`
and summarizes readiness without creating a session or lease:

```bash
cybernetics doctor
```

It reports, per check, `ready` / `unavailable` / `unknown` for API, Auth,
Health, Training, Sampling, DreamZero SFT, and DreamZero RL. When a loss family
you asked about is not in the backend's advertised set, the detail column says
so explicitly, e.g. `flow_rwr not advertised by this backend`. To require a
specific RL loss and exit nonzero if it is not ready:

```bash
cybernetics doctor --rl-loss flow_rwr --require-rl
```

`--require-rl` exits `1` unless the backend advertises DreamZero RL readiness —
useful as a CI preflight gate. (`--rl-loss` defaults to `flow_rwr`.)

**From Python**, query the same capabilities through `ServiceClient`:

```python
capabilities = service_client.get_server_capabilities()
# capabilities.supports_training / .supports_sampling / .loss_families, etc.
if "cross_entropy" not in (capabilities.loss_families or []):
    raise SystemExit("This backend does not advertise cross_entropy; check `cybernetics doctor`.")
```

The doctor reads these fields from the capabilities response: `supports_training`,
`supports_sampling`, `loss_families`, `dreamzero_rl_available`, and
`dreamzero_rl_unavailable_reason`. DreamZero SFT is considered ready when
training is not disabled and `cross_entropy` is in `loss_families`. If the
capability list comes back empty, doctor reports the capability as `unknown`
rather than guessing.

### Cold-start and lease timeouts

Two different timeouts can stop a call, and they raise two different exceptions:

- **`APITimeoutError`** (an `APIError` / `APIConnectionError` subclass) — the
  underlying *HTTP request* timed out. Message: `"Request timed out."`.
- **`TimeoutError`** (the Python builtin) — you passed a `timeout=` to an
  `APIFuture` (`future.result(timeout=...)`) or to a client constructor that
  waits for a cold start, and that wait elapsed first.

Several `ServiceClient` constructors accept a `timeout` for exactly this: e.g.
`create_lora_training_client(...)` documents `timeout` as "Optional seconds to
wait for model creation/cold start," and `create_sampling_client(...)` documents
it as "Optional seconds to wait for sampler creation/cold start." When a backend
is scaling a model up from cold, the first call can take meaningfully longer than
steady state.

What to do:

- For genuine cold starts, **raise the `timeout`** on the constructor (or pass
  `timeout=None` to wait indefinitely). A short timeout against a cold backend
  produces a `TimeoutError` that a longer wait would not.
- Run `cybernetics doctor` first. It performs a lightweight health probe
  (`/health`, then `/api/v1/healthz`) and a capability check **without** creating
  a session or lease, so you can tell "backend unreachable" apart from "backend
  reachable but still warming up."
- A `ConflictError` (409) can also appear under contention for a leased
  resource; the transport already auto-retries 409, so a persistent 409 means
  something else is holding the resource — not just a slow start.

### Missing `[tokenizers]` extra

Tokenizer helpers depend on the optional `transformers` package, which is **not**
installed by the base `pip install cybernetic-physics`. Calling
`get_tokenizer()` without it raises a plain `ImportError` (not a
`CyberneticsError`) with the exact remedy:

```text
ImportError: get_tokenizer() requires the optional 'transformers' dependency.
Install it with: pip install 'cybernetic-physics[tokenizers]'
```

Fix it by installing the extra:

```bash
pip install 'cybernetic-physics[tokenizers]'
```

The `tokenizers` extra pulls in `transformers`. If you want every optional
dependency at once, the `all` extra includes it
(`cybernetic-physics[tokenizers,aiohttp,torch,behavior-ci]`):

```bash
pip install 'cybernetic-physics[all]'
```

Note the brand split again: the **install** name is `cybernetic-physics`, but
the **import** name is `cybernetics` — `import cybernetics`, never
`import cybernetic_physics`.

Separately, some training paths require PyTorch. Custom `forward_backward` work
raises `ImportError("PyTorch is not installed. Cannot run custom
forward_backward.")` when `torch` is absent; install the `torch` extra (or the
`all` extra) to enable it.

## Quick reference

| Symptom | Exception | Section |
| ------- | --------- | ------- |
| Key missing or invalid | `AuthenticationError` (401) | [401 — authentication](#401--authentication) |
| Authenticated but not authorized | `PermissionDeniedError` (403) | [status table](#http-status-code--exception) |
| Unsupported loss / out-of-range value | `UnprocessableEntityError` (422) | [capability not advertised](#capability-not-advertised) |
| Rate limited | `RateLimitError` (429) | [status table](#http-status-code--exception) |
| Backend error | `InternalServerError` (500+) | [status table](#http-status-code--exception) |
| Cannot reach control plane | `APIConnectionError` / `APITimeoutError` | [cold-start timeouts](#cold-start-and-lease-timeouts) |
| Async submitted work failed | `RequestFailedError` | [async-request failures](#async-request-failures-requestfailederror) |
| `future.result(timeout=...)` elapsed | `TimeoutError` (builtin) | [cold-start timeouts](#cold-start-and-lease-timeouts) |
| `get_tokenizer()` import fails | `ImportError` | [missing tokenizers extra](#missing-tokenizers-extra) |
| Local subprocess failure | `SidecarError` subclasses | [sidecar errors](#sidecar-errors) |

## Retries and timeouts

The internal transport already retries the transient codes for you, with
exponential backoff and jitter: **408**, **409**, **429**, and **5xx**. It also
honors a server-sent `x-should-retry` header (`true` forces a retry, `false`
suppresses one). So by the time one of those surfaces as an exception to your
code, the SDK has already retried and given up — adding your own naive retry loop
on top usually will not help and can make rate limiting worse.

## See also

- [Authentication](../guides/authentication.md) — key resolution, `cp_live_`
  keys, `cybernetics auth login`.
- [Client API](./client-api.md) — `ServiceClient`, `TrainingClient`,
  `SamplingClient`, `RestClient`, `APIFuture`.
- [CLI reference](./cli.md) — `cybernetics doctor` and friends.
- [Data Contracts](./data-contracts.md) — request/response shapes behind 400/422.
- [Checkpoints](../guides/checkpoints.md) — `worldlines://` paths behind 404.
