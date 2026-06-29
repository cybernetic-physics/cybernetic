# Introduction

The Cybernetics SDK (`cybernetic-physics`) is the Python client for the hosted
Cybernetics control plane — a Tinker-style API for **rollout, sampling, and LoRA
training of robot foundation models** on platform-managed GPU compute leases.

You write ordinary Python on your laptop or in CI. The SDK opens a session
against the control plane, that session leases GPUs and runs a backend worker,
and you drive training and inference through a small set of high-level clients.
There is no infrastructure for you to provision: you call `forward_backward`,
`optim_step`, and `sample`, and the hosted side does the rest.

This page is the front door. It explains what the SDK is, the one supported way
to use it, and where to go next. If you just want to run something, jump to the
[Quickstart](./quickstart.md).

## What you actually call

The SDK ships two layers. Only one of them is the public API.

The **blessed path** is the hand-written high-level clients re-exported at the
package root:

| Symbol | Role |
|---|---|
| `ServiceClient` | Entry point. Opens a session; creates the other clients. |
| `TrainingClient` | `forward`, `forward_backward`, `optim_step`, save checkpoints. |
| `SamplingClient` | `sample`, `compute_logprobs` against a base model or saved weights. |
| `APIFuture` | The return type of async-submitted work; call `.result()` or `await` it. |

`ServiceClient`, `TrainingClient`, `SamplingClient`, and `APIFuture` are exported
from `cybernetics` directly (see `cybernetics.__all__`). There is also a
`RestClient` for REST queries (training runs, checkpoints, weights
info) and checkpoint deletion; you do **not** import it yourself — you get one from
`service_client.create_rest_client()`.

```python
import cybernetics
from cybernetics import types

service_client = cybernetics.ServiceClient(project_id="robotics-lab")
training_client = service_client.create_lora_training_client(base_model="dreamzero-droid")
```

> **The internal transport is off-limits.** `import cybernetics` also exposes
> `AsyncCybernetics` (aliased as `Cybernetics`) and a `cybernetics.resources.*`
> tree. That layer is the Stainless-generated HTTP transport the high-level
> clients are built on. It is not the supported surface, its shape can change
> without notice, and nothing in these docs targets it. If you find yourself
> reaching for `AsyncCybernetics` or `cybernetics.resources`, stop — the
> operation you want is on one of the four clients above.

For the full method-by-method listing, see the [client API reference](./reference/client-api.md).

## Honest scope: training-first today

The control plane's training surface is the mature one. The
Tinker-compatible loop — `forward_backward(...).result()` then
`optim_step(...).result()` — is what the hosted backend is built around, and
it is what the shipped examples (`examples/dreamzero_sft_smoke.py`,
`examples/dreamzero_rl_smoke.py`) exercise end to end.

Sampling works and is part of the public API, but the worked examples are
**training-first**: you generally reach `SamplingClient` *after* training, via
`training_client.save_weights_and_get_sampling_client(...)` or
`service_client.create_sampling_client(base_model=...)`. Treat sampling as the
inference half of a training workflow rather than a standalone serving product,
and read the [Sampling & Inference guide](./guides/sampling.md) for what is and
isn't covered. The [SFT vs RL](./concepts/sft-vs-rl.md) page explains which loss
families the backend advertises.

## Install

The distribution name and the import name differ. Install
**`cybernetic-physics`**; import **`cybernetics`**.

```bash
pip install cybernetic-physics
export CYBERNETICS_API_KEY="cp_live_..."   # or run: cybernetics auth login
cybernetics doctor                          # read-only API/auth/SFT/RL readiness check
```

`requires-python` is **`>= 3.11`**. The wheel ships only the `cybernetics`
package; the backend never enters the distribution.

To pin the current repo version instead of a PyPI release:

```bash
uv add "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git"
```

### Optional extras

`numpy` is a core dependency. The heavier libraries are not — they are lazily
imported and gated behind extras, so install only what your workflow needs:

| Extra | Pulls in | Needed for |
|---|---|---|
| `tokenizers` | `transformers` | `get_tokenizer()` on the training/sampling clients |
| `torch` | `torch` | local tensor interop |
| `aiohttp` | `aiohttp`, `httpx_aiohttp` | the optional aiohttp transport |
| `behavior-ci` | `pyyaml` | reading [Behavior CI](./guides/behavior-ci.md) YAML config/eval files |
| `all` | all of the above | everything |

```bash
pip install 'cybernetic-physics[tokenizers]'
pip install 'cybernetic-physics[all]'
```

If you call `get_tokenizer()` without the `tokenizers` extra installed, the
import of `transformers` fails at call time rather than at install time — that's
expected. Install the extra and retry.

## Names you will see

The product has accreted a few names. None of them are typos; here is how they
map.

- **Cybernetics** — the product and the SDK. The import package is `cybernetics`;
  the CLI is `cybernetics`.
- **Cybernetic Physics / CP** — the company behind it. It survives in the
  `CP_API_KEY` environment variable and in API keys prefixed `cp_live_`.
- **Worldlines** — the legacy name for the session/storage layer. It survives in
  `worldlines://` checkpoint and weight paths, in the `WORLDLINES_API_KEY`
  environment variable (**deprecated** — use `CYBERNETICS_API_KEY`), and in
  internal session IDs.

Authentication uses your `cp_live_` key as an `Authorization: Bearer ...`
header. The key is resolved in this order:

1. an explicit `api_key=` argument,
2. `CYBERNETICS_API_KEY`,
3. `CP_API_KEY`,
4. `WORLDLINES_API_KEY` (deprecated),
5. the stored login from `cybernetics auth login`
   (`~/.config/cybernetics/auth.json`).

Full details are in the [Authentication guide](./guides/authentication.md).

## Check readiness before spending GPU

`cybernetics doctor` is a read-only probe. It checks API reachability,
authentication, and advertised training/sampling support **without** creating a
session, model, compute lease, or future — so it costs nothing and is safe in
CI.

```bash
cybernetics doctor
cybernetics doctor --require-rl     # exit nonzero unless the RL loss family is advertised
cybernetics --format json doctor
```

See the [CLI reference](./reference/cli.md) for every command and flag.

## Where to go next

- **[Quickstart](./quickstart.md)** — the shortest path from `pip install` to a
  real training step and a saved checkpoint.
- **[Mental Model](./concepts/mental-model.md)** — how sessions, compute leases,
  futures, and the four clients fit together. Read this before you build
  anything non-trivial.
- **[SFT vs RL](./concepts/sft-vs-rl.md)** — which training regimes the backend
  supports and how to tell.
- **[Client API reference](./reference/client-api.md)** — exhaustive method and
  signature lookup.
- **[Errors & Troubleshooting](./reference/errors.md)** — the exception
  hierarchy (`CyberneticsError` and friends) and what to do when work fails.
