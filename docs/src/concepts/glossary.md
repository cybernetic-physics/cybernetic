# Glossary

This page reconciles every brand name and piece of jargon you will hit while
reading the source, the `examples/` directory, and the rest of these docs.
Definitions are one line, honest, and grounded in the code you can
read yourself. Where a term names server-side behavior the SDK only talks to
over the wire, it is marked **see backend docs** rather than guessed at.

For the conceptual picture behind these terms, read the
[mental model](./mental-model.md); for the exact public types and signatures,
see the [client API reference](../reference/client-api.md).

## Brand and naming terms

The product has accumulated three names. They are not synonyms for different
things — they are the same platform at different layers, and you will see all
three in code. Here is the map.

| Term | What it is | Where you see it |
|---|---|---|
| **Cybernetics** | The product and the SDK. The blessed, hand-written public API. | The import package `cybernetics`; `CYBERNETICS_API_KEY`; `cybernetics` CLI; `ServiceClient`. |
| **Cybernetic Physics / CP** | The company / platform brand, abbreviated **CP**. | The PyPI distribution `cybernetic-physics`; `CP_API_KEY`; `cp_live_` key prefix; `CP_API_BASE`. |
| **Worldlines** | The legacy name. Survives in the resource-path scheme and one deprecated env var. | `worldlines://` checkpoint paths; the deprecated `WORLDLINES_API_KEY`; the term "Worldlines session" in client internals. |

If you remember one thing: **Cybernetics is what you import, Cybernetic Physics
(CP) is what you install and authenticate as, and Worldlines is the old name
baked into paths you cannot rename.**

### cybernetic-physics (distribution name)

The package you install from PyPI or Git. The import name is different — see
below.

```bash
pip install cybernetic-physics
```

### cybernetics (import package)

What you actually `import`. The distribution is `cybernetic-physics`; the
top-level module is `cybernetics`.

```python
import cybernetics
from cybernetics import ServiceClient
```

This split is intentional and is the most common first-run confusion — installing
`cybernetics` from PyPI gets you a different, unrelated project.

### CYBERNETICS_API_KEY

The current, preferred environment variable for your API key. First env var
checked during key resolution (after an explicit `api_key=` argument). Defined in
`cybernetics.lib.credentials` as `API_KEY_ENV`.

### CP_API_KEY

A shorter alias env var "used in Cybernetic Physics examples and dev-infra
runbooks." Checked **after** `CYBERNETICS_API_KEY` and before the deprecated
Worldlines var. Defined as `CP_API_KEY_ENV`.

### WORLDLINES_API_KEY (deprecated)

Legacy env var, read for **one release** so existing exports keep working. When
it is the source of your key, the SDK emits a `DeprecationWarning` telling you to
set `CYBERNETICS_API_KEY` instead. It is last in the resolution order before the
stored login file.

```text
WORLDLINES_API_KEY is deprecated; set CYBERNETICS_API_KEY instead.
```

The full resolution order (from `resolve_api_key`):

1. explicit `api_key=` argument,
2. `CYBERNETICS_API_KEY`,
3. `CP_API_KEY`,
4. `WORLDLINES_API_KEY` (deprecated, warns),
5. the stored login file at `~/.config/cybernetics/auth.json`.

See [Authentication](../guides/authentication.md) for the full walkthrough.

### cp_live_ (key prefix)

The literal prefix on a live API key, e.g. `cp_live_...`. The "CP" stands for
Cybernetic Physics. The SDK sends the key as `Authorization: Bearer ...`; the
examples additionally forward it as an `X-API-Key` header to match the validated
drivers. The SDK does not validate the prefix itself — it forwards whatever
string you give it.

### worldlines:// (path scheme)

The URI scheme for checkpoints and weights on the control plane. The legacy name
survives here. Format:

```text
worldlines://<training_run_id>/<weights|sampler_weights>/<checkpoint_id>
```

Parsed by `ParsedCheckpointCyberneticsPath.from_worldlines_path`: the second
segment must be `weights` (a `training` checkpoint) or `sampler_weights` (a
`sampler` checkpoint), or parsing raises `ValueError`. Example:
`worldlines://run-123/weights/final`. See [Checkpoints](../guides/checkpoints.md).

### CYBERNETICS_BASE_URL / CP_API_BASE

The control-plane base URL env vars, resolved by `resolve_base_url` in order:
explicit arg, then `CYBERNETICS_BASE_URL` (`BASE_URL_ENV`), then `CP_API_BASE`
(`CP_BASE_URL_ENV`), then the stored login file.

### Repo slug

The Git source is `github.com/cybernetic-physics/cybernetic` — note the org is
`cybernetic-physics` and the repo is `cybernetic` (singular, no `s`):

```bash
uv add "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git"
```

## The blessed public API

These five hand-written, high-level types in
`cybernetics.lib.public_interfaces` (re-exported at the package root) are the
**supported** API. Everything else is internal transport — see
[AsyncCybernetics](#asynccybernetics-internal-transport) below.

### ServiceClient

The main entry point. Construct it, then ask it for the other clients. Creating
one establishes a Worldlines session against the control plane.

```python
client = cybernetics.ServiceClient(project_id="robotics-lab")
training_client = client.create_lora_training_client(base_model="dreamzero-droid")
sampling_client = client.create_sampling_client(base_model="dreamzero-droid")
rest_client = client.create_rest_client()
```

It exposes `session_id` (the "Worldlines session ID backing this client") and
factory methods for the three sub-clients.

### TrainingClient

Drives the train loop. The blessed surface is Tinker-style:
`forward_backward([datum], loss_fn)` → `optim_step(adam)` → `save_state(name)`,
each returning a [future](#apifuture). See [Training](../guides/training.md).

```python
training_client.forward_backward([datum], "cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

### SamplingClient

Generates completions from a model. Its `sample(prompt, num_samples,
sampling_params, ...)` method returns a `Future[SampleResponse]`.

```python
params = types.SamplingParams(max_tokens=20, temperature=0.7)
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()
```

Honest limit: the shipped examples are **training-first** — all four validated
backends are exercised through `forward_backward`/`optim_step`/`save_state`, and
the repo ships no end-to-end VLA-sampling example. Treat the sampling surface as
real but lightly exercised here. See [Sampling & Inference](../guides/sampling.md).

### RestClient

Read-oriented REST operations: listing training runs and checkpoints, resolving
weights info, and getting checkpoint archive URLs (e.g. `get_training_run`,
`list_training_runs`, `list_checkpoints`, `get_weights_info_by_worldlines_path`,
`get_checkpoint_archive_url`). Created with `client.create_rest_client()`.

### APIFuture

The unified future returned by training/sampling calls. It can be awaited in
async code or blocked on synchronously:

```python
result = await api_future          # async: same as await api_future.result_async()
result = api_future.result()       # sync: blocks until complete
```

`result_async(timeout=...)` raises `TimeoutError` if the timeout is exceeded.

### AsyncCybernetics (internal transport)

The Stainless-generated, async-first client (`cybernetics.AsyncCybernetics`,
aliased as `cybernetics.Cybernetics`) and the `cybernetics.resources.*` layer
underneath it. This is the **internal transport** the blessed clients call. Do
not build against it directly — use `ServiceClient` / `TrainingClient` /
`SamplingClient` / `RestClient` instead. It is mentioned here only so you
recognize it and avoid it.

## Model and backend terms

These come from the four validated backends in the repo's `examples/`
directory. Each is selected by its `base_model`
string passed to `create_lora_training_client` / `create_sampling_client`.

### base_model

The string that selects the backend on the wire, e.g. `"dreamzero-droid"`,
`"groot-n1.5"`, `"pi0.5"`, `"cosmos3-nano"`. Only this string changes the
runtime; the client-side datum construction is otherwise shared where the serde
is shared.

### DreamZero

A flow-matching vision-language-action (VLA) model of Wan lineage. The
`dreamzero-droid` backend trains it with a LoRA, DROID recipe; the server-side
loss is a timestep-reweighted velocity MSE. DreamZero is the reference backend —
the SFT and RL smokes (`examples/dreamzero_sft_smoke.py`,
`examples/dreamzero_rl_smoke.py`) are the templates the others mirror.

### GR00T N1.5

NVIDIA's GR00T N1.5 robot foundation model (`base_model="groot-n1.5"`). Described
as "the Wan-lineage groot": it consumes the **exact same** datum as
`dreamzero-droid` (they share `cybernetics.lib.dreamzero.serde`), so only the
`base_model` string differs on the wire.

### pi0.5 / openpi

The openpi pi0.5 vision-language-action model (`base_model="pi0.5"`). Trained as
a **full finetune** (not LoRA), with a model-internal flow-matching velocity MSE.
"openpi" is the upstream stack; its runtime venv is needed only on the backend
that serves `pi0.5`. The client code lives in `cybernetics.lib.pi05`.

### Cosmos

NVIDIA Cosmos 3 omni world model (video). `base_model="cosmos3-nano"` is the
nano variant, trained as a rectified-flow LoRA. Its runtime needs the `diffusers`
+ `diffusers_cosmos3` venv on the backend. Client code: `cybernetics.lib.cosmos`.

### VLA (vision-language-action)

A model class that maps vision + language inputs to robot actions. DreamZero,
GR00T N1.5, and pi0.5 are VLAs; Cosmos is a (video) world model used to generate
data for them.

### flywheel

The synthetic-data loop demonstrated by `examples/flywheel_demo.py`: a world
model (`cosmos3-nano`) dreams robot-manipulation video, an inverse-dynamics model
(IDM) labels the dreamed frames with pseudo-actions, and a VLA (`groot-n1.5`)
trains on `(dreamed video, pseudo-actions)` — improving the policy with no new
teleop.

```text
Cosmos3-Nano dream-video
    -> placeholder-IDM pseudo-actions
    -> a dreamzero/groot collate (images = the dreamed frames)
    -> forward_backward on groot-n1.5
```

Honest limit: the IDM in the demo is a **placeholder** (a crude frame-diff motion
proxy), not the real GR00T-Dreams `idm_training.py`.

### DROID

A robot-manipulation dataset / recipe. `dreamzero-droid` uses the DROID recipe,
and the GR00T smoke builds a "synthetic DROID collate." Specifics of the dataset
itself are upstream — **see backend docs**.

### LeRobot

The HuggingFace robot-dataset format. `examples/finetune_on_real_data.py` is the
template for finetuning `groot-n1.5` on a real LeRobot dataset. Format details
are upstream — **see backend docs**.

### IDM (inverse-dynamics model)

A model that infers actions from observed frame transitions. Used in the
[flywheel](#flywheel) to pseudo-label dreamed video. The demo's IDM is a
placeholder; the real one is NVIDIA GR00T-Dreams `idm_training.py`.

## Workflow and platform jargon

### Tinker-compatibility

The SDK presents a "Tinker-like" / Tinker-style training contract:
`forward_backward([datum], loss_fn)` → `optim_step(adam)` → `save_state(name)`.
The same surface serves SFT and RL — the RL smoke routes a trajectory through the
DreamZero `flow_rwr` loss using the same `forward_backward` call. This is a
shape/ergonomics compatibility claim, not a statement that Cybernetics is Tinker.

### rollout

Listed in the SDK's own one-liner as one of the three things the platform does
("rollout, sampling, and LoRA training"). In the RL context it means generating
trajectories from a policy to train on. The repo ships no dedicated rollout
example beyond the RL smoke's synthetic trajectory; deeper rollout semantics are
server-side — **see backend docs**.

### sampling

Generating completions/actions from a model, via `SamplingClient.sample`. As
noted above, the shipped examples are training-first, so sampling is real in the
client but not exercised end-to-end for the VLA backends here.

### Datum

The single on-the-wire training example (`cybernetics.types.Datum`). The client
builds a per-model `collate` dict locally (tokenization, image stacks, normalized
actions), encodes it into **one** `Datum` per sample with that model's serde, and
ships it. The runtime decodes it back into the model's native input — no model
dataset is instantiated server-side. See [Data Contracts](../reference/data-contracts.md).

### serde

The per-model serialize/deserialize code that turns a collate dict into a `Datum`
and back. Pure `numpy`/`torch` (no GPU, no `groot`/`openpi`/`diffusers` needed to
build a datum). `dreamzero-droid` and `groot-n1.5` share
`cybernetics.lib.dreamzero.serde`.

### loss_fn

The loss-name string passed to `forward_backward`, e.g. `"cross_entropy"`.
Honest caveat: for `pi0.5` and `cosmos3-nano` this string is **ignored
server-side** — the loss is the model-internal flow / rectified-flow MSE — and
the examples pass the literal `"cross_entropy"` only because the wire
`LossFnType` Literal has no flow-matching name.

### flow_rwr

The DreamZero RL loss family, used by the RL smoke and the default required by
`cybernetics doctor --require-rl`. Exact algorithm is server-side — **see backend
docs**.

### session / Worldlines session

The unit of work a `ServiceClient` opens against the control plane; exposed as
`ServiceClient.session_id` and internally called the "Worldlines session ID."
Remote examples **cancel their session on exit** so a smoke does not leave paid
compute running.

### compute lease

A platform-managed GPU allocation backing a session ("platform-managed GPU
compute leases"). `--keep-lease` on the remote examples leaves the lease alive
for debugging instead of cancelling on exit; without it the session/lease is
released when the example finishes. Billing/scheduling specifics are server-side
— **see backend docs**.

### future

A handle to in-flight async work returned by training/sampling calls — concretely
an [APIFuture](#apifuture) (or a `concurrent.futures.Future`). Resolve it with
`.result()` (sync) or `await` (async). The readiness checks deliberately avoid
creating "a Worldlines session, model, compute lease, or future."

### doctor

The read-only CLI readiness command, `cybernetics doctor`: checks API
reachability, authentication, advertised training/sampling support, and DreamZero
SFT/RL capabilities **without** creating a session, model, lease, or future. Run
it before spending GPU time.

```bash
cybernetics doctor
cybernetics doctor --require-rl     # exits nonzero unless flow_rwr (RL) is advertised
cybernetics --format json doctor
```

See [CLI](../reference/cli.md) and [Errors & Troubleshooting](../reference/errors.md).

### save_state / checkpoint

`save_state(name)` writes a checkpoint artifact to the backend's `artifact_root`
as `/data/<run>/weights/<name>/` (`model.safetensors` + `config.json` +
`metadata.pt` + a `COMPLETE` sentinel), addressable as a
[`worldlines://`](#worldlines-path-scheme) path. A `Checkpoint` has a
`checkpoint_type` of `"training"` (`weights/`) or `"sampler"` (`sampler_weights/`).
See [Checkpoints](../guides/checkpoints.md).

## See also

- [Mental Model](./mental-model.md) — how the pieces fit together.
- [SFT vs RL](./sft-vs-rl.md) — the two training modes these terms feed into.
- [Authentication](../guides/authentication.md) — key resolution in depth.
- [Client API](../reference/client-api.md) — exact signatures for the blessed clients.
- [Data Contracts](../reference/data-contracts.md) — `Datum`, `TensorData`, and the wire types.
