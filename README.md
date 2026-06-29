# Cybernetics Python SDK

`cybernetic-physics` is the Python client for the hosted Cybernetics platform —
a Tinker-like API for rollout, sampling, and LoRA training on platform-managed
GPU compute leases.

```bash
pip install cybernetic-physics      # distribution name; the import package is `cybernetics`
export CYBERNETICS_API_KEY="cp_live_..."   # or run: cybernetics auth login
cybernetics doctor                  # read-only API/auth/SFT/RL readiness check
```

## Documentation

Full documentation lives in [`docs/`](docs/src/SUMMARY.md) as an
[mdBook](https://rust-lang.github.io/mdBook/): a [Quickstart](docs/src/quickstart.md),
[concepts](docs/src/concepts/mental-model.md) (the mental model, SFT-vs-RL, a
glossary), task [guides](docs/src/guides/authentication.md) (auth, training,
sampling, checkpoints, [Behavior CI](docs/src/guides/behavior-ci.md)), and a
complete [API / CLI / schema reference](docs/src/reference/client-api.md).

Build it locally with `mdbook serve docs` (or `mdbook build docs`).

Install directly from GitHub when you want the current repo version:

```bash
uv add "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git"
pip install "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git"
```

For `pyproject.toml`:

```toml
dependencies = [
  "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git",
]
```

```python
import cybernetics
from cybernetics import types

service_client = cybernetics.ServiceClient(project_id="robotics-lab")
training_client = service_client.create_lora_training_client(base_model="dreamzero-droid")

tokenizer = training_client.get_tokenizer()  # requires the [tokenizers] extra
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokenizer.encode("example")),
    loss_fn_inputs={
        "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
        "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
    },
)
training_client.forward_backward([datum], "cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

## Authentication

The SDK authenticates to the control plane with your `cp_live_` key sent as
`Authorization: Bearer ...`. Keys are resolved in this order:

1. an explicit `api_key=` argument,
2. the `CYBERNETICS_API_KEY` environment variable,
3. the stored login written by `cybernetics auth login`
   (`~/.config/cybernetics/auth.json`).

## Readiness checks

Run `cybernetics doctor` before launching GPU work. It checks API reachability,
authentication, advertised training/sampling support, and DreamZero SFT/RL
capabilities without creating a Worldlines session, model, compute lease, or
future.

```bash
CYBERNETICS_BASE_URL=https://luc-api.cyberneticphysics.com cybernetics doctor
cybernetics doctor --require-rl
cybernetics --format json doctor
```

`--require-rl` exits nonzero unless the backend advertises the requested
DreamZero RL loss family (`flow_rwr` by default). This keeps SFT-only
deployments usable while making RL readiness explicit in CI and runbooks.

## DreamZero examples

The repository ships an SDK-native DreamZero SFT smoke at
`examples/dreamzero_sft_smoke.py`. Its default mode is local-only and safe for
CI:

```bash
python examples/dreamzero_sft_smoke.py
python examples/dreamzero_rl_smoke.py
```

Run `cybernetics doctor` first, then add `--remote-run` only when you want the
example to create a hosted session/model, run `forward_backward`, apply
`optim_step`, and save a checkpoint. Remote examples cancel their session on
exit by default; pass `--keep-lease` only when you deliberately want to debug
inside the paid container. Use `--timeout` to bound cold-start/model-create
waits. During hosted startup, SDK queue logs surface sanitized provider progress
and worker-heartbeat waits when the control plane has that detail; once the
backend accepts work, a cold DreamZero `create_model` can also report active
`register_model` while Wan/DreamZero assets hydrate and load.

For RL readiness, use:

```bash
cybernetics doctor --require-rl
python examples/dreamzero_sft_smoke.py --remote-run --timeout 2400 --cleanup-timeout 300
python examples/dreamzero_rl_smoke.py --remote-run --timeout 2400 --cleanup-timeout 300
```

The RL smoke sends a tiny synthetic DreamZero trajectory through `flow_rwr`.
That validates the same Tinker-compatible `TrainingClient.forward_backward`
surface as SFT while exercising the DreamZero RL loss path. On a cold backend
image the first run may spend several minutes pulling and extracting the hosted
backend image before the worker heartbeat appears, then hydrate the
Wan/DreamZero asset cache and load sharded model weights. Use a timeout large
enough for that first run, and use `cybernetics doctor --require-rl` before
spending GPU time.

## Optional dependencies

| Extra | Adds | Needed for |
|---|---|---|
| `tokenizers` | `transformers` | `get_tokenizer()` |
| `aiohttp` | `aiohttp` | the aiohttp transport |
| `torch` | `torch` | local tensor interop |
| `all` | all of the above | everything |

`numpy` is a core dependency. `transformers`, `aiohttp`, and `torch` are not —
install the relevant extra (`pip install 'cybernetic-physics[tokenizers]'`) when
you need them.
