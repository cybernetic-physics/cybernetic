# Cybernetics Python SDK

`cybernetic-physics` is the Python client for the hosted Cybernetics platform —
a Tinker-like API for rollout, sampling, and LoRA training on platform-managed
GPU compute leases, plus robotics and simulation-asset helpers for hosted robot
workflows.

```bash
pip install cybernetic-physics      # distribution name; the import package is `cybernetics`
export CYBERNETICS_API_KEY="cp_live_..."   # or run: cybernetics auth login
cybernetics doctor                  # read-only API/auth/SFT/RL readiness check
```

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

## Simulation assets

The SDK includes an MVP `sim` namespace for packaging local simulation assets as
Cybernetics environments, launching hosted Isaac preview sessions, and producing
asset references that can be reused by RobotTask specs.

```bash
cybernetics sim inspect ./scene-folder --format json
cybernetics sim import ./scene.usdz --name warehouse-demo
cybernetics sim import ./robot.urdf --bundle-path robot.bundle.zip --source-url https://example.test/robot
cybernetics sim launch cybernetics://envs/env_.../versions/ver_... --wait
cybernetics sim render ./scene-folder --root-stage scene.usd --wait --out preview.jpg
```

The top-level `cybernetics.Client` is a composition root. Product namespaces
attach under it, so simulation helpers live at `client.sim` without making
`cybernetics.sim` own the package root.

```python
import cybernetics

with cybernetics.Client() as client:
    imported = client.sim.import_asset("./scene-folder", name="warehouse-demo")
    sim_asset_ref = imported.to_asset_ref().to_dict()

    preview = client.sim.render(imported, wait=True, out="preview.jpg")
    print(sim_asset_ref["uri"])
    print(preview.preview_url)
    print(preview.launch_url)
```

`cybernetics.sim` owns asset packaging, import, preview, render, catalog, and
launch helpers. `cybernetics.robotics` owns `RobotTaskSpec`, backend adapters,
run records, policy artifacts, replay artifacts, datasets, and evaluation
records. The two compose through plain serialized asset references:
`client.sim.import_asset(...).to_asset_ref().to_dict()` can be placed in
`RobotTaskSpec.asset_refs[]` without making robotics import the sim namespace.

`SimImportResult.to_asset_ref()` returns a `simulation-asset-ref/v1` descriptor:

```python
{
    "schema_version": "simulation-asset-ref/v1",
    "ref_kind": "environment_version",  # or "local_bundle" / "catalog_asset"
    "uri": "cybernetics://envs/env_.../versions/ver_...",
    "env_id": "env_...",
    "version_id": "ver_...",
    "root_stage_relpath": "scene.usd",
    "asset_kind": "usd_stage",
    "compatibility_status": "ready_to_render",
    "content_sha256": "...",
    "metadata": {},
}
```

RobotTask code consumes that value as an opaque descriptor:

```python
from cybernetics.robotics import RobotTaskSpec

task_payload = build_robot_task_payload(...)
task_payload["asset_refs"] = [sim_asset_ref]
task = RobotTaskSpec.from_dict(task_payload)
```

The current MVP renders USD-family assets (`.usd`, `.usda`, `.usdc`, `.usdz`)
by creating an environment version and starting a hosted session. URDF, Xacro,
SDF, and MJCF files are detected and packaged as `needs_conversion` until the
converter path lands. Public `/sim/<slug>` artifact pages are also future work;
`cybernetics sim render --public` fails explicitly instead of pretending to
publish a durable public artifact.

Uploaded bundle manifests intentionally avoid host-local absolute source paths.
They keep safe provenance only: source basename, optional user-supplied
`--source-url`, root stage, asset kind, compatibility status, and per-file
hashes/sizes.

`cybernetics sim render` is preview/evidence plumbing. It answers whether an
asset can be imported, launched, and visually inspected. Robot task success,
policy evaluation, rollout records, replay evidence, datasets, and VLA/eval
records remain owned by `cybernetics.robotics`.

## RobotTask SDK contracts

`cybernetics.robotics` provides dependency-light contracts and helper adapters
for robot workflows:

- `RobotTaskSpec` for task definitions and simulator backend configuration
- `RobotEnv` / `StepResult` for backend adapter shape
- `RobotRunRecord` for rollout records
- `PolicyArtifact` for policy/checkpoint metadata
- `TrajectoryDatasetArtifact`, replay helpers, VLA eval records, world-model
  artifact metadata, and provider templates such as Unitree G1

The base robotics package is designed to import without sim, Isaac, ROS2,
MuJoCo, Worldlines, or Cosmos runtime packages installed. Heavy backend
execution belongs behind backend adapters, not in the package import path.

## Robot tasks

The `robotics` namespace defines the RobotTask contract, and
`cybernetics.Client().robot_tasks` provides the first ergonomic SDK facade for
loading, validating, locally smoke-testing, and writing task/policy artifacts.

```python
import cybernetics

client = cybernetics.Client()

task = client.robot_tasks.load("robot-task.json")
run = client.robot_tasks.run_fixture(task, "artifacts/fixture-run", seed=42)

policy = client.robot_tasks.policy_artifact(
    task,
    artifact_id="pol_fixture",
    created_by_run_id=run.run_record.run_id,
    checkpoint_uri="worldlines://fixture/checkpoints/latest",
    policy_format="worldlines",
    eval_metrics={"success_rate": 1.0},
    rollout_artifacts=[str(run.rollout_path)],
)
client.robot_tasks.write_policy_artifact(policy, "artifacts/policy.json")
```

This facade is intentionally dependency-light. It validates serialized
`RobotTaskSpec` dictionaries, writes stable JSON artifacts, and runs the fixture
environment path without importing MuJoCo, Isaac, ROS2, Unitree, Worldlines, or
Cosmos runtime packages. Hosted training, LocoMuJoCo adapters, and Isaac/Neko
replay can build behind the same task/run/policy artifact boundary.

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
