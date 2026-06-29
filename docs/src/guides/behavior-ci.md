# Behavior CI

**Behavior CI runs a robot policy through a pinned simulation eval and emits a red/green verdict, per-trial metrics, and replay video as a stable `behavior-ci/v1` artifact bundle.** The tagline from the source says it best:

> CodeRabbit reviews whether the code looks right. Cybernetic Physics reviews whether the robot still works.

You wire it into a pull request the same way you wire a unit-test job: a CLI command returns an exit code, the exit code drives a GitHub check red or green, and a sticky comment summarizes the run. This page is the task recipe — author three files, run the command, read the bundle, and paste in a reference workflow.

For the exhaustive field-by-field schema, see the [Behavior CI config reference](../reference/behavior-ci-config.md). For the conceptual picture of policies, evals, and worlds, see the [mental model](../concepts/mental-model.md).

> Behavior CI lives in the `cybernetics.behavior_ci` package. It is **not** part of the core `import cybernetics` surface — it is imported lazily by the `cybernetics behavior-ci` CLI and pulls no heavy dependencies. The high-level clients (`ServiceClient`, `TrainingClient`, `SamplingClient`, `RestClient`) are unrelated to this subsystem; you drive Behavior CI through the CLI or the `BehaviorCiRunner` class shown at the end.

---

## Install

```bash
pip install 'cybernetic-physics[behavior-ci]'
```

The install name is `cybernetic-physics`; the import package is `cybernetics`. The `[behavior-ci]` extra pulls in PyYAML, which the runner needs to read the config and eval YAML. Without it you get a `ConfigError` telling you to install the extra.

Fixture mode (the default, described below) needs **no** API key and makes **no** network calls. You only need credentials for the hosted `isaac-session` adapter — see [Authentication](./authentication.md).

---

## The three files you author

A Behavior CI repo checks in three things. All examples below are fixture-mode and copy-pasteable; they are the exact fixtures the runner is tested against.

### 1. `cybernetic-behavior-ci.yaml` — the central manifest

```yaml
schema_version: cybernetic-behavior-ci-config/v1
project: unitree-g1-vla-policies
robot: Unitree G1-compatible humanoid proxy
artifacts:
  out: artifacts/behavior-ci
simulator:
  adapter: fixture
  session:
    scene_env: behavior-ci-tabletop-welding
    camera: /World/Cameras/BehaviorCI_PassFailCamera
evals:
  obstacle_shift: evals/g1_weld_obstacle_shift.yaml
```

- `simulator.adapter` is `fixture` or `isaac-session`. `fixture` is deterministic and offline; `isaac-session` boots a hosted Isaac Sim session (see the honesty note below).
- `evals` maps a short name (used on the CLI as `--eval obstacle_shift`) to a path. You may also write `evals: {obstacle_shift: {path: evals/...yaml}}`.
- `artifacts.out` defaults to `artifacts/behavior-ci` when omitted.
- Paths are resolved **relative to the directory containing the config file**.

The full surface (replay provenance gating, hosted-session settings, per-backend config) is in the [config reference](../reference/behavior-ci-config.md).

### 2. The eval YAML — the pinned behavior eval

```yaml
schema_version: behavior-ci-eval/v1
world: tabletop_welding_obstacle_shift_v1
behavior: g1_weld_approach
runs: 8
checks:
  target_reach:
    metric: torch_tip_distance_to_target_cm
    operator: "<="
    value: 2.0
  collision_free:
    metric: collision_count
    operator: "=="
    value: 0
  safety_zone_clear:
    metric: restricted_zone_intrusions
    operator: "=="
    value: 0
  base_stable:
    metric: max_base_tilt_degrees
    operator: "<="
    value: 5.0
  timeout_free:
    metric: elapsed_seconds
    operator: "<"
    value: 30
scenarios:
  - {obstacle_shift_cm: 3,  required_clearance_cm: 3}
  - {obstacle_shift_cm: 5,  required_clearance_cm: 5}
  - {obstacle_shift_cm: 4,  required_clearance_cm: 4}
  - {obstacle_shift_cm: 11, required_clearance_cm: 11, stresses: safety_zone}
  - {obstacle_shift_cm: 6,  required_clearance_cm: 6}
  - {obstacle_shift_cm: 13, required_clearance_cm: 13, stresses: collision}
  - {obstacle_shift_cm: 5,  required_clearance_cm: 5}
  - {obstacle_shift_cm: 10, required_clearance_cm: 10, stresses: timeout}
```

- `runs` must be in `[1, 64]`.
- Each `checks` entry is one scalar rule: `metric <operator> value`. Allowed operators are `==`, `!=`, `<`, `<=`, `>`, `>=`. Checks are `required: true` by default; a trial passes when every *required* check passes.
- `scenarios` is a per-run list (indexed by run); runs beyond the list length get an empty scenario. In fixture mode `required_clearance_cm` and `stresses` drive whether a trial fails — see [How the verdict is computed](#how-the-verdict-is-computed).
- A check that fails maps to a PR-comment failure code: `safety_zone_clear → SAFETY_ZONE_INTRUSION`, `collision_free → OBSTACLE_COLLISION`, `timeout_free → TARGET_TIMEOUT`, `target_reach → TARGET_MISS`, `base_stable → BASE_INSTABILITY`.

### 3. The policy `.pt` manifest

In this subsystem a `.pt` file is an **honest JSON manifest**, not a Torch checkpoint. The `controller` block carries *readable* parameters that actually drive behavior, instead of a hidden lookup by `policy_id`.

```json
{
  "schema_version": "behavior-ci-policy/v1",
  "policy_id": "g1_weld_approach_v19",
  "display_filename": "g1_weld_approach_v19.pt",
  "behavior": "g1_weld_approach",
  "robot": "Unitree G1-compatible humanoid proxy",
  "backend": "scripted-vla-shim",
  "controller": {"type": "scripted_trajectory", "clearance_margin_cm": 14.0}
}
```

- `backend` must be one of `scripted-vla-shim`, `gr00t-adapter-stub`, or `real-vla`.
- The policy's `behavior` must equal the eval's `behavior`, or the run fails with a `ConfigError` (exit `2`).
- **Honesty:** only `real-vla` claims a real learned policy — and `real-vla` is **not implemented yet**. Selecting it raises a `ConfigError` ("policy backend 'real-vla' is not implemented yet … wire RealVlaBackend (CYB-68)"). `scripted-vla-shim` and `gr00t-adapter-stub` both resolve to the same deterministic scripted backend and report `policy_backend_real_vla = false` in provenance.

To reproduce the red/green demo, drop in a second manifest that differs **only** in `clearance_margin_cm` — e.g. `g1_weld_approach_v18.pt` with `"clearance_margin_cm": 6.0`. v18 fails three trials; v19 (margin `14.0`) passes all eight.

---

## How the verdict is computed

In **fixture mode** there is no Isaac and no learned policy. Each trial's metrics are a transparent function of the per-run scenario and the controller's `clearance_margin_cm`: a stressed trial fails exactly when `clearance_margin_cm < required_clearance_cm`. That is why a regression is a property of a readable parameter, not a hidden table keyed by `policy_id`.

The aggregate `status` is `passed` only if **every** trial passes **all** its required checks. Any failing trial makes the whole run `failed`.

This deterministic evaluator core is shared by both adapters — the hosted `isaac-session` adapter feeds the same evaluator with metrics it reads back from a live session.

---

## Run it

```bash
cybernetics behavior-ci validate-config --config cybernetic-behavior-ci.yaml
```

```text
OK: project=unitree-g1-vla-policies robot=Unitree G1-compatible humanoid proxy adapter=fixture evals=['obstacle_shift']
```

`validate-config` parses the config (and resolves it) without running anything. A malformed config exits `2`.

Now run an eval against a policy:

```bash
cybernetics behavior-ci run \
  --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v19.pt \
  --eval obstacle_shift \
  --out artifacts/behavior-ci
```

Passing policy (`v19`, margin `14.0`):

```text
[PASS] g1_weld_approach_v19.pt: 8/8 trials passed (adapter=fixture, replay=fixture-generated)
```

Regressed policy (`v18`, margin `6.0`) — note this run exits `1`:

```text
[FAIL] g1_weld_approach_v18.pt: 5/8 trials passed (adapter=fixture, replay=fixture-generated)
  - run 3 SAFETY_ZONE_INTRUSION: torch path entered the red restricted zone
  - run 5 OBSTACLE_COLLISION: end-effector collided with the shifted obstacle
  - run 7 TARGET_TIMEOUT: failed to reach the weld start pose before timeout
```

`run` flags:

| Flag | Required | Meaning |
|---|---|---|
| `--config` | yes | Path to `cybernetic-behavior-ci.yaml` (must exist). |
| `--policy-ref` | yes | Path to a policy `.pt` manifest. |
| `--eval` | yes | An eval name from the config, or a direct path to an eval YAML. |
| `--out` | no | Output dir for the bundle. Defaults to `artifacts.out` from the config. |
| `--commit` | no | Commit SHA to record. Defaults to `git rev-parse --short HEAD` (or `unknown`). |
| `--keep-session` | no | For `isaac-session` only: do not stop the hosted session on exit. |

Print the rendered comment from a produced bundle:

```bash
cybernetics behavior-ci render-comment --artifact-dir artifacts/behavior-ci
```

This just prints `comment.md` to stdout (exit `4` if the file is missing). It does **not** re-run the eval. See the full CLI surface in the [CLI reference](../reference/cli.md).

---

## The exit-code contract

The whole CI integration hangs on these exit codes. They are stable and you should wire your check status directly to them.

| Exit | Name | Meaning |
|---|---|---|
| `0` | pass | Behavior passed and the bundle is valid. |
| `1` | regression | A behavior regression. **Artifacts are still written** — upload them. |
| `2` | config | Invalid input/config/manifest/eval (a `ConfigError`). |
| `3` | infra | Hosted session/MCP failure (`isaac-session` only — an `IsaacSessionError`). |
| `4` | contract | The produced bundle or report violated the `behavior-ci/v1` contract. |

Exit `1` is the only "the robot got worse" signal. `2`, `3`, and `4` mean the run could not produce a trustworthy verdict — treat them as a broken pipeline, not a robot regression. Exit `3` only ever comes from the hosted adapter; fixture mode never hits it. See [Errors & Troubleshooting](../reference/errors.md) for what each maps to.

---

## The `behavior-ci/v1` bundle

`run` writes a stable bundle to `--out`. The layout is a contract — keys consumed by reports and comments are never renamed (fields may be *added* compatibly):

```text
artifacts/behavior-ci/
├── result.json               top-level verdict + provenance
├── metrics.json              aggregate + per-run metrics/checks
├── comment.md                sticky PR comment (markdown)
├── report/index.html         self-contained static report
├── replays/replay-passed.mp4 replay clip — written whenever a trial passed
├── replays/replay-failed.mp4 replay clip — written ONLY when a run failed
├── manifest.normalized.json  the effective, resolved run inputs
└── provenance.json           honesty provenance, standalone for auditing
```

An honest bundle never ships a `replay-failed.mp4` for a green run. The runner validates the bundle after writing it and raises a `ContractError` (exit `4`) if anything is missing, empty, or not a real MP4.

### Provenance, never overclaimed

Every bundle writes a `honesty` block (mirrored in `provenance.json`). This is where the system refuses to lie about what produced the verdict:

| Field | Fixture-mode value | Meaning |
|---|---|---|
| `simulator_adapter` | `fixture` | Which adapter ran the trials. |
| `replay_source` | `fixture-generated` | Where the replay came from. |
| `policy_backend` | `scripted-vla-shim` | The resolved policy backend. |
| `policy_backend_real_vla` | `false` | Whether a real learned VLA ran. |
| `production_eval_path_used` | `false` | Whether the production eval path was used. |
| `artifact_contract_version` | `behavior-ci/v1` | The bundle contract version. |

In fixture mode the provenance `notes` read, verbatim:

```text
Fixture mode: trial outcomes computed deterministically from readable controller
parameters (not a learned policy, not live Isaac). Replay source: fixture-generated.
```

A `fixture-generated` replay is a clearly-labelled placeholder MP4, and the HTML report renders a loud banner over it:

```text
⚠ Placeholder clip — NOT a real Isaac capture. Run the hosted isaac-session
workflow to attach genuine replay video from the pass/fail camera.
```

To attach **real** replay evidence while still running offline, point `artifacts.replay_source_dir` at a directory of checked-in Isaac captures (`replay-passed.mp4` / `replay-failed.mp4`); those are labelled `checked-in-demo-evidence` instead. You can require a minimum provenance with `artifacts.require_replay_provenance` — see the [config reference](../reference/behavior-ci-config.md).

---

## Reference GitHub Actions workflow

This wires the exit codes to the check status and upserts a single sticky PR comment. The comment carries a stable marker (`<!-- cybernetic-behavior-ci -->`) so we update one comment instead of spamming the thread. The example runs **fixture mode**, so it needs no secrets; for the hosted adapter add `CYBERNETICS_API_KEY` (see below).

```yaml
name: behavior-ci
on: [pull_request]

permissions:
  contents: read
  pull-requests: write   # required to upsert the PR comment

jobs:
  behavior-ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install 'cybernetic-physics[behavior-ci]'

      - name: Run Behavior CI
        id: run
        run: |
          set +e
          cybernetics behavior-ci run \
            --config cybernetic-behavior-ci.yaml \
            --policy-ref policies/g1_weld_approach_v19.pt \
            --eval obstacle_shift \
            --out artifacts/behavior-ci
          echo "code=$?" >> "$GITHUB_OUTPUT"

      # Always upload the bundle — even on a regression (exit 1).
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: behavior-ci
          path: artifacts/behavior-ci

      # Upsert the sticky comment from the rendered comment.md.
      - name: Upsert PR comment
        if: always() && github.event_name == 'pull_request'
        env:
          GH_TOKEN: ${{ github.token }}
          PR: ${{ github.event.pull_request.number }}
        run: |
          set -e
          cybernetics behavior-ci render-comment \
            --artifact-dir artifacts/behavior-ci > comment.md || exit 0
          MARKER='<!-- cybernetic-behavior-ci -->'
          ID=$(gh pr view "$PR" --json comments \
                 --jq ".comments[] | select(.body | contains(\"$MARKER\")) | .url" \
                 | head -n1)
          if [ -n "$ID" ]; then
            gh api -X PATCH "${ID/html\/.*\/issuecomment-/issues\/comments\/}" \
              -f body="$(cat comment.md)" || gh pr comment "$PR" --body-file comment.md
          else
            gh pr comment "$PR" --body-file comment.md
          fi

      # Translate the run's exit code into the check status.
      #   0 pass · 1 regression · 2 config · 3 infra · 4 contract
      - name: Enforce verdict
        if: always()
        run: |
          code='${{ steps.run.outputs.code }}'
          case "$code" in
            0) echo "Behavior passed."; exit 0 ;;
            1) echo "::error::Behavior regression."; exit 1 ;;
            2) echo "::error::Invalid config/manifest/eval."; exit 1 ;;
            3) echo "::error::Hosted session/infra failure."; exit 1 ;;
            4) echo "::error::Artifact contract failure."; exit 1 ;;
            *) echo "::error::Unexpected exit $code"; exit 1 ;;
          esac
```

The `render-comment`/`gh` upsert step above is a reference design, not shipped tooling: the CLI gives you the marker-stamped `comment.md`; the matching-and-PATCH logic is standard `gh` glue you adapt to your repo. The exit-code translation, the marker, and the always-upload-on-regression behavior are the load-bearing parts.

---

## Fixture vs. hosted: be clear-eyed

| | `fixture` | `isaac-session` |
|---|---|---|
| Network / credentials | none | hosted session, needs an API key |
| What runs the trial | deterministic function of controller params + scenario | a real Cybernetic Physics Isaac Sim session over HTTP + MCP |
| Replay source | `fixture-generated` placeholder, or `checked-in-demo-evidence` | `isaac-sim-session-video` captured from the named pass/fail camera |
| Exit `3` (infra) possible? | no | yes |

Fixture mode is the fast path for local dev, unit tests, and the default public CI job — it is honest about not being Isaac. The hosted `isaac-session` adapter drives a real session (`POST /v1/sessions` → poll until the bridge is ready → `isaac.*` MCP tool calls → `isaac.capture_video` → stop the session) and stamps `replay_source = isaac-sim-session-video`. To switch, set `simulator.adapter: isaac-session` in the config and provide an API key:

```bash
export CYBERNETICS_API_KEY=cp_live_...   # or run: cybernetics auth login
```

Key resolution order (shared with the rest of the SDK): explicit `api_key=` → `CYBERNETICS_API_KEY` → `CP_API_KEY` → the deprecated `WORLDLINES_API_KEY` (one release only) → the stored login file. Only API keys are secrets; the base URL, MCP URL, workspace, and env id are non-secret config that lives in the config file (or env-var fallbacks). See [Authentication](./authentication.md) for the full story, and note the brand mapping: **Cybernetics** is the SDK; **CP** (`CP_API_KEY`, `cp_live_` keys) is Cybernetic Physics; **Worldlines** is the legacy name surviving only in `WORLDLINES_API_KEY` and `worldlines://` paths.

---

## Driving it from Python

The CLI is a thin wrapper over `BehaviorCiRunner`. If you need to call it in-process (e.g. a custom harness), the runner is the entrypoint — but it is **not** re-exported at the package root; import it from the subpackage:

```python
from cybernetics.behavior_ci import BehaviorCiRunner

runner = BehaviorCiRunner.from_config("cybernetic-behavior-ci.yaml")
result = runner.run_policy(
    policy_ref="policies/g1_weld_approach_v19.pt",
    eval_ref="obstacle_shift",
    out_dir="artifacts/behavior-ci",
)
assert result.passed                      # status == "passed"
print(result.summary["passed_runs"], "/", result.summary["total_runs"])
```

`run_policy` raises `ConfigError`, `IsaacSessionError`, or `ContractError` rather than calling `sys.exit` — the CLI is what maps those to exit codes `2`, `3`, and `4`. On success or regression it returns a `BehaviorCiResult` and writes the bundle.

---

## Related pages

- [Behavior CI config reference](../reference/behavior-ci-config.md) — the full `cybernetic-behavior-ci.yaml`, eval, and policy-manifest schema.
- [CLI reference](../reference/cli.md) — every `cybernetics` command.
- [Errors & Troubleshooting](../reference/errors.md) — what each exit code and error class means.
- [Mental Model](../concepts/mental-model.md) — policies, evals, worlds, and how a behavior is defined.
- [Authentication](./authentication.md) — API keys, the login store, and the CP/Worldlines brand mapping.
- [Checkpoints](./checkpoints.md) — producing the policies you gate with Behavior CI.
