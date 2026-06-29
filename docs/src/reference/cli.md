# CLI

The `cybernetics` console script is the command-line front end to the
Cybernetics SDK. It is installed as part of the package:

```bash
pip install cybernetic-physics    # distribution name
cybernetics --help                # console script (entry point)
```

The distribution on PyPI is **`cybernetic-physics`**; the import package is
**`cybernetics`** (see [Introduction](../introduction.md) for the naming
reconciliation). The `[project.scripts]` entry point maps `cybernetics` to
`cybernetics.cli.__main__:cli`. You can also invoke it as a module:

```bash
python -m cybernetics.cli --help
```

This page is an exhaustive reference for every command, flag, and exit code.
For task-oriented walkthroughs see the guides:
[Authentication](../guides/authentication.md),
[Training](../guides/training.md),
[Checkpoints](../guides/checkpoints.md), and
[Behavior CI](../guides/behavior-ci.md).

> **What the CLI talks to.** Read-only commands (`run`, `checkpoint`) go through
> the supported high-level `RestClient`
> (see [Client API](./client-api.md)). `auth` and `doctor` talk directly to the
> control plane with `httpx`. You never interact with the internal
> Stainless-generated `AsyncCybernetics` / `cybernetics.resources.*` transport —
> do not call it directly.

---

## Naming you will see in CLI output

- **Cybernetics** — the product / SDK.
- **Worldlines** — the legacy name. It survives in checkpoint paths
  (`worldlines://run-id/weights/0001`) and the deprecated `WORLDLINES_API_KEY`
  environment variable. Several command help strings and docstrings still say
  "worldlines checkpoint …" — the command is actually `cybernetics checkpoint …`.
- **CP** — short for Cybernetic Physics, used in the `CP_API_KEY` /
  `CP_API_BASE` environment variables and in `cp_live_` key prefixes.

---

## Global options

`cybernetics` is a Click group. The only root-level option is the output format,
which is read by the read-only subcommands (`run`, `checkpoint`, `doctor`).

| Option | Alias | Values | Default | Effect |
| --- | --- | --- | --- | --- |
| `--format` | `-f` | `table`, `json` | `table` | Output format. `table` renders human-readable tables and progress bars; `json` emits machine-readable JSON and suppresses progress bars. |
| `--help` | `-h` | — | — | Show help and exit. |

The format must come **before** the subcommand, because it is an option on the
root group:

```bash
cybernetics --format json run list
cybernetics -f json checkpoint info worldlines://run-123/weights/final
```

`auth`, `behavior-ci`, and `version` ignore `--format` and print their own
fixed-format output.

### Error handling and exit codes (top level)

`main()` wraps the CLI. A `CyberneticsCliError` prints `Error: <message>` (and an
optional detail line) to **stderr** and exits with that error's `exit_code`.
`Ctrl+C` exits with **130** (standard Unix SIGINT). The `behavior-ci run` and
`doctor --require-rl` subcommands define their own exit-code contracts —
documented in their sections below.

---

## Authentication & environment

API-key resolution is shared by the SDK client and the CLI. The order is
(first match wins):

1. an explicit `--api-key` (only `doctor` exposes this flag),
2. `CYBERNETICS_API_KEY`,
3. `CP_API_KEY` (Cybernetic Physics examples / dev-infra runbooks),
4. `WORLDLINES_API_KEY` — **deprecated**, read for one release, emits a
   `DeprecationWarning`,
5. the stored login file written by `cybernetics auth login`
   (`$XDG_CONFIG_HOME/cybernetics/auth.json`, falling back to
   `~/.config/cybernetics/auth.json`; `0700` dir, `0600` file).

Base-URL resolution is similar: explicit `--base-url` → `CYBERNETICS_BASE_URL` →
`CP_API_BASE` → stored login → default `https://api.cyberneticphysics.com`.

| Variable | Purpose |
| --- | --- |
| `CYBERNETICS_API_KEY` | Primary API key (a `cp_live_…` key). |
| `CP_API_KEY` | Alternative key variable used in CP examples. |
| `WORLDLINES_API_KEY` | Deprecated key fallback (one release only). |
| `CYBERNETICS_BASE_URL` | Override the control-plane origin. |
| `CP_API_BASE` | Alternative base-URL variable. |
| `XDG_CONFIG_HOME` | Relocate the credential directory. |

---

## `cybernetics auth`

Browser login plus local key management. See the
[Authentication guide](../guides/authentication.md) for the recommended flow.

### `auth login`

Run an OAuth 2.0 Device Authorization Grant (RFC 8628) against the control
plane and store the resulting long-lived `cp_live_` key.

| Option | Default | Description |
| --- | --- | --- |
| `--base-url` | `None` (resolves to stored/env/default) | Control-plane API base URL. |
| `--client-name` | `cybernetics-cli` | Label recorded for the minted key. |
| `--no-browser` | off | Do not auto-open a browser; just print the verification URL and code. |

```bash
cybernetics auth login
```

Expected output (stderr carries the prompt; the final line goes to stdout):

```text
  First copy your one-time code: WXYZ-1234
  Then open: https://app.cyberneticphysics.com/device?user_code=WXYZ-1234

Press Enter to open the browser…
✓ Logged in as you@example.com (workspace ws_abc123)
```

Failure modes (each raises a `CyberneticsCliError`):

- The code expires before you approve: `The login code expired. Run 'cybernetics auth login' again.`
- You never approve in time: `Login timed out before approval.`
- The device-code request fails: `Could not start device login (HTTP <code>).`

On success the key is persisted to `auth.json` together with the resolved user
and workspace.

### `auth logout`

Best-effort revoke the stored key, then delete the local credential file. If the
revoke HTTP call fails, the local file is still removed so you end up logged out.

```bash
cybernetics auth logout
```

```text
✓ Logged out.
```

If there is no stored login it prints `Not logged in.` and returns.

### `auth status`

Show the current identity (a `whoami` against `/v1/me`). Exits **1** if no key
is resolvable.

```bash
cybernetics auth status
```

```text
Logged in
  User:      you@example.com
  Workspace: ws_abc123
  Key:       cp_live_abcdef0…
  Source:    stored login
```

`Source` is `stored login` when the resolved key matches the credential file,
otherwise `environment` (the key came from an environment variable). When no key
is found it prints `Not logged in. Run 'cybernetics auth login'.` and exits 1.

### `auth token`

Print the resolved API key to stdout — the intended use is shell capture:

```bash
export CYBERNETICS_API_KEY=$(cybernetics auth token)
```

If no key is found it raises a `CyberneticsCliError` (`No API key found.`) and
exits nonzero.

---

## `cybernetics run`

Read-only inspection of training runs through `RestClient`. See the
[Training guide](../guides/training.md) for context on what a run is.

### `run list`

List training runs, newest activity first, paginating in batches of 100.

| Option | Alias | Default | Description |
| --- | --- | --- | --- |
| `--limit` | — | `20` | Maximum runs to fetch. `--limit=0` fetches **all** runs. |
| `--columns` | `-c` | `id,model,lora,updated,status` | Comma-separated columns (see below). |

Available `--columns` values:

| Name | Header | Meaning |
| --- | --- | --- |
| `id` | Run ID | Training run ID. |
| `model` | Base Model | Base model name. |
| `owner` | Owner | Model owner. |
| `lora` | LoRA | LoRA status and rank. |
| `updated` | Last Update | Last request time. |
| `status` | Status | `OK`, or `Failed` if the run is corrupted. |
| `checkpoint` | Last Checkpoint | Most recent training checkpoint ID. |
| `checkpoint_time` | Checkpoint Time | Time of the last checkpoint. |

An unknown column name raises a Click `BadParameter` error.

```bash
cybernetics run list --limit 5 --columns id,model,owner,status
```

```text
5 training runs (12 more not shown, use --limit to see more)
Run ID                Base Model              Owner        Status
run-abc123            meta-llama/Llama-3-8B   you          OK
run-def456            Qwen/Qwen2.5-7B         you          Failed
...
```

### `run info`

Show full detail for one run, including both the last training checkpoint and
the last sampler checkpoint, and any user metadata.

| Argument | Description |
| --- | --- |
| `RUN_ID` | Training run ID (required positional). |

```bash
cybernetics run info run-abc123
```

```text
Training Run: run-abc123
Property                    Value
Run ID                      run-abc123
Base Model                  meta-llama/Llama-3-8B
Owner                       you
LoRA                        Yes (Rank 16)
Last Update                 2026-06-20 14:03:11
Status                      OK
Last Training Checkpoint    0042
  - Time                    2026-06-20 14:00:00
  - Path                    worldlines://run-abc123/weights/0042
Last Sampler Checkpoint     0042
  - Time                    2026-06-20 14:00:30
  - Path                    worldlines://run-abc123/sampler_weights/0042
```

`--format json` returns the full `model_dump()` of the `TrainingRun`.

---

## `cybernetics checkpoint`

Manage checkpoints. Almost every subcommand takes a **worldlines path** of the
form `worldlines://<run-id>/<weights|sampler_weights>/<checkpoint-id>`; a path
that does not start with `worldlines://` is rejected with a
`CyberneticsCliError`. See the [Checkpoints guide](../guides/checkpoints.md).

> Several `checkpoint` help strings and docstrings still read
> "worldlines checkpoint …". The command is `cybernetics checkpoint …`; the
> `worldlines://` scheme in **paths** is intentional and current.

### `checkpoint list`

List checkpoints, either for a single run or across all of your runs.

| Option | Default | Description |
| --- | --- | --- |
| `--run-id` | `None` | List checkpoints for this run (no pagination — returns all of the run's checkpoints). |
| `--limit` | `20` | Cap when listing across **all** runs. `--limit=0` shows all (paginated in batches of 1000). Ignored when `--run-id` is given. |

```bash
cybernetics checkpoint list --run-id run-abc123
```

Table columns: `Checkpoint ID`, `Type`, `Size`, `Public`, `Created`,
`Expires`, `Path`. `Expires` shows `Never` when there is no TTL.

```text
2 checkpoints for run run-abc123
Checkpoint ID  Type      Size     Public  Created              Expires  Path
0041           training  1.2 GiB  No      2026-06-20 13:30:00  Never    worldlines://run-abc123/weights/0041
0042           training  1.2 GiB  No      2026-06-20 14:00:00  Never    worldlines://run-abc123/weights/0042
```

### `checkpoint info`

Show one checkpoint plus LoRA detail pulled from its training run.

| Argument | Description |
| --- | --- |
| `CHECKPOINT_PATH` | A `worldlines://` path (required). |

```bash
cybernetics checkpoint info worldlines://run-abc123/weights/0042
```

```text
Checkpoint: 0042
Property            Value
Checkpoint ID       0042
Type                training
Cybernetics Path    worldlines://run-abc123/weights/0042
Size                1.2 GiB
Public              No
Created             2026-06-20 14:00:00
Expires             Never
Training Run ID     run-abc123
LoRA                Yes (Rank 16)
```

### `checkpoint publish` / `checkpoint unpublish`

Toggle public accessibility of a checkpoint. Only the owner of the training run
can do this. Both take a single `CHECKPOINT_PATH` argument and print nothing on
success (they exit 0).

```bash
cybernetics checkpoint publish   worldlines://run-abc123/weights/0042
cybernetics checkpoint unpublish worldlines://run-abc123/weights/0042
```

### `checkpoint set-ttl`

Set or clear a checkpoint's expiration. You must pass exactly one of `--ttl` or
`--remove`.

| Option | Default | Description |
| --- | --- | --- |
| `--ttl` | `None` | TTL in seconds from now (positive integer). |
| `--remove` | off | Remove the expiration entirely. |

```bash
# Expire this checkpoint in 7 days
cybernetics checkpoint set-ttl worldlines://run-abc123/weights/0042 --ttl 604800

# Make it permanent again
cybernetics checkpoint set-ttl worldlines://run-abc123/weights/0042 --remove
```

Passing neither, or both, raises a `CyberneticsCliError`
(`Must specify either --ttl <seconds> or --remove` /
`Cannot specify both --ttl and --remove`).

### `checkpoint delete`

Permanently delete one or more checkpoints. **This cannot be undone.** Deletions
run concurrently (up to 32 in flight) with a progress bar.

| Argument / Option | Default | Description |
| --- | --- | --- |
| `CHECKPOINT_PATHS…` | — | Zero or more explicit `worldlines://` paths. |
| `--run-id` | `None` | Delete all checkpoints for a run (cannot combine with explicit paths). |
| `--type` | `None` | Filter by type: `weights` or `sampler_weights`. Requires `--run-id`. |
| `--before` | `None` | Created-before date (UTC, ISO 8601). Requires `--run-id`. |
| `--after` | `None` | Created-after date (UTC, ISO 8601). Requires `--run-id`. |
| `--yes` / `-y` | off | Skip the confirmation prompt. |

Argument rules (each violation raises a `CyberneticsCliError`):

- You must supply either paths or `--run-id`.
- You cannot supply both paths and `--run-id`.
- `--type`, `--before`, `--after` only apply with `--run-id`.

Dates accept `2024-01-01`, `2024-01-01T08:00:00`, `2024-01-01T08:00:00Z`, or an
explicit offset; date-only values are treated as midnight UTC. A bad date raises
`Invalid date: …`.

```bash
# Explicit paths
cybernetics checkpoint delete \
  worldlines://run-abc123/weights/0001 worldlines://run-abc123/weights/0002

# Everything for a run, training weights only, older than a date, no prompt
cybernetics checkpoint delete --run-id run-abc123 --type weights \
  --before 2026-01-01 --yes
```

Without `--yes` you get a summary and a confirmation prompt before anything is
deleted. With `--format json` the result is
`{"deleted_count": N, "failed": [{"worldlines_path": …, "error": …}]}`.

### `checkpoint download`

Download a checkpoint archive and extract it into a directory named after the
checkpoint (path with `worldlines://` stripped and `/` replaced by `_`). The
`.tar` is deleted after successful extraction. Extraction is hardened: tar
members that escape the target directory, or that are symlinks/hardlinks, are
rejected.

| Option | Alias | Default | Description |
| --- | --- | --- | --- |
| `--output` | `-o` | current directory | Parent directory for the extracted checkpoint. Created if missing. |
| `--force` | — | off | Overwrite the target directory if it already exists. |

```bash
# Creates ./run-123_weights_final/
cybernetics checkpoint download worldlines://run-123/weights/final

# Creates ./models/run-123_weights_final/
cybernetics checkpoint download worldlines://run-123/weights/final --output ./models/
```

If the target directory exists and `--force` is not given, the command fails
with `Target directory already exists: …`.

### `checkpoint push-hf`

Upload a checkpoint to the Hugging Face Hub as a PEFT/LoRA adapter. Requires the
optional `huggingface_hub` dependency and a Hub login (`hf auth login`); a
missing dependency or login raises a `CyberneticsCliError` with install/login
guidance.

| Option | Alias | Default | Description |
| --- | --- | --- | --- |
| `--repo` | `-r` | derived from the run | Target Hub repo ID, e.g. `username/my-lora-adapter`. If omitted, a name is derived from the base model and run ID. |
| `--public` | — | off (private) | Create/upload to a public repo. |
| `--revision` | — | `None` | Target branch/revision. |
| `--commit-message` | — | `None` | Commit message for the upload. |
| `--create-pr` | — | off | Open a pull request instead of pushing to the main branch. |
| `--allow-pattern` | — | — | Only upload files matching this glob (repeatable). |
| `--ignore-pattern` | — | — | Skip files matching this glob (repeatable). |
| `--no-model-card` | — | off | Do not generate a `README.md` model card when one is missing. |

```bash
cybernetics checkpoint push-hf \
  worldlines://run-abc123/sampler_weights/0042 \
  --repo you/llama3-8b-mylora --public --commit-message "Export from Cybernetics"
```

The command validates that the archive actually contains a PEFT adapter
(`adapter_config.json` plus `adapter_model.safetensors` or `adapter_model.bin`)
and a `checkpoint_complete` marker before uploading; otherwise it errors out. If
the target repo already holds a *different* Cybernetics checkpoint (detected via
the `worldlines://` line in its README), the upload is refused.

---

## `cybernetics doctor`

Read-only readiness probe. It checks the control-plane health endpoint, your
auth, and the backend's advertised capabilities **without creating a session or
lease**. Nothing is billed and nothing is launched.

| Option | Default | Description |
| --- | --- | --- |
| `--base-url` | resolved | Control-plane API base URL. |
| `--api-key` | resolved | API-key override. Prefer `CYBERNETICS_API_KEY` or `cybernetics auth login`. |
| `--rl-loss` | `flow_rwr` | DreamZero RL loss family to require/report. |
| `--require-rl` | off | Exit nonzero unless the backend advertises DreamZero RL readiness. |

If no API key resolves, it raises `No API key found.` Each readiness row is
`ready` / `unavailable` / `unknown` (the latter when the backend does not
advertise a definite capability).

```bash
cybernetics doctor
```

```text
Cybernetics Doctor
Check          Status   Detail
API            ready    https://api.cyberneticphysics.com (default)
Auth           ready    stored login
Health         ready    /health HTTP 200
Training       ready
Sampling       ready
DreamZero SFT  ready    cross_entropy
DreamZero RL   unknown  flow_rwr not advertised by this backend
```

The `--require-rl` flag is a CI gate: the command exits **1** unless
`DreamZero RL` resolves to `ready` (it stays 0 otherwise). Use it to fail a
pipeline early when the backend cannot run RL:

```bash
cybernetics doctor --require-rl --rl-loss flow_rwr || echo "RL not available"
```

`--format json` emits the full readiness object (`base_url`, `auth_source`,
`health`, `supports_training`, `supports_sampling`, `loss_families`,
`dreamzero_sft_ready`, `dreamzero_rl_ready`, `dreamzero_rl_loss`,
`dreamzero_rl_unavailable_reason`, …).

---

## `cybernetics behavior-ci`

Run a robot-behavior regression check and emit the `behavior-ci/v1` artifact
bundle consumed by the GitHub Action. See the
[Behavior CI guide](../guides/behavior-ci.md) and the config reference at
[Behavior CI Config](./behavior-ci-config.md). The config file is
`cybernetic-behavior-ci.yaml`.

This group ignores the root `--format` flag and prints its own fixed output.

### Exit codes

The `run` subcommand's exit code is the contract that turns the CI check
red/green. This table currently lives only in the module docstring — it is
surfaced here:

| Exit code | Meaning |
| --- | --- |
| `0` | Behavior passed and artifacts are valid. |
| `1` | Behavior regression (artifacts are still written). |
| `2` | Invalid input/config. |
| `3` | Hosted infrastructure / session failure. |
| `4` | Artifact / report contract failure. |

`validate-config` exits `2` on an invalid config (otherwise `0`).
`render-comment` exits `4` if there is no `comment.md` in the artifact
directory.

### `behavior-ci validate-config`

Parse and validate a config file without running anything.

| Option | Required | Description |
| --- | --- | --- |
| `--config` | yes | Path to `cybernetic-behavior-ci.yaml` (must exist). |

```bash
cybernetics behavior-ci validate-config --config cybernetic-behavior-ci.yaml
```

```text
OK: project=warehouse-arm robot=franka adapter=isaac evals=['pick-place', 'stack']
```

An invalid config prints `invalid config: <detail>` to stderr and exits **2**.

### `behavior-ci run`

Run an eval against a policy and write the artifact bundle.

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--config` | yes | — | Path to the config file (must exist). |
| `--policy-ref` | yes | — | Path to a policy manifest (`.pt`). |
| `--eval` | yes | — | Eval name (from the config) or a path. |
| `--out` | no | `None` | Artifact bundle output directory. |
| `--commit` | no | git `HEAD` | Commit SHA to record. |
| `--keep-session` | no | off | Do not stop the hosted session on exit. |

```bash
cybernetics behavior-ci run \
  --config cybernetic-behavior-ci.yaml \
  --policy-ref ./artifacts/policy.pt \
  --eval pick-place \
  --out ./behavior-ci-out
```

```text
[PASS] policy.pt: 19/20 trials passed (adapter=isaac, replay=hosted)
```

On a behavior regression the verdict is `[FAIL]`, each failing trial is listed
(`  - run <n> <code>: <message>`), the artifacts are still written, and the
command exits **1**. A config error exits 2, a hosted-session/infrastructure
error exits 3, and an artifact-contract error exits 4 — each printing a
matching line to stderr.

### `behavior-ci render-comment`

Print the PR-comment markdown from a produced artifact bundle (the bundle's
`comment.md`).

| Option | Required | Description |
| --- | --- | --- |
| `--artifact-dir` | yes | Directory containing a produced bundle (must exist). |

```bash
cybernetics behavior-ci render-comment --artifact-dir ./behavior-ci-out
```

If `comment.md` is missing it prints `no comment.md in <dir>` to stderr and
exits **4**.

---

## `cybernetics version`

Print the installed package version.

```bash
cybernetics version
```

```text
cybernetics 0.16.1
```

If the version cannot be imported it prints `cybernetics (version unavailable)`.

---

## See also

- [Authentication guide](../guides/authentication.md) — login, keys, env vars.
- [Checkpoints guide](../guides/checkpoints.md) — what checkpoints and
  `worldlines://` paths are.
- [Behavior CI guide](../guides/behavior-ci.md) and
  [Behavior CI Config](./behavior-ci-config.md).
- [Client API](./client-api.md) — the supported high-level clients the CLI is
  built on.
- [Errors & Troubleshooting](./errors.md) — error types and exit behavior.
