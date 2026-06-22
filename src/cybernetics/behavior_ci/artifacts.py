"""Write and validate the ``behavior-ci/v1`` artifact bundle.

Bundle layout (stable contract):

    result.json              top-level verdict + provenance
    metrics.json             aggregate + per-run metrics/checks
    comment.md               sticky PR comment (markdown)
    report/index.html        self-contained static report
    replays/replay-*.mp4     replay clips (present per produced clips)
    manifest.normalized.json the effective, resolved run inputs
    provenance.json          honesty provenance, standalone for auditing

``replay-passed.mp4`` is written whenever a trial passed; ``replay-failed.mp4``
only when the run failed — an honest bundle never ships a "failed" clip for a
green run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from . import renderers
from .schemas import (
    METRICS_SCHEMA_VERSION,
    BehaviorCiResult,
    EvalSpec,
)
from .simulators.base import ReplayResult, looks_like_mp4

BASE_REQUIRED = (
    "result.json",
    "metrics.json",
    "comment.md",
    "report/index.html",
    "manifest.normalized.json",
    "provenance.json",
)


def write_bundle(
    result: BehaviorCiResult,
    replays: List[ReplayResult],
    eval_spec: EvalSpec,
    normalized_manifest: Dict[str, Any],
    bundle_dir: Path,
    artifact_url: str = "",
) -> None:
    bundle_dir = Path(bundle_dir)
    (bundle_dir / "report").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "replays").mkdir(parents=True, exist_ok=True)

    # 1) replays first, so result.artifacts is populated before rendering.
    for r in replays:
        rel = f"replays/{r.name}.mp4"
        (bundle_dir / rel).write_bytes(r.data)
        result.artifacts[r.name.replace("-", "_")] = rel
    if "replay_failed" in result.artifacts:
        result.artifacts["replay_video"] = result.artifacts["replay_failed"]
    elif "replay_passed" in result.artifacts:
        result.artifacts["replay_video"] = result.artifacts["replay_passed"]

    # 2) link the rest.
    result.artifacts.update(
        {
            "result_json": "result.json",
            "metrics_json": "metrics.json",
            "report": "report/index.html",
        }
    )

    # 3) render + write.
    _write_json(bundle_dir / "result.json", result.to_dict())
    _write_json(bundle_dir / "metrics.json", _metrics_doc(result, eval_spec))
    (bundle_dir / "comment.md").write_text(renderers.render_comment(result, artifact_url))
    (bundle_dir / "report" / "index.html").write_text(renderers.render_report_html(result))
    _write_json(bundle_dir / "manifest.normalized.json", normalized_manifest)
    _write_json(bundle_dir / "provenance.json", result.honesty.to_dict())


def validate_bundle(bundle_dir: Path, result: BehaviorCiResult | None = None) -> List[str]:
    """Return a list of contract violations (empty list = valid bundle)."""

    bundle_dir = Path(bundle_dir)
    problems: List[str] = []

    for rel in BASE_REQUIRED:
        p = bundle_dir / rel
        if not p.exists():
            problems.append(f"missing required artifact: {rel}")
        elif p.stat().st_size == 0:
            problems.append(f"empty artifact: {rel}")

    replays = sorted((bundle_dir / "replays").glob("*.mp4"))
    if not replays:
        problems.append("no replay clips in replays/")
    for clip in replays:
        if not looks_like_mp4(clip.read_bytes()):
            problems.append(f"replay is not a valid MP4: replays/{clip.name}")

    if result is not None:
        if result.status == "failed" and not (bundle_dir / "replays/replay-failed.mp4").exists():
            problems.append("failed run is missing replays/replay-failed.mp4")
        if (
            result.summary.get("passed_runs", 0)
            and not (bundle_dir / "replays/replay-passed.mp4").exists()
        ):
            problems.append("passing trials present but replays/replay-passed.mp4 is missing")

    return problems


def _metrics_doc(result: BehaviorCiResult, spec: EvalSpec) -> Dict[str, Any]:
    check_pass_counts = {
        name: sum(1 for t in result.trials if t.checks[name].passed) for name in spec.checks
    }
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "aggregate": result.metrics,
        "check_pass_counts": check_pass_counts,
        "runs": [t.to_dict() for t in result.trials],
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
