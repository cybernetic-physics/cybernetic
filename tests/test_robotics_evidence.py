from __future__ import annotations

import json

from test_robotics_runtime_contracts import job_dict

from cybernetics.behavior_ci.artifacts import validate_bundle
from cybernetics.robotics import (
    EPISODE_MANIFEST_SCHEMA_VERSION,
    EpisodeManifest,
    RoboticsJobSpec,
    write_robotics_behavior_bundle,
)


def _episode(job: RoboticsJobSpec, index: int, *, success: float) -> EpisodeManifest:
    return EpisodeManifest.from_dict(
        {
            "schema_version": EPISODE_MANIFEST_SCHEMA_VERSION,
            "run_id": "evrun_fixture",
            "episode_id": f"episode-{index:04d}",
            "seed": job.rollout.resolved_seeds()[index],
            "status": "succeeded" if success else "failed",
            "job_hash": job.job_hash(),
            "simulator_package_hash": job.simulator.package_hash(),
            "task_package_hash": job.task.package_hash(),
            "policy_deployment_hash": job.policy.deployment_hash(),
            "policy_deployment_id": job.policy.deployment_id,
            "policy_revision": job.policy.revision,
            "simulator_image": job.simulator.runtime_image,
            "step_count": index + 1,
            "metrics": {"success": success, "spl": success * 0.75},
            "artifacts": [],
            "started_at": "2026-07-10T00:00:00Z",
            "finished_at": "2026-07-10T00:00:01Z",
            "requested_placement": job.placement.to_dict(),
            "actual_placement": {"topology": "in_process_fixture"},
            "error_code": None if success else "NAVIGATION_FAILED",
            "error_message": None if success else "goal was not reached",
        }
    )


def test_external_runtime_writes_valid_behavior_ci_bundle(tmp_path) -> None:
    job = RoboticsJobSpec.from_dict(job_dict())
    episodes = [_episode(job, 0, success=1.0), _episode(job, 1, success=1.0)]

    result = write_robotics_behavior_bundle(
        job,
        episodes,
        tmp_path,
        allow_fixture_replay=True,
    )

    assert result.passed is True
    assert result.metrics["success"] == 1.0
    assert result.honesty.simulator_adapter == "robotics-runtime"
    assert result.honesty.pins_verified is True
    assert validate_bundle(tmp_path, result) == []
    assert (tmp_path / "robotics-job.json").exists()
    assert (tmp_path / "episodes/episode-0000.json").exists()
    normalized = json.loads((tmp_path / "manifest.normalized.json").read_text())
    assert normalized["job_hash"] == job.job_hash()


def test_failed_episode_produces_failed_behavior_ci_result(tmp_path) -> None:
    job = RoboticsJobSpec.from_dict(job_dict())
    episodes = [_episode(job, 0, success=1.0), _episode(job, 1, success=0.0)]

    result = write_robotics_behavior_bundle(
        job,
        episodes,
        tmp_path,
        allow_fixture_replay=True,
    )

    assert result.passed is False
    assert result.summary["failed_runs"] == 1
    assert result.failures[0]["code"] == "NAVIGATION_FAILED"
    assert (tmp_path / "replays/replay-failed.mp4").exists()


def test_replay_paths_and_boolean_success_metrics_are_supported(tmp_path) -> None:
    job = RoboticsJobSpec.from_dict(job_dict())
    episodes = [_episode(job, 0, success=True), _episode(job, 1, success=True)]
    replay = tmp_path / "source.mp4"
    replay.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 32)

    result = write_robotics_behavior_bundle(
        job,
        episodes,
        tmp_path / "bundle",
        replay_videos={episode.episode_id: replay for episode in episodes},
    )

    assert result.passed is True
    assert result.metrics["success"] == 1.0
