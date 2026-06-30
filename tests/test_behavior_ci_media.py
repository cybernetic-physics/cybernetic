"""mp4 -> inline GIF transcode + replay-token substitution in the PR comment."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cybernetics.behavior_ci import media, renderers
from cybernetics.behavior_ci.artifacts import validate_bundle
from cybernetics.behavior_ci.schemas import BehaviorCiResult, HonestyProvenance


def _sample_mp4(tmp_path: Path) -> bytes:
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = tmp_path / "in.mp4"
    subprocess.run(
        [
            ff,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x90:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


def test_mp4_to_gif_produces_valid_capped_gif(tmp_path: Path):
    gif = media.mp4_to_gif(_sample_mp4(tmp_path))
    assert gif is not None
    assert media.looks_like_gif(gif)
    assert len(gif) <= media.DEFAULT_SIZE_CAP


def test_mp4_to_gif_handles_empty_and_garbage():
    assert media.mp4_to_gif(b"") is None
    with pytest.raises(Exception):
        media.mp4_to_gif(b"not an mp4 at all")  # ffmpeg errors -> caller swallows


def test_looks_like_gif():
    assert media.looks_like_gif(b"GIF89a....")
    assert not media.looks_like_gif(b"\x00\x00\x00")


def _result():
    h = HonestyProvenance(
        simulator_adapter="isaac-session",
        replay_source="isaac-sim-session-video",
        policy_backend="scripted-vla-shim",
        policy_backend_real_vla=False,
        production_eval_path_used=True,
        scene_env="x",
        camera="/World/Cameras/Cam",
    )
    return BehaviorCiResult(
        status="passed",
        behavior="g1_weld_approach",
        robot="G1",
        world="w",
        scene_env="x",
        camera="/World/Cameras/Cam",
        policy="v21.pt",
        policy_id="v21",
        policy_backend="scripted-vla-shim",
        commit="abc1234",
        summary={"passed_runs": 16, "total_runs": 16},
        checks={"target_reach": True},
        metrics={"mean_torch_tip_error_cm": 0.1},
        failures=[],
        trials=[],
        honesty=h,
    )


def test_comment_leaves_replay_token_without_url():
    body = renderers.render_comment(_result())
    assert renderers.REPLAY_TOKEN in body


def test_comment_substitutes_inline_gif_with_url():
    url = "https://raw.githubusercontent.com/o/r/ci-media/behavior-ci/pr-9/abc.gif"
    body = renderers.render_comment(_result(), replay_gif_url=url)
    assert f"![Real Isaac G1 weld-approach replay — commit abc1234]({url})" in body
    assert "settled pass/fail camera" in body
    # the bare token line must be gone once substituted
    assert renderers.REPLAY_TOKEN not in body.replace(url, "")


def test_validate_bundle_rejects_corrupt_gif(tmp_path: Path):
    (tmp_path / "replays").mkdir()
    (tmp_path / "replays" / "replay-passed.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (tmp_path / "replays" / "replay-passed.gif").write_bytes(b"NOT A GIF")
    problems = validate_bundle(tmp_path)
    assert any("not a valid GIF" in p for p in problems)
