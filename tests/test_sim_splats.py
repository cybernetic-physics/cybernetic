from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner

from cybernetics.cli.__main__ import main_cli
from cybernetics.sim import (
    SimulationClient,
    SimulationError,
    detect_gaussian_splat_format,
    inspect_local_asset,
    package_local_asset,
)

BASE = "https://api.test"

_GAUSSIAN_PLY_PROPS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(9)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_test")
    monkeypatch.setenv("CYBERNETICS_BASE_URL", BASE)
    monkeypatch.delenv("CP_API_KEY", raising=False)
    monkeypatch.delenv("CP_API_BASE", raising=False)


def _write_gaussian_ply(path: Path, *, gaussians: int = 3) -> Path:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {gaussians}\n"
        + "".join(f"property float {name}\n" for name in _GAUSSIAN_PLY_PROPS)
        + "end_header\n"
    )
    values = [0.0] * (gaussians * len(_GAUSSIAN_PLY_PROPS))
    path.write_bytes(header.encode("ascii") + struct.pack(f"<{len(values)}f", *values))
    return path


def _write_mesh_ply(path: Path) -> Path:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + struct.pack("<3f", 0.0, 0.0, 0.0))
    return path


def test_detect_gaussian_splat_format_sniffs_ply_and_extensions(tmp_path) -> None:
    gaussian = _write_gaussian_ply(tmp_path / "site.ply")
    mesh = _write_mesh_ply(tmp_path / "mesh.ply")
    spz = tmp_path / "scene.spz"
    spz.write_bytes(b"\x1f\x8b\x00\x00")

    assert detect_gaussian_splat_format(gaussian) == "ply"
    assert detect_gaussian_splat_format(mesh) is None
    assert detect_gaussian_splat_format(spz) == "spz"
    assert detect_gaussian_splat_format(tmp_path / "missing.ply") is None


def test_gaussian_ply_is_classified_for_conversion_with_count(tmp_path) -> None:
    splat = _write_gaussian_ply(tmp_path / "construction_site.ply", gaussians=5)

    inspection = inspect_local_asset(splat)

    assert inspection.root_relpath == "construction_site.ply"
    assert inspection.asset_kind == "gaussian_splat_ply"
    assert inspection.compatibility_status == "needs_conversion"
    assert inspection.splat == {"format": "ply", "gaussian_count": 5}


def test_usd_root_still_wins_over_splat_in_bundle(tmp_path) -> None:
    (tmp_path / "scene.usda").write_text("#usda 1.0\n")
    _write_gaussian_ply(tmp_path / "capture.ply")

    inspection = inspect_local_asset(tmp_path)

    assert inspection.root_relpath == "scene.usda"
    assert inspection.asset_kind == "usd_stage"
    assert inspection.compatibility_status == "ready_to_render"


def test_package_gaussian_splat_records_manifest_metadata(tmp_path) -> None:
    splat = _write_gaussian_ply(tmp_path / "site.ply")

    package = package_local_asset(splat, output_path=tmp_path / "bundle.zip")

    assert package.asset_kind == "gaussian_splat_ply"
    assert package.compatibility_status == "needs_conversion"
    assert package.root_stage_relpath is None
    with zipfile.ZipFile(package.bundle_path) as zf:
        manifest = json.loads(zf.read("cybernetics_sim_asset_manifest.json"))
    assert manifest["splat"] == {"format": "ply", "gaussian_count": 3}
    assert manifest["asset_kind"] == "gaussian_splat_ply"


@respx.mock
def test_import_asset_uploads_splat_bundle_without_root_stage(tmp_path) -> None:
    splat = _write_gaussian_ply(tmp_path / "site.ply")

    respx.post(f"{BASE}/v1/envs").mock(
        return_value=httpx.Response(200, json={"id": "env_splat", "name": "sim-site"})
    )
    version_route = respx.post(f"{BASE}/v1/envs/env_splat/versions").mock(
        return_value=httpx.Response(
            200,
            json={
                "version": {"id": "ver_1", "envId": "env_splat"},
                "upload": {"url": "https://s3.test/post", "fields": {"key": "envs/x"}},
            },
        )
    )
    respx.post("https://s3.test/post").mock(return_value=httpx.Response(204))
    finalize_route = respx.post(f"{BASE}/v1/envs/env_splat/versions/ver_1/finalize").mock(
        return_value=httpx.Response(
            200, json={"id": "ver_1", "envId": "env_splat", "status": "ready"}
        )
    )

    with SimulationClient() as client:
        result = client.import_asset(splat)

    assert result.environment_ref is not None
    assert result.environment_ref.env_id == "env_splat"
    version_body = json.loads(version_route.calls.last.request.content)
    assert "rootStageRelpath" not in version_body
    finalize_body = json.loads(finalize_route.calls.last.request.content)
    assert "rootStageRelpath" not in finalize_body

    asset_ref = result.to_asset_ref()
    assert asset_ref.asset_kind == "gaussian_splat_ply"
    assert asset_ref.compatibility_status == "needs_conversion"


def test_launch_rejects_unconverted_splat_with_conversion_hint(tmp_path) -> None:
    splat = _write_gaussian_ply(tmp_path / "site.ply")

    with SimulationClient() as client:
        with pytest.raises(SimulationError, match="cybernetics splat upload"):
            client.launch(splat)


@respx.mock
def test_upload_splat_presigns_creates_job_and_returns_artifacts(tmp_path) -> None:
    splat = _write_gaussian_ply(tmp_path / "site.ply")

    presign_route = respx.post(f"{BASE}/v1/uploads/presign").mock(
        return_value=httpx.Response(
            200,
            json={
                "uploadId": "upload_1",
                "inputPrefix": "s3://bucket/inputs/upload_1/",
                "presignedUrls": [
                    {"url": "https://s3.test/post", "fields": {"key": "inputs/upload_1/site.ply"}}
                ],
            },
        )
    )
    respx.post("https://s3.test/post").mock(return_value=httpx.Response(204))
    jobs_route = respx.post(f"{BASE}/v1/jobs").mock(
        return_value=httpx.Response(200, json={"jobId": "job_1", "status": "queued"})
    )
    respx.get(f"{BASE}/v1/jobs/job_1").mock(
        return_value=httpx.Response(200, json={"jobId": "job_1", "status": "succeeded"})
    )
    respx.get(f"{BASE}/v1/jobs/job_1/artifacts").mock(
        return_value=httpx.Response(
            200,
            json={
                "artifacts": {"usdz": "artifacts/export.usdz"},
                "downloadUrls": {"usdz": "https://s3.test/export.usdz?sig"},
            },
        )
    )

    with SimulationClient() as client:
        uploaded = client.upload_splat(splat)
        job = client.create_splat_convert_job(uploaded["inputPrefix"])
        finished = client.wait_for_job(job["jobId"], timeout_seconds=5, poll_interval_seconds=0.01)
        artifacts = client.job_artifacts(finished["jobId"])

    assert uploaded == {
        "uploadId": "upload_1",
        "inputPrefix": "s3://bucket/inputs/upload_1/",
        "format": "ply",
        "name": "site.ply",
    }
    presign_body = json.loads(presign_route.calls.last.request.content)
    assert presign_body["inputKind"] == "splat"
    assert presign_body["files"][0]["name"] == "site.ply"
    job_body = json.loads(jobs_route.calls.last.request.content)
    assert job_body["inputKind"] == "splat"
    assert job_body["costGuardrails"]["maxRuntimeMinutes"] == 30
    assert artifacts["downloadUrls"]["usdz"].startswith("https://s3.test/export.usdz")


def test_upload_splat_rejects_non_splat_file(tmp_path) -> None:
    mesh = _write_mesh_ply(tmp_path / "mesh.ply")

    with SimulationClient() as client:
        with pytest.raises(SimulationError, match="not a recognized Gaussian splat"):
            client.upload_splat(mesh)


def test_wait_for_job_raises_on_failed_job() -> None:
    with respx.mock:
        respx.get(f"{BASE}/v1/jobs/job_9").mock(
            return_value=httpx.Response(
                200, json={"jobId": "job_9", "status": "failed", "errorMessage": "boom"}
            )
        )
        with SimulationClient() as client:
            with pytest.raises(SimulationError, match="boom"):
                client.wait_for_job("job_9", timeout_seconds=1, poll_interval_seconds=0.01)


def test_top_level_help_lists_splat() -> None:
    result = CliRunner().invoke(main_cli, ["--help"])
    assert result.exit_code == 0
    assert "splat" in result.output


def test_cli_splat_upload_no_convert_json(tmp_path) -> None:
    splat = _write_gaussian_ply(tmp_path / "site.ply")

    with respx.mock:
        respx.post(f"{BASE}/v1/uploads/presign").mock(
            return_value=httpx.Response(
                200,
                json={
                    "uploadId": "upload_2",
                    "inputPrefix": "s3://bucket/inputs/upload_2/",
                    "presignedUrls": [
                        {
                            "url": "https://s3.test/post",
                            "fields": {"key": "inputs/upload_2/site.ply"},
                        }
                    ],
                },
            )
        )
        respx.post("https://s3.test/post").mock(return_value=httpx.Response(204))
        result = CliRunner().invoke(
            main_cli,
            ["--format", "json", "splat", "upload", str(splat), "--no-convert"],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["uploadId"] == "upload_2"
    assert payload["format"] == "ply"
    assert "job_id" not in payload


def test_cli_splat_upload_refuses_convert_for_non_ply(tmp_path) -> None:
    spz = tmp_path / "scene.spz"
    spz.write_bytes(b"\x1f\x8b\x00\x00")

    with respx.mock:
        respx.post(f"{BASE}/v1/uploads/presign").mock(
            return_value=httpx.Response(
                200,
                json={
                    "uploadId": "upload_3",
                    "inputPrefix": "s3://bucket/inputs/upload_3/",
                    "presignedUrls": [
                        {
                            "url": "https://s3.test/post",
                            "fields": {"key": "inputs/upload_3/scene.spz"},
                        }
                    ],
                },
            )
        )
        respx.post("https://s3.test/post").mock(return_value=httpx.Response(204))
        result = CliRunner().invoke(main_cli, ["splat", "upload", str(spz)])

    assert result.exit_code != 0
    assert ".ply splats only" in str(result.exception)
