from __future__ import annotations

import hashlib
import json
from zipfile import ZipFile

import httpx
import pytest
import respx
from test_robotics_runtime_contracts import job_dict

from cybernetics import Client
from cybernetics.robotics import (
    AssetBundleRef,
    RobotEvalsClient,
    RobotEvalsError,
    RoboticsJobSpec,
)
from cybernetics.robotics.client import _inspect_zip

BASE = "https://api.test"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_test")
    monkeypatch.setenv("CYBERNETICS_BASE_URL", BASE)


@respx.mock
def test_submit_get_and_cancel_robotics_run() -> None:
    create = respx.post(f"{BASE}/v1/eval/runs").mock(
        return_value=httpx.Response(200, json={"id": "evrun_1", "status": "queued"})
    )
    get = respx.get(f"{BASE}/v1/eval/runs/evrun_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "evrun_1",
                "status": "running",
                "events": [{"event_type": "runtime_started"}],
                "artifacts": [{"id": 1, "role": "run_result"}],
            },
        )
    )
    cancel = respx.post(f"{BASE}/v1/eval/runs/evrun_1/cancel").mock(
        return_value=httpx.Response(204)
    )
    job = RoboticsJobSpec.from_dict(job_dict())

    with RobotEvalsClient() as client:
        assert client.submit(job, budget_usd_limit=3.5)["id"] == "evrun_1"
        assert client.get_run("evrun_1")["status"] == "running"
        assert client.list_events("evrun_1")[0]["event_type"] == "runtime_started"
        assert client.list_artifacts("evrun_1")[0]["role"] == "run_result"
        client.cancel("evrun_1")

    payload = json.loads(create.calls[0].request.content)
    assert payload["job"]["schema_version"] == "robotics-job/v1"
    assert "gpu_type" not in payload["job"]["resources"]
    assert payload["budgetUsdLimit"] == 3.5
    assert get.called and cancel.called


@respx.mock
def test_upload_asset_bundle_returns_immutable_ref(tmp_path) -> None:
    bundle = tmp_path / "r2r.zip"
    bundle.write_bytes(b"fixture-dataset-bundle")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    create = respx.post(f"{BASE}/v1/eval/asset-bundles").mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "raset_1",
                "uri": "cybernetics://artifacts/raset_1",
                "status": "uploading",
                "upload": {
                    "url": "https://s3.test/upload",
                    "fields": {"key": "robot-assets/raset_1/bundle"},
                },
            },
        )
    )
    upload = respx.post("https://s3.test/upload").mock(return_value=httpx.Response(204))
    finalize = respx.post(f"{BASE}/v1/eval/asset-bundles/raset_1/finalize").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "raset_1",
                "status": "ready",
                "contentSha256": digest,
            },
        )
    )

    with RobotEvalsClient() as client:
        ref = client.upload_asset_bundle(bundle, name="R2R validation")

    assert isinstance(ref, AssetBundleRef)
    assert ref.content_sha256 == digest
    assert ref.uri == "cybernetics://artifacts/raset_1"
    request = json.loads(create.calls[0].request.content)
    assert request["contentSha256"] == digest
    assert upload.called and finalize.called


@pytest.mark.parametrize("member", ["data\\scene.usd", "data//scene.usd", "data/./scene.usd"])
def test_asset_zip_inspection_rejects_nonportable_paths(tmp_path, member: str) -> None:
    bundle = tmp_path / "unsafe.zip"
    with ZipFile(bundle, "w") as archive:
        archive.writestr(member, "usd")

    with pytest.raises(RobotEvalsError, match="unsafe path"):
        _inspect_zip(bundle)


def test_asset_zip_inspection_rejects_empty_and_duplicate_archives(tmp_path) -> None:
    empty = tmp_path / "empty.zip"
    with ZipFile(empty, "w"):
        pass
    with pytest.raises(RobotEvalsError, match="no files"):
        _inspect_zip(empty)

    duplicate = tmp_path / "duplicate.zip"
    with ZipFile(duplicate, "w") as archive:
        archive.writestr("data/scene.usd", "first")
        archive.writestr("data/scene.usd", "second")
    with pytest.raises(RobotEvalsError, match="duplicate path"):
        _inspect_zip(duplicate)


def test_composition_client_exposes_robotics_namespace() -> None:
    with Client() as client:
        assert isinstance(client.robotics, RobotEvalsClient)
