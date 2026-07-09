from __future__ import annotations

import json

import httpx
import pytest
import respx

from cybernetics import Client
from cybernetics.robotics import ROBOT_TASK_SCHEMA_VERSION, RobotTaskSpec
from cybernetics.sim import SimulationAssetRef, SimulationClient, SimulationError

BASE = "https://api.test"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_test")
    monkeypatch.setenv("CYBERNETICS_BASE_URL", BASE)
    monkeypatch.delenv("CP_API_KEY", raising=False)
    monkeypatch.delenv("CP_API_BASE", raising=False)


def _robot_task_dict(asset_refs: list[dict]) -> dict:
    return {
        "schema_version": ROBOT_TASK_SCHEMA_VERSION,
        "task_id": "preview_asset_task",
        "robot_id": "fixture_bot",
        "simulator_backend": "fixture",
        "backend_config": {"image": "fixture"},
        "asset_refs": asset_refs,
        "joint_map": {"left_hip": "L_HIP", "right_hip": "R_HIP"},
        "actuator_model": {"kind": "position"},
        "observation_space": {"position": {"dtype": "float32", "shape": []}},
        "action_space": {"delta": {"dtype": "float32", "shape": []}},
        "sim_dt": 0.01,
        "control_dt": 0.02,
        "reset_spec": {"position": 0.0},
        "reward_spec": {"kind": "fixture_position"},
        "success_metric": {"metric": "position", "operator": ">=", "value": 1.0},
        "randomization": {},
        "termination": {"max_steps": 8},
        "eval_protocol": {"episodes": 1, "max_steps": 8},
    }


@respx.mock
def test_import_asset_uploads_bundle_and_finalizes_version(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    env_route = respx.post(f"{BASE}/v1/envs").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "env_demo",
                "name": "sim-demo",
                "workspaceId": "ws_1",
                "createdAt": "2026-07-09T00:00:00Z",
                "updatedAt": "2026-07-09T00:00:00Z",
            },
        )
    )
    version_route = respx.post(f"{BASE}/v1/envs/env_demo/versions").mock(
        return_value=httpx.Response(
            200,
            json={
                "version": {"id": "ver_demo", "envId": "env_demo", "status": "uploading"},
                "upload": {
                    "url": "https://s3.test/upload",
                    "fields": {"key": "envs/env_demo/versions/ver_demo/bundle.zip"},
                    "key": "envs/env_demo/versions/ver_demo/bundle.zip",
                    "expiresAt": "2026-07-09T01:00:00Z",
                    "putUrl": "https://s3.test/upload",
                },
            },
        )
    )
    upload_route = respx.post("https://s3.test/upload").mock(return_value=httpx.Response(204))
    finalize_route = respx.post(f"{BASE}/v1/envs/env_demo/versions/ver_demo/finalize").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "ver_demo",
                "envId": "env_demo",
                "status": "ready",
                "rootStageRelpath": "scene.usd",
                "contentSha256": "abc",
            },
        )
    )

    with SimulationClient() as client:
        result = client.import_asset(tmp_path, name="sim-demo")

    assert result.env_id == "env_demo"
    assert result.version_id == "ver_demo"
    assert result.environment_ref is not None
    assert result.environment_ref.uri == "cybernetics://envs/env_demo/versions/ver_demo"
    assert env_route.called
    assert version_route.called
    assert upload_route.called
    assert finalize_route.called

    version_request = json.loads(version_route.calls[0].request.content)
    assert version_request["rootStageRelpath"] == "scene.usd"
    finalize_request = json.loads(finalize_route.calls[0].request.content)
    assert finalize_request["rootStageRelpath"] == "scene.usd"

    asset_ref = result.to_asset_ref()
    assert asset_ref == SimulationAssetRef.from_dict(asset_ref.to_dict())
    assert asset_ref.to_dict() == {
        "schema_version": "simulation-asset-ref/v1",
        "ref_kind": "environment_version",
        "uri": "cybernetics://envs/env_demo/versions/ver_demo",
        "asset_kind": "usd_stage",
        "compatibility_status": "ready_to_render",
        "metadata": {
            "environment_name": "sim-demo",
            "file_count": 1,
            "source_name": tmp_path.name,
            "version_status": "ready",
        },
        "env_id": "env_demo",
        "version_id": "ver_demo",
        "root_stage_relpath": "scene.usd",
        "content_sha256": "abc",
    }
    task = RobotTaskSpec.from_dict(_robot_task_dict([asset_ref.to_dict()]))
    assert task.asset_refs == [asset_ref.to_dict()]


def test_import_asset_stops_before_upload_for_robot_description(tmp_path) -> None:
    robot = tmp_path / "robot.urdf"
    robot.write_text("<robot name='demo'></robot>\n")

    with SimulationClient() as client:
        result = client.import_asset(robot)

    assert result.environment_ref is None
    assert result.package is not None
    assert result.package.compatibility_status == "needs_conversion"


def test_import_asset_emits_local_bundle_asset_ref_for_kept_robot_description(tmp_path) -> None:
    robot = tmp_path / "robot.urdf"
    robot.write_text("<robot name='demo'></robot>\n")

    with SimulationClient() as client:
        result = client.import_asset(robot, bundle_path=tmp_path / "robot.bundle.zip")

    asset_ref = result.to_asset_ref().to_dict()
    assert asset_ref["schema_version"] == "simulation-asset-ref/v1"
    assert asset_ref["ref_kind"] == "local_bundle"
    assert asset_ref["uri"].startswith("file://")
    assert asset_ref["asset_kind"] == "urdf_robot"
    assert asset_ref["compatibility_status"] == "needs_conversion"
    assert "root_stage_relpath" not in asset_ref
    assert len(asset_ref["content_sha256"]) == 64
    task = RobotTaskSpec.from_dict(_robot_task_dict([asset_ref]))
    assert task.asset_refs == [asset_ref]


def test_launch_rejects_conversion_needed_assets(tmp_path) -> None:
    robot = tmp_path / "robot.urdf"
    robot.write_text("<robot name='demo'></robot>\n")

    with SimulationClient() as client:
        with pytest.raises(SimulationError, match="needs_conversion"):
            client.launch(robot)


@respx.mock
def test_launch_environment_ref_without_version_omits_null_base_version() -> None:
    session_route = respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess_demo", "status": "queued"})
    )

    with SimulationClient() as client:
        result = client.launch("cybernetics://envs/env_demo")

    assert result.session_id == "sess_demo"
    request_body = json.loads(session_route.calls[0].request.content)
    assert request_body["envId"] == "env_demo"
    assert "baseVersionId" not in request_body


@respx.mock
def test_wait_for_session_accepts_public_session_readiness_shape() -> None:
    respx.get(f"{BASE}/v1/sessions/sess_demo").mock(
        return_value=httpx.Response(
            200,
            json={
                "sessionId": "sess_demo",
                "status": "running",
                "runtimeStatus": "running",
                "access": {"viewerUrl": "https://viewer.test/sess_demo"},
            },
        )
    )

    with SimulationClient() as client:
        session = client.wait_for_session(
            "sess_demo",
            timeout_seconds=0.1,
            poll_interval_seconds=0.01,
        )

    assert session["sessionId"] == "sess_demo"


def test_render_public_flag_is_explicitly_not_mvp(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    with SimulationClient() as client:
        with pytest.raises(SimulationError, match="Public /sim artifact pages"):
            client.render(tmp_path, public=True)


@respx.mock
def test_render_wait_keeps_session_running_by_default(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    respx.post(f"{BASE}/v1/envs").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "env_demo",
                "name": "sim-demo",
                "workspaceId": "ws_1",
                "createdAt": "2026-07-09T00:00:00Z",
                "updatedAt": "2026-07-09T00:00:00Z",
            },
        )
    )
    respx.post(f"{BASE}/v1/envs/env_demo/versions").mock(
        return_value=httpx.Response(
            200,
            json={
                "version": {"id": "ver_demo", "envId": "env_demo", "status": "uploading"},
                "upload": {
                    "url": "https://s3.test/upload",
                    "fields": {"key": "envs/env_demo/versions/ver_demo/bundle.zip"},
                    "key": "envs/env_demo/versions/ver_demo/bundle.zip",
                    "expiresAt": "2026-07-09T01:00:00Z",
                    "putUrl": "https://s3.test/upload",
                },
            },
        )
    )
    respx.post("https://s3.test/upload").mock(return_value=httpx.Response(204))
    respx.post(f"{BASE}/v1/envs/env_demo/versions/ver_demo/finalize").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "ver_demo",
                "envId": "env_demo",
                "status": "ready",
                "rootStageRelpath": "scene.usd",
            },
        )
    )
    respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess_demo", "status": "starting"})
    )
    respx.get(f"{BASE}/v1/sessions/sess_demo").mock(
        return_value=httpx.Response(
            200,
            json={
                "sessionId": "sess_demo",
                "status": "running",
                "bridgeStatus": {"isaac_extension_ready": True},
                "access": {"viewerUrl": "https://viewer.test/sess_demo"},
            },
        )
    )
    respx.post(f"{BASE}/v1/sessions/sess_demo/cua-grant").mock(
        return_value=httpx.Response(
            200,
            json={
                "neko_url": "wss://neko.test/token",
                "neko_username": "neko",
                "neko_password": "secret",
            },
        )
    )
    respx.post("https://neko.test/token/api/login").mock(
        return_value=httpx.Response(200, json={"token": "login_demo"})
    )
    with SimulationClient() as client:
        result = client.render(tmp_path, name="sim-demo", wait=True)

    assert result.status == "preview_ready"
    assert result.preview_url == "https://neko.test/token/api/shot.jpg?quality=90&token=login_demo"
    assert result.launch_url == "https://viewer.test/sess_demo"
    assert all(call.request.url.path != "/v1/sessions/sess_demo/stop" for call in respx.calls)


@respx.mock
def test_render_stops_session_when_preview_download_fails(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    respx.post(f"{BASE}/v1/envs").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "env_demo",
                "name": "sim-demo",
                "workspaceId": "ws_1",
                "createdAt": "2026-07-09T00:00:00Z",
                "updatedAt": "2026-07-09T00:00:00Z",
            },
        )
    )
    respx.post(f"{BASE}/v1/envs/env_demo/versions").mock(
        return_value=httpx.Response(
            200,
            json={
                "version": {"id": "ver_demo", "envId": "env_demo", "status": "uploading"},
                "upload": {
                    "url": "https://s3.test/upload",
                    "fields": {"key": "envs/env_demo/versions/ver_demo/bundle.zip"},
                    "key": "envs/env_demo/versions/ver_demo/bundle.zip",
                    "expiresAt": "2026-07-09T01:00:00Z",
                    "putUrl": "https://s3.test/upload",
                },
            },
        )
    )
    respx.post("https://s3.test/upload").mock(return_value=httpx.Response(204))
    respx.post(f"{BASE}/v1/envs/env_demo/versions/ver_demo/finalize").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "ver_demo",
                "envId": "env_demo",
                "status": "ready",
                "rootStageRelpath": "scene.usd",
            },
        )
    )
    respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess_demo", "status": "starting"})
    )
    respx.get(f"{BASE}/v1/sessions/sess_demo").mock(
        return_value=httpx.Response(
            200,
            json={
                "sessionId": "sess_demo",
                "status": "running",
                "runtimeStatus": "running",
                "access": {"viewerUrl": "https://viewer.test/sess_demo"},
            },
        )
    )
    respx.post(f"{BASE}/v1/sessions/sess_demo/cua-grant").mock(
        return_value=httpx.Response(
            200,
            json={
                "neko_url": "wss://neko.test/token",
                "neko_username": "neko",
                "neko_password": "secret",
            },
        )
    )
    respx.post("https://neko.test/token/api/login").mock(
        return_value=httpx.Response(200, json={"token": "login_demo"})
    )
    respx.get("https://neko.test/token/api/shot.jpg").mock(
        return_value=httpx.Response(401, json={"message": "unauthorized"})
    )
    stop_route = respx.post(f"{BASE}/v1/sessions/sess_demo/stop").mock(
        return_value=httpx.Response(204)
    )

    with SimulationClient() as client:
        with pytest.raises(SimulationError, match="GET CUA preview image"):
            client.render(
                tmp_path,
                name="sim-demo",
                wait=True,
                keep_session=False,
                out=tmp_path / "preview.jpg",
            )

    assert stop_route.called


def test_top_level_client_exposes_sim_namespace() -> None:
    client = Client()
    try:
        assert isinstance(client.sim, SimulationClient)
    finally:
        client.sim.close()
