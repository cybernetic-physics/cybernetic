from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from cybernetics import Client, ReplayClient, ReplayObservation, ReplayQuery
from cybernetics.replay import (
    AgentReplayOptions,
    ReplayChunk,
    ReplayError,
    ReplayEvent,
    ReplayImage,
    ReplayIntegrityError,
    ReplayRecording,
    ReplaySchemaError,
    ReplaySummary,
)
from cybernetics.replay.models import redact_replay_text

BASE = "https://api.test"
SESSION_ID = "sess_demo"
JPEG = b"\xff\xd8\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\nfixture"
POSTGRES_BIGINT_MAX = 9_223_372_036_854_775_807


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_test")
    monkeypatch.setenv("CYBERNETICS_BASE_URL", BASE)


def _recording() -> dict[str, Any]:
    return {
        "recordingId": "rec_demo",
        "sessionId": SESSION_ID,
        "workspaceId": "ws_demo",
        "writerId": "writer_demo",
        "status": "ready",
        "formatVersion": 1,
        "captureConfig": {},
        "channels": ["camera/main", "world/pose"],
        "eventCount": 2,
        "chunkCount": 2,
        "sizeBytes": 100,
        "startTimeNs": "100",
        "endTimeNs": "500",
    }


def _summary() -> dict[str, Any]:
    return {
        "sessionId": SESSION_ID,
        "formatVersion": 1,
        "recordings": [_recording()],
        "controlEvents": {"count": 0, "firstTimeNs": None, "lastTimeNs": None},
    }


def _chunk(
    chunk_id: str,
    *,
    sequence: int,
    channel: str,
    time_ns: int,
    content: bytes,
) -> dict[str, Any]:
    return {
        "chunkId": chunk_id,
        "recordingId": "rec_demo",
        "sequence": sequence,
        "objectKey": f"sessions/{SESSION_ID}/{chunk_id}.ndjson.gz",
        "startTimeNs": str(time_ns),
        "endTimeNs": str(time_ns),
        "eventCount": 1,
        "sizeBytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "contentMd5": "fixture",
        "channels": [channel],
        "contentType": "application/vnd.cybernetic.timeline+ndjson",
        "compression": "gzip",
    }


def _observation(*, include_image: bool = True) -> dict[str, Any]:
    encoded = base64.b64encode(JPEG).decode("ascii")
    return {
        "schema": "cybernetic.replay-observation/v1",
        "sessionId": SESSION_ID,
        "window": {
            "startTimeNs": "999999970000000009",
            "endTimeNs": "1000000000000000009",
            "semantics": "event-window",
            "completeState": False,
            "order": "timeNs,source,eventId",
        },
        "archive": {
            "formatVersion": 1,
            "channels": ["camera/main", "world/pose"],
            "recordings": [_recording()],
        },
        "events": [
            {
                "eventId": "evt_camera",
                "channel": "camera/main",
                "type": "camera.frame",
                "source": "isaac-sim",
                "timeNs": "1000000000000000005",
                "simTimeNs": "5000000001",
                "semantics": "sample",
                "mediaIds": ["img_main"] if include_image else [],
                "payload": {
                    "cameraPath": "/OmniverseKit_Persp",
                    "dataBase64": encoded,
                    "nested": {
                        "access_token": "should-never-appear",
                        "clientSecret": "also-never-appear",
                        "AWS_SECRET_ACCESS_KEY": "mapping-aws-secret",
                        "AWS_ACCESS_KEY_ID": "mapping-access-key-id",
                    },
                    "image_url": f"prefix data:image/jpeg;base64,{encoded}",
                    "svg_url": "data:image/svg+xml,%3Csvg%3Eunsafe%3C/svg%3E",
                    "message": (
                        'authorization=raw-secret {"token":"quoted-secret"} '
                        "--api-key cli-secret\n"
                        "OPENAI_API_KEY=env-secret\n"
                        "AWS_SECRET_ACCESS_KEY='assignment-aws-secret'\n"
                        "Authorization: Basic dXNlcjpwYXNz\n"
                        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAFIXTURE, "
                        "SignedHeaders=host;x-amz-date, Signature=aws-signature\n"
                        "Cookie: sid=cookie-secret\n"
                        "Set-Cookie: refresh=set-cookie-secret"
                    ),
                },
            }
        ],
        "images": (
            [
                {
                    "imageId": "img_main",
                    "eventId": "evt_camera",
                    "channel": "camera/main",
                    "source": "isaac-sim",
                    "timeNs": "1000000000000000005",
                    "simTimeNs": "5000000001",
                    "mimeType": "image/jpeg",
                    "width": 1,
                    "height": 1,
                    "sizeBytes": len(JPEG),
                    "sha256": hashlib.sha256(JPEG).hexdigest(),
                    "dataBase64": encoded,
                }
            ]
            if include_image
            else []
        ),
        "quality": {
            "stateBasis": "window_delta",
            "unitsDeclared": False,
            "framesDeclared": False,
            "truncated": False,
            "truncationReasons": [],
            "scannedChunks": 1,
            "scannedCompressedBytes": 4,
            "scannedDecodedBytes": 8,
            "matchedEvents": 1,
            "matchedEventsExact": True,
            "returnedEvents": 1,
            "matchedImages": 1 if include_image else 0,
            "returnedImages": 1 if include_image else 0,
            "warnings": ["Units and frames are source-defined."],
        },
    }


@respx.mock
def test_raw_event_pagination_channel_filter_and_camera_binary_safety() -> None:
    camera_event = {
        "eventId": "evt_camera",
        "channel": "camera/main",
        "type": "camera.frame",
        "source": "isaac-sim",
        "timeNs": "200",
        "payload": {"dataBase64": base64.b64encode(JPEG).decode("ascii")},
    }
    pose_event = {
        "eventId": "evt_pose",
        "channel": "world/pose",
        "type": "pose",
        "source": "isaac-sim",
        "timeNs": "300",
        "payload": {"position": [0, 0, 0]},
    }
    camera_bytes = gzip.compress((json.dumps(camera_event) + "\n").encode(), mtime=0)
    pose_bytes = gzip.compress((json.dumps(pose_event) + "\n").encode(), mtime=0)

    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=_summary())
    )
    chunk_route = respx.get(
        f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks"
    ).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "items": [
                        _chunk(
                            "chunk_camera",
                            sequence=0,
                            channel="camera/main",
                            time_ns=200,
                            content=camera_bytes,
                        )
                    ],
                    "nextCursor": "cursor_2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "items": [
                        _chunk(
                            "chunk_pose",
                            sequence=1,
                            channel="world/pose",
                            time_ns=300,
                            content=pose_bytes,
                        )
                    ],
                    "nextCursor": None,
                },
            ),
        ]
    )
    respx.get(
        f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks/chunk_camera/content"
    ).mock(return_value=httpx.Response(200, content=camera_bytes))

    with ReplayClient() as client:
        events = list(
            client.iter_events(
                SESSION_ID,
                ReplayQuery(
                    channels=("camera/*",),
                    start_time_ns=100,
                    end_time_ns=500,
                    include_control_events=False,
                ),
            )
        )

    assert [event.event_id for event in events] == ["evt_camera"]
    assert events[0].payload["dataBase64"]
    assert "dataBase64" not in json.dumps(events[0].to_dict())
    assert len(chunk_route.calls) == 2
    assert chunk_route.calls[0].request.url.params["startTimeNs"] == "100"
    assert chunk_route.calls[0].request.url.params["endTimeNs"] == "500"
    assert chunk_route.calls[1].request.url.params["cursor"] == "cursor_2"


@respx.mock
def test_observation_passes_channel_globs_and_preserves_exact_nanoseconds() -> None:
    summary = _summary()
    summary["recordings"][0]["startTimeNs"] = "999999970000000009"
    summary["recordings"][0]["endTimeNs"] = "1000000000000000009"
    summary_route = respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=summary)
    )
    observation_route = respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/observations").mock(
        return_value=httpx.Response(200, json=_observation())
    )

    with ReplayClient() as client:
        observation = client.get_observation(
            SESSION_ID,
            ReplayQuery(channels=("camera/*",)),
            image_data=True,
        )

    assert observation.start_time_ns == 999_999_970_000_000_009
    assert observation.events[0].time_ns == 1_000_000_000_000_000_005
    assert observation.events[0].semantics == "sample"
    assert observation.events[0].media_ids == ("img_main",)
    assert observation_route.calls[0].request.url.params["channels"] == "camera/*"
    assert summary_route.called is False
    serialized = json.dumps(observation.to_dict())
    assert '"1000000000000000005"' in serialized
    assert "dataBase64" not in serialized
    assert "should-never-appear" not in serialized
    assert "also-never-appear" not in serialized
    assert "mapping-aws-secret" not in serialized
    assert "mapping-access-key-id" not in serialized
    assert "raw-secret" not in serialized
    assert "quoted-secret" not in serialized
    assert "cli-secret" not in serialized
    assert "env-secret" not in serialized
    assert "assignment-aws-secret" not in serialized
    assert "dXNlcjpwYXNz" not in serialized
    assert "AKIAFIXTURE" not in serialized
    assert "aws-signature" not in serialized
    assert "cookie-secret" not in serialized
    assert "set-cookie-secret" not in serialized
    assert "data:image/jpeg;base64" not in serialized
    assert "data:image/svg+xml" not in serialized
    assert "untrusted_replay_data" in serialized
    assert "cybernetic-replay-agent-observation/v1" in serialized
    assert "delta window" in " ".join(observation.warnings)


def test_provider_adapters_label_frames_before_images_and_end_with_context() -> None:
    observation = ReplayObservation.from_api_dict(_observation())

    openai_content = observation.to_openai_content(detail="low")
    assert [block["type"] for block in openai_content] == [
        "input_text",
        "input_image",
        "input_text",
    ]
    assert "time_ns=1000000000000000005" in openai_content[0]["text"]
    assert "offset_ns=29999999996" in openai_content[0]["text"]
    assert "sim_time_ns=5000000001" in openai_content[0]["text"]
    assert "channel=camera/main" in openai_content[0]["text"]
    assert "event=evt_camera" in openai_content[0]["text"]
    assert openai_content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "dataBase64" not in openai_content[-1]["text"]
    assert "untrusted_replay_data" in openai_content[-1]["text"]

    anthropic_content = observation.to_anthropic_content()
    assert [block["type"] for block in anthropic_content] == ["text", "image", "text"]
    assert anthropic_content[1]["source"]["type"] == "base64"
    assert anthropic_content[1]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(anthropic_content[1]["source"]["data"]) == JPEG
    assert "dataBase64" not in anthropic_content[-1]["text"]
    assert "untrusted_replay_data" in anthropic_content[-1]["text"]


def test_agent_bundle_is_deterministic_text_safe_and_camera_linked(tmp_path: Path) -> None:
    observation = ReplayObservation.from_api_dict(_observation())

    class FakeClient:
        def get_observation(self, *args: Any, **kwargs: Any) -> ReplayObservation:
            assert kwargs["image_data"] is True
            return observation

    client = ReplayClient(api_key="cp_live_test", http_client=object())
    client.get_observation = FakeClient().get_observation  # type: ignore[method-assign]
    first = client.export_agent_bundle(SESSION_ID, tmp_path / "first")
    second = client.export_agent_bundle(SESSION_ID, tmp_path / "second")
    overwritten = client.export_agent_bundle(
        SESSION_ID,
        tmp_path / "first",
        options=AgentReplayOptions(overwrite=True),
    )

    assert len(first.frame_paths) == 1
    assert first.frame_paths[0].read_bytes() == JPEG
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert overwritten.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.events_path.read_bytes() == second.events_path.read_bytes()
    all_text = "\n".join(
        path.read_text()
        for path in (
            first.manifest_path,
            first.context_path,
            first.events_path,
            first.observations_path,
        )
    )
    assert "dataBase64" not in all_text
    assert base64.b64encode(JPEG).decode("ascii") not in all_text
    assert "should-never-appear" not in all_text
    assert "also-never-appear" not in all_text
    assert "mapping-aws-secret" not in all_text
    assert "mapping-access-key-id" not in all_text
    assert "raw-secret" not in all_text
    assert "quoted-secret" not in all_text
    assert "cli-secret" not in all_text
    assert "env-secret" not in all_text
    assert "assignment-aws-secret" not in all_text
    assert "dXNlcjpwYXNz" not in all_text
    assert "AKIAFIXTURE" not in all_text
    assert "aws-signature" not in all_text
    assert "cookie-secret" not in all_text
    assert "set-cookie-secret" not in all_text
    assert "data:image/jpeg;base64" not in all_text
    assert "frames/0001-1000000000000000005-" in all_text
    assert '"media_ids":["img_main"]' in first.events_path.read_text()
    assert '"offset_ns":"29999999996"' in first.observations_path.read_text()
    assert "offset_ns=`29999999996`" in first.context_path.read_text()
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["state_complete"] is False
    assert manifest["counts"] == {"events": 1, "frame_files": 1, "images": 1}
    assert manifest["query"]["latest_seconds"] == 30.0


def test_agent_bundle_rejects_unmanaged_and_symlink_destinations(tmp_path: Path) -> None:
    observation = ReplayObservation.from_api_dict(_observation(include_image=False))

    class FakeClient:
        def get_observation(self, *args: Any, **kwargs: Any) -> ReplayObservation:
            return observation

    client = ReplayClient(api_key="cp_live_test", http_client=object())
    client.get_observation = FakeClient().get_observation  # type: ignore[method-assign]
    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    (unmanaged / "keep.txt").write_text("mine")
    with pytest.raises(ReplaySchemaError, match="unmanaged"):
        client.export_agent_bundle(
            SESSION_ID,
            unmanaged,
            options=AgentReplayOptions(overwrite=True),
        )

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReplaySchemaError, match="symbolic link"):
        client.export_agent_bundle(SESSION_ID, link)


def test_agent_bundle_exports_png_with_truthful_inventory(tmp_path: Path) -> None:
    payload = _observation()
    encoded = base64.b64encode(PNG).decode("ascii")
    payload["images"][0].update(
        {
            "mimeType": "image/png",
            "sizeBytes": len(PNG),
            "sha256": hashlib.sha256(PNG).hexdigest(),
            "dataBase64": encoded,
        }
    )
    observation = ReplayObservation.from_api_dict(payload)

    class FakeClient:
        def get_observation(self, *args: Any, **kwargs: Any) -> ReplayObservation:
            return observation

    client = ReplayClient(api_key="cp_live_test", http_client=object())
    client.get_observation = FakeClient().get_observation  # type: ignore[method-assign]
    bundle = client.export_agent_bundle(SESSION_ID, tmp_path / "png-bundle")

    assert bundle.frame_paths[0].suffix == ".png"
    assert bundle.frame_paths[0].read_bytes() == PNG
    manifest = json.loads(bundle.manifest_path.read_text())
    frame = next(item for item in manifest["files"] if item["path"].endswith(".png"))
    assert frame["media_type"] == "image/png"


def test_failed_overwrite_preserves_prior_agent_bundle(tmp_path: Path) -> None:
    observation = ReplayObservation.from_api_dict(_observation())

    class FakeClient:
        def get_observation(self, *args: Any, **kwargs: Any) -> ReplayObservation:
            return observation

    client = ReplayClient(api_key="cp_live_test", http_client=object())
    client.get_observation = FakeClient().get_observation  # type: ignore[method-assign]
    destination = tmp_path / "bundle"
    first = client.export_agent_bundle(SESSION_ID, destination)
    original_manifest = first.manifest_path.read_bytes()

    def fail(*args: Any, **kwargs: Any) -> ReplayObservation:
        raise ReplayError("fixture network failure")

    client.get_observation = fail  # type: ignore[method-assign]
    with pytest.raises(ReplayError, match="fixture network failure"):
        client.export_agent_bundle(
            SESSION_ID,
            destination,
            options=AgentReplayOptions(overwrite=True),
        )
    assert first.manifest_path.read_bytes() == original_manifest


def test_observation_rejects_raw_only_filters_before_network() -> None:
    client = ReplayClient(api_key="cp_live_test", http_client=object())
    with pytest.raises(ReplaySchemaError, match="raw iter_events"):
        client.get_observation(SESSION_ID, ReplayQuery(recording_ids=("rec_demo",)))
    with pytest.raises(ReplaySchemaError, match="raw iter_events"):
        client.get_observation(SESSION_ID, ReplayQuery(include_control_events=False))


def test_observation_schema_exact_ns_and_complete_state_are_strict() -> None:
    missing_schema = _observation()
    missing_schema.pop("schema")
    with pytest.raises(ReplaySchemaError, match="observation.schema"):
        ReplayObservation.from_api_dict(missing_schema)

    numeric_time = _observation()
    numeric_time["window"]["startTimeNs"] = 123
    with pytest.raises(ReplaySchemaError, match="decimal nanosecond string"):
        ReplayObservation.from_api_dict(numeric_time)

    complete = _observation()
    complete["window"]["completeState"] = True
    with pytest.raises(ReplaySchemaError, match="completeState"):
        ReplayObservation.from_api_dict(complete)

    wrong_state_basis = _observation()
    wrong_state_basis["quality"]["stateBasis"] = "periodic_snapshot"
    with pytest.raises(ReplaySchemaError, match="stateBasis='window_delta'"):
        ReplayObservation.from_api_dict(wrong_state_basis)

    mismatched_count = _observation()
    mismatched_count["quality"]["returnedEvents"] = 2
    with pytest.raises(ReplaySchemaError, match="returned-event count"):
        ReplayObservation.from_api_dict(mismatched_count)

    unknown_media = _observation()
    unknown_media["events"][0]["mediaIds"] = ["img_missing"]
    with pytest.raises(ReplaySchemaError, match="unknown media ID"):
        ReplayObservation.from_api_dict(unknown_media)

    invalid_boolean = _observation()
    invalid_boolean["quality"]["truncated"] = "false"
    with pytest.raises(ReplaySchemaError, match="must be a boolean"):
        ReplayObservation.from_api_dict(invalid_boolean)

    reasons_without_flag = _observation()
    reasons_without_flag["quality"]["truncationReasons"] = ["event_limit_before"]
    with pytest.raises(ReplaySchemaError, match="true exactly when"):
        ReplayObservation.from_api_dict(reasons_without_flag)

    flag_without_reasons = _observation()
    flag_without_reasons["quality"]["truncated"] = True
    with pytest.raises(ReplaySchemaError, match="true exactly when"):
        ReplayObservation.from_api_dict(flag_without_reasons)


def test_text_redaction_covers_headers_vendor_assignments_and_cli_flags() -> None:
    source = (
        '{"OPENAI_API_KEY":"json-secret","safe":"retained"}\n'
        "AWS_SECRET_ACCESS_KEY=aws-env-secret\n"
        "Authorization: Bearer bearer-secret\n"
        "Authorization: Basic basic-secret\n"
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAFIXTURE, "
        "SignedHeaders=host, Signature=aws-auth-secret\n"
        "Cookie: session=cookie-secret\n"
        "Set-Cookie: refresh=set-cookie-secret\n"
        "launch --api-key=flag-secret --access-key 'access-flag-secret'\n"
        "cp_live_1234567890 ghp_1234567890abcdef sk-1234567890abcdef"
    )

    redacted = redact_replay_text(source)

    for secret in (
        "json-secret",
        "aws-env-secret",
        "bearer-secret",
        "basic-secret",
        "AKIAFIXTURE",
        "aws-auth-secret",
        "cookie-secret",
        "set-cookie-secret",
        "flag-secret",
        "access-flag-secret",
        "cp_live_1234567890",
        "ghp_1234567890abcdef",
        "sk-1234567890abcdef",
    ):
        assert secret not in redacted
    assert '"OPENAI_API_KEY":"[REDACTED]"' in redacted
    assert '"safe":"retained"' in redacted
    assert "Authorization: [REDACTED]" in redacted
    assert "Cookie: [REDACTED]" in redacted
    assert "--api-key=[REDACTED]" in redacted
    assert "--access-key [REDACTED]" in redacted


def test_ingested_nanoseconds_respect_postgres_bigint_storage_contract() -> None:
    too_large = str(POSTGRES_BIGINT_MAX + 1)

    recording = _recording()
    recording["startTimeNs"] = too_large
    with pytest.raises(ReplaySchemaError, match="must be between"):
        ReplayRecording.from_api_dict(recording)

    summary = _summary()
    summary["controlEvents"]["lastTimeNs"] = too_large
    with pytest.raises(ReplaySchemaError, match="must be between"):
        ReplaySummary.from_api_dict(summary)

    compressed = gzip.compress(b"{}\n", mtime=0)
    chunk = _chunk(
        "chunk_bigint",
        sequence=0,
        channel="world/pose",
        time_ns=100,
        content=compressed,
    )
    chunk["endTimeNs"] = too_large
    with pytest.raises(ReplaySchemaError, match="must be between"):
        ReplayChunk.from_api_dict(chunk)

    for field in ("timeNs", "simTimeNs"):
        event = {
            "eventId": "evt_bigint",
            "channel": "world/pose",
            "type": "sample",
            "source": "fixture",
            "timeNs": "100",
            "simTimeNs": "100",
            "payload": {},
        }
        event[field] = too_large
        with pytest.raises(ReplaySchemaError, match="must be between"):
            ReplayEvent.from_api_dict(event)

    image = _observation()["images"][0]
    for field in ("timeNs", "simTimeNs"):
        invalid_image = dict(image)
        invalid_image[field] = too_large
        with pytest.raises(ReplaySchemaError, match="must be between"):
            ReplayImage.from_api_dict(invalid_image)

    for field in ("startTimeNs", "endTimeNs"):
        observation = _observation()
        observation["window"][field] = too_large
        with pytest.raises(ReplaySchemaError, match="must be between"):
            ReplayObservation.from_api_dict(observation)

    maximum_event = ReplayEvent.from_api_dict(
        {
            "eventId": "evt_max",
            "channel": "world/pose",
            "type": "sample",
            "source": "fixture",
            "timeNs": str(POSTGRES_BIGINT_MAX),
            "simTimeNs": str(POSTGRES_BIGINT_MAX),
            "payload": {},
        }
    )
    assert maximum_event.time_ns == POSTGRES_BIGINT_MAX
    assert maximum_event.sim_time_ns == POSTGRES_BIGINT_MAX
    assert (
        ReplayQuery(
            start_time_ns=POSTGRES_BIGINT_MAX,
            end_time_ns=POSTGRES_BIGINT_MAX,
        ).end_time_ns
        == POSTGRES_BIGINT_MAX
    )


@respx.mock
def test_one_sided_query_rejects_derived_end_above_postgres_bigint() -> None:
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=_summary())
    )

    with ReplayClient() as client:
        with pytest.raises(ReplaySchemaError, match="signed BIGINT limit"):
            client.get_observation(
                SESSION_ID,
                ReplayQuery(start_time_ns=POSTGRES_BIGINT_MAX, latest_seconds=1),
            )


@pytest.mark.parametrize("payload", [None, 7, "observed", [1, 2, 3]])
def test_raw_events_preserve_non_object_json_payloads(payload: Any) -> None:
    event = ReplayEvent.from_api_dict(
        {
            "eventId": "evt_payload",
            "channel": "custom/value",
            "type": "sample",
            "source": "fixture",
            "timeNs": "100",
            "payload": payload,
        }
    )
    assert event.payload == payload
    assert event.to_dict()["payload"] == payload
    assert event.to_dict(include_binary=True)["payload"] == payload


@pytest.mark.parametrize(
    "data_url",
    (
        "data:,plain-text",
        "data:text/plain;charset=utf-8,hello%20world",
        "data:image/svg+xml,%3Csvg%3Eunsafe%3C/svg%3E",
        "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'></svg>",
    ),
)
def test_raw_event_text_output_omits_every_data_url_variant(data_url: str) -> None:
    event = ReplayEvent.from_api_dict(
        {
            "eventId": "evt_data_url",
            "channel": "custom/value",
            "type": "sample",
            "source": "fixture",
            "timeNs": "100",
            "payload": {"value": f"prefix {data_url} suffix"},
        }
    )

    serialized = json.dumps(event.to_dict())
    assert "data:" not in serialized
    assert "[DATA URL OMITTED]" in serialized


@respx.mock
def test_observation_resolves_custom_one_sided_window_and_verifies_session() -> None:
    summary = _summary()
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=summary)
    )
    response = _observation(include_image=False)
    response["window"]["startTimeNs"] = "100"
    response["window"]["endTimeNs"] = "60000000100"
    response["events"] = []
    response["quality"].update({"returnedEvents": 0, "matchedEvents": 0, "returnedImages": 0})
    route = respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/observations").mock(
        return_value=httpx.Response(200, json=response)
    )

    with ReplayClient() as client:
        observation = client.get_observation(
            SESSION_ID,
            ReplayQuery(start_time_ns=100, latest_seconds=60),
        )

    assert route.calls[0].request.url.params["startTimeNs"] == "100"
    assert route.calls[0].request.url.params["endTimeNs"] == "60000000100"
    assert observation.effective_query is not None
    assert observation.effective_query.latest_seconds is None


@respx.mock
def test_default_observation_lets_server_anchor_empty_session_latest_window() -> None:
    summary = _summary()
    summary["recordings"] = []
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=summary)
    )
    response = _observation(include_image=False)
    response["window"].update(
        {
            "startTimeNs": "1000000000000",
            "endTimeNs": "1030000000000",
        }
    )
    response["archive"] = {"formatVersion": 1, "channels": [], "recordingIds": []}
    response["events"] = []
    response["quality"].update(
        {
            "returnedEvents": 0,
            "matchedEvents": 0,
            "matchedEventsExact": True,
            "returnedImages": 0,
            "matchedImages": 0,
        }
    )
    route = respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/observations").mock(
        return_value=httpx.Response(200, json=response)
    )

    with ReplayClient() as client:
        observation = client.get_observation(SESSION_ID)

    params = route.calls[0].request.url.params
    assert "startTimeNs" not in params
    assert "endTimeNs" not in params
    assert observation.start_time_ns == 1_000_000_000_000
    assert observation.end_time_ns == 1_030_000_000_000
    assert observation.effective_query is not None
    assert observation.effective_query.start_time_ns == observation.start_time_ns
    assert observation.effective_query.end_time_ns == observation.end_time_ns
    assert observation.effective_query.latest_seconds is None


@respx.mock
def test_custom_implicit_latest_observation_anchors_on_server_without_summary_race() -> None:
    anchor = _observation(include_image=False)
    selected = _observation(include_image=False)
    selected["window"]["startTimeNs"] = "999999940000000009"
    route = respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/observations").mock(
        side_effect=[
            httpx.Response(200, json=anchor),
            httpx.Response(200, json=selected),
        ]
    )

    with ReplayClient() as client:
        observation = client.get_observation(
            SESSION_ID,
            ReplayQuery(latest_seconds=60.0),
        )

    assert len(route.calls) == 2
    anchor_params = route.calls[0].request.url.params
    assert "startTimeNs" not in anchor_params
    assert "endTimeNs" not in anchor_params
    assert anchor_params["maxEvents"] == "1"
    assert anchor_params["maxImages"] == "0"
    selected_params = route.calls[1].request.url.params
    assert selected_params["startTimeNs"] == "999999940000000009"
    assert selected_params["endTimeNs"] == "1000000000000000009"
    assert observation.effective_query is not None
    assert observation.effective_query.start_time_ns == 999_999_940_000_000_009
    assert observation.effective_query.end_time_ns == 1_000_000_000_000_000_009


@respx.mock
def test_observation_rejects_cross_session_response() -> None:
    response = _observation(include_image=False)
    response["sessionId"] = "sess_other"
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/observations").mock(
        return_value=httpx.Response(200, json=response)
    )
    query = ReplayQuery(
        start_time_ns=999_999_970_000_000_009,
        end_time_ns=1_000_000_000_000_000_009,
    )
    with ReplayClient() as client:
        with pytest.raises(ReplayError, match="does not match the requested session"):
            client.get_observation(SESSION_ID, query)


def test_strict_resource_ids_reject_url_delimiters() -> None:
    client = ReplayClient(api_key="cp_live_test", http_client=object())
    with pytest.raises(ReplaySchemaError, match="resource ID"):
        client.get_summary("sess?redirect=/v1/admin")


def test_query_rejects_ambiguous_channels_and_out_of_range_nanoseconds() -> None:
    with pytest.raises(ReplaySchemaError, match="no commas"):
        ReplayQuery(channels=("camera/main,world/pose",))
    with pytest.raises(ReplaySchemaError, match="must be between"):
        ReplayQuery(start_time_ns=POSTGRES_BIGINT_MAX + 1)
    with pytest.raises(ReplaySchemaError, match="Python integer"):
        ReplayQuery(start_time_ns=1.0)  # type: ignore[arg-type]
    with pytest.raises(ReplaySchemaError, match="Python integer"):
        ReplayQuery(max_events=True)  # type: ignore[arg-type]
    with pytest.raises(ReplaySchemaError, match="cannot exceed 300 seconds"):
        ReplayQuery(start_time_ns=0, end_time_ns=300_000_000_001)
    with pytest.raises(ReplaySchemaError, match="required when either time bound"):
        ReplayQuery(latest_seconds=None)


def test_channel_patterns_match_server_star_and_question_semantics() -> None:
    from cybernetics.replay.client import _channel_matches

    assert _channel_matches("camera/main", ("camera/*",))
    assert _channel_matches("camera/a", ("camera/?",))
    assert _channel_matches("camera/[ab]", ("camera/[ab]",))
    assert not _channel_matches("camera/a", ("camera/[ab]",))


def test_composed_client_replay_namespace_is_lazy_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls: list[bool] = []
    monkeypatch.setattr(ReplayClient, "close", lambda self: close_calls.append(True))

    client = Client(api_key="cp_live_test", base_url=BASE)
    assert client._replay is None
    assert isinstance(client.replay, ReplayClient)
    client.close()
    assert close_calls == [True]


def test_composed_client_closes_sim_when_replay_cleanup_fails() -> None:
    calls: list[str] = []

    class Namespace:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} cleanup failed")

    client = Client(api_key="cp_live_test", base_url=BASE)
    client._replay = Namespace("replay", fail=True)  # type: ignore[assignment]
    client._sim = Namespace("sim")

    with pytest.raises(RuntimeError, match="replay cleanup failed"):
        client.close()
    assert calls == ["replay", "sim"]


def test_replay_context_manager_preserves_primary_error_when_close_fails() -> None:
    class CloseFailure:
        def close(self) -> None:
            raise RuntimeError("transport cleanup failed")

    client = ReplayClient(api_key="cp_live_test", http_client=CloseFailure())
    client._owns_client = True

    with pytest.raises(ValueError, match="primary failure") as captured:
        with client:
            raise ValueError("primary failure")

    assert any("transport cleanup failed" in note for note in captured.value.__notes__)


def test_raw_scan_budget_fails_before_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    from cybernetics.replay import client as replay_client_module

    monkeypatch.setattr(replay_client_module, "_MAX_SELECTED_CHUNKS", 0)

    class FakeReplayClient(ReplayClient):
        def get_summary(self, session_id: str):
            from cybernetics.replay.models import ReplaySummary

            return ReplaySummary.from_api_dict(_summary())

        def _iter_chunks(self, session_id: str, recording_id: str, **kwargs: Any):
            from cybernetics.replay.models import ReplayChunk

            content = gzip.compress(b"{}\n", mtime=0)
            yield ReplayChunk.from_api_dict(
                _chunk(
                    "chunk_camera",
                    sequence=0,
                    channel="camera/main",
                    time_ns=200,
                    content=content,
                )
            )

    client = FakeReplayClient(api_key="cp_live_test", http_client=object())
    with pytest.raises(ReplayError, match="narrower time window"):
        list(client.iter_events(SESSION_ID, ReplayQuery(include_control_events=False)))


def test_raw_event_budget_is_preflighted_before_chunk_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cybernetics.replay import client as replay_client_module
    from cybernetics.replay.models import ReplayChunk, ReplaySummary

    monkeypatch.setattr(replay_client_module, "_MAX_EVENTS_SCANNED", 1)
    content = gzip.compress(b"fixture", mtime=0)
    chunk_data = _chunk(
        "chunk_camera",
        sequence=0,
        channel="camera/main",
        time_ns=200,
        content=content,
    )
    chunk_data["eventCount"] = 2
    chunk = ReplayChunk.from_api_dict(chunk_data)
    downloaded = False

    class FakeReplayClient(ReplayClient):
        def get_summary(self, session_id: str) -> ReplaySummary:
            return ReplaySummary.from_api_dict(_summary())

        def _iter_chunks(self, *args: Any, **kwargs: Any):
            yield chunk

        def _download_chunk_events(self, *args: Any, **kwargs: Any):
            nonlocal downloaded
            downloaded = True
            return (), 0

    client = FakeReplayClient(api_key="cp_live_test", http_client=object())
    with pytest.raises(ReplayError, match="event scan budget"):
        list(client.iter_events(SESSION_ID, ReplayQuery(include_control_events=False)))
    assert downloaded is False


def test_raw_events_preserve_same_event_id_from_distinct_sources() -> None:
    from cybernetics.replay.models import ReplayChunk, ReplaySummary

    content = gzip.compress(b"fixture", mtime=0)
    chunk_data = _chunk(
        "chunk_camera",
        sequence=0,
        channel="camera/main",
        time_ns=200,
        content=content,
    )
    chunk_data["eventCount"] = 2
    chunk = ReplayChunk.from_api_dict(chunk_data)
    events = tuple(
        ReplayEvent.from_api_dict(
            {
                "eventId": "shared_id",
                "channel": "camera/main",
                "type": "sample",
                "source": source,
                "timeNs": "200",
                "payload": {},
            }
        )
        for source in ("source_b", "source_a")
    )

    class FakeReplayClient(ReplayClient):
        def get_summary(self, session_id: str) -> ReplaySummary:
            return ReplaySummary.from_api_dict(_summary())

        def _iter_chunks(self, *args: Any, **kwargs: Any):
            yield chunk

        def _download_chunk_events(self, *args: Any, **kwargs: Any):
            return events, 7

    client = FakeReplayClient(api_key="cp_live_test", http_client=object())
    result = list(
        client.iter_events(
            SESSION_ID,
            ReplayQuery(
                start_time_ns=100,
                end_time_ns=500,
                include_control_events=False,
            ),
        )
    )
    assert [(event.event_id, event.source) for event in result] == [
        ("shared_id", "source_a"),
        ("shared_id", "source_b"),
    ]


@respx.mock
def test_raw_chunk_event_count_must_match_index() -> None:
    event = {
        "eventId": "evt_camera",
        "channel": "camera/main",
        "type": "camera.frame",
        "source": "isaac-sim",
        "timeNs": "200",
        "payload": {},
    }
    content = gzip.compress((json.dumps(event) + "\n").encode(), mtime=0)
    chunk = _chunk(
        "chunk_camera",
        sequence=0,
        channel="camera/main",
        time_ns=200,
        content=content,
    )
    chunk["eventCount"] = 2
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=_summary())
    )
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks").mock(
        return_value=httpx.Response(200, json={"items": [chunk], "nextCursor": None})
    )
    respx.get(
        f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks/chunk_camera/content"
    ).mock(return_value=httpx.Response(200, content=content))

    with ReplayClient() as client:
        with pytest.raises(ReplayError, match="event count does not match"):
            list(
                client.iter_events(
                    SESSION_ID,
                    ReplayQuery(
                        start_time_ns=100,
                        end_time_ns=500,
                        include_control_events=False,
                    ),
                )
            )


@respx.mock
def test_raw_chunk_pagination_rejects_repeated_cursor() -> None:
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=_summary())
    )
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks").mock(
        side_effect=[
            httpx.Response(200, json={"items": [], "nextCursor": "repeat"}),
            httpx.Response(200, json={"items": [], "nextCursor": "repeat"}),
        ]
    )
    with ReplayClient() as client:
        with pytest.raises(ReplaySchemaError, match="repeated a cursor"):
            list(
                client.iter_events(
                    SESSION_ID,
                    ReplayQuery(include_control_events=False),
                )
            )


@respx.mock
def test_raw_chunk_index_page_budget_is_global_across_recordings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cybernetics.replay import client as replay_client_module

    monkeypatch.setattr(replay_client_module, "_MAX_CHUNK_INDEX_PAGES", 1)
    summary = _summary()
    second = dict(_recording())
    second["recordingId"] = "rec_second"
    second["writerId"] = "writer_second"
    summary["recordings"].append(second)
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=summary)
    )
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks").mock(
        return_value=httpx.Response(200, json={"items": [], "nextCursor": None})
    )
    second_route = respx.get(
        f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_second/chunks"
    ).mock(return_value=httpx.Response(200, json={"items": [], "nextCursor": None}))

    with ReplayClient() as client:
        with pytest.raises(ReplayError, match="exceeds 64 pages"):
            list(
                client.iter_events(
                    SESSION_ID,
                    ReplayQuery(include_control_events=False),
                )
            )
    assert second_route.called is False


@respx.mock
def test_observation_response_is_bound_to_requested_limits_channels_and_image_mode() -> None:
    path = f"{BASE}/v1/sessions/{SESSION_ID}/timeline/observations"
    query = ReplayQuery(
        channels=("camera/*",),
        start_time_ns=999_999_970_000_000_009,
        end_time_ns=1_000_000_000_000_000_009,
        max_events=1,
        max_images=1,
    )

    too_many = _observation(include_image=False)
    second_event = dict(too_many["events"][0])
    second_event["eventId"] = "evt_camera_2"
    second_event["timeNs"] = "1000000000000000006"
    too_many["events"].append(second_event)
    too_many["quality"].update({"returnedEvents": 2, "matchedEvents": 2})
    respx.get(path).mock(return_value=httpx.Response(200, json=too_many))
    with ReplayClient() as client:
        with pytest.raises(ReplayIntegrityError, match="max_events"):
            client.get_observation(SESSION_ID, query)

    wrong_channel = _observation(include_image=False)
    wrong_channel["events"][0]["channel"] = "world/pose"
    respx.get(path).mock(return_value=httpx.Response(200, json=wrong_channel))
    with ReplayClient() as client:
        with pytest.raises(ReplayIntegrityError, match="outside the requested channels"):
            client.get_observation(SESSION_ID, query)

    respx.get(path).mock(return_value=httpx.Response(200, json=_observation()))
    with ReplayClient() as client:
        with pytest.raises(ReplayIntegrityError, match="image_data was omitted"):
            client.get_observation(SESSION_ID, query, image_data=False)

    respx.get(path).mock(return_value=httpx.Response(200, json=_observation()))
    with ReplayClient() as client:
        with pytest.raises(ReplayIntegrityError, match="max_images"):
            client.get_observation(
                SESSION_ID,
                ReplayQuery(
                    channels=("camera/*",),
                    start_time_ns=query.start_time_ns,
                    end_time_ns=query.end_time_ns,
                    max_images=0,
                ),
                image_data=True,
            )


@respx.mock
def test_summary_and_chunk_responses_are_bound_to_requested_identities() -> None:
    cross_session = _summary()
    cross_session["sessionId"] = "sess_other"
    cross_session["recordings"][0]["sessionId"] = "sess_other"
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, json=cross_session)
    )
    with ReplayClient() as client:
        with pytest.raises(ReplayIntegrityError, match="summary session_id"):
            client.get_summary(SESSION_ID)

    content = gzip.compress(b"{}\n", mtime=0)
    wrong_recording = _chunk(
        "chunk_other",
        sequence=0,
        channel="camera/main",
        time_ns=200,
        content=content,
    )
    wrong_recording["recordingId"] = "rec_other"
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline/recordings/rec_demo/chunks").mock(
        return_value=httpx.Response(
            200,
            json={"items": [wrong_recording], "nextCursor": None},
        )
    )
    with ReplayClient() as client:
        with pytest.raises(ReplayIntegrityError, match="requested recording"):
            list(client._iter_chunks(SESSION_ID, "rec_demo"))


def test_v1_summary_recording_chunk_and_time_bound_contracts_are_strict() -> None:
    invalid_summary = _summary()
    invalid_summary["formatVersion"] = 2
    with pytest.raises(ReplaySchemaError, match="summary.format_version must be 1"):
        ReplaySummary.from_api_dict(invalid_summary)

    invalid_recording = _recording()
    invalid_recording["formatVersion"] = 2
    with pytest.raises(ReplaySchemaError, match="recording.format_version must be 1"):
        ReplayRecording.from_api_dict(invalid_recording)

    reversed_recording = _recording()
    reversed_recording["startTimeNs"] = "501"
    with pytest.raises(ReplaySchemaError, match="must not precede"):
        ReplayRecording.from_api_dict(reversed_recording)

    invalid_identity = _summary()
    invalid_identity["recordings"][0]["sessionId"] = "sess_other"
    with pytest.raises(ReplaySchemaError, match="recording session_id"):
        ReplaySummary.from_api_dict(invalid_identity)

    content = gzip.compress(b"{}\n", mtime=0)
    reversed_chunk = _chunk(
        "chunk_reversed",
        sequence=0,
        channel="camera/main",
        time_ns=200,
        content=content,
    )
    reversed_chunk["startTimeNs"] = "201"
    with pytest.raises(ReplaySchemaError, match="must not precede"):
        ReplayChunk.from_api_dict(reversed_chunk)

    invalid_archive = _observation()
    invalid_archive["archive"]["formatVersion"] = 2
    with pytest.raises(ReplaySchemaError, match="archive.format_version must be 1"):
        ReplayObservation.from_api_dict(invalid_archive)


def test_observation_requires_exact_image_event_linkage_and_metadata() -> None:
    mismatched_link = _observation()
    mismatched_link["images"][0]["eventId"] = "evt_other"
    with pytest.raises(ReplaySchemaError, match="exactly one matching"):
        ReplayObservation.from_api_dict(mismatched_link)

    mismatched_metadata = _observation()
    mismatched_metadata["images"][0]["simTimeNs"] = "5000000002"
    with pytest.raises(ReplaySchemaError, match="metadata does not match"):
        ReplayObservation.from_api_dict(mismatched_metadata)

    missing_media_type = _observation()
    missing_media_type["images"][0].pop("mimeType")
    with pytest.raises(ReplaySchemaError, match="declare mimeType or format"):
        ReplayObservation.from_api_dict(missing_media_type)

    oversized_metadata = _observation()
    oversized_metadata["images"][0].pop("dataBase64")
    oversized_metadata["images"][0].pop("sha256")
    oversized_metadata["images"][0]["sizeBytes"] = 16 * 1024 * 1024 + 1
    with pytest.raises(ReplaySchemaError, match="inline image byte limit"):
        ReplayObservation.from_api_dict(oversized_metadata)


def test_observation_exposes_inexact_matched_event_lower_bound() -> None:
    response = _observation(include_image=False)
    response["quality"]["matchedEventsExact"] = False

    observation = ReplayObservation.from_api_dict(response)

    assert observation.available == 1
    assert observation.available_exact is False
    assert observation.to_dict()["page"]["available_exact"] is False
    assert "lower bound" in " ".join(observation.warnings)

    legacy_response = _observation(include_image=False)
    del legacy_response["quality"]["matchedEventsExact"]
    legacy = ReplayObservation.from_api_dict(legacy_response)
    assert legacy.available_exact is False
    assert "lower bound" in " ".join(legacy.warnings)


@respx.mock
def test_json_ingress_rejects_non_finite_numbers_and_excessive_depth() -> None:
    non_finite = json.dumps(_summary()).replace('"captureConfig": {}', '"captureConfig": NaN')
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, content=non_finite.encode())
    )
    with ReplayClient() as client:
        with pytest.raises(ReplaySchemaError, match="strict JSON"):
            client.get_summary(SESSION_ID)

    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(70):
        child: dict[str, Any] = {}
        cursor["nested"] = child
        cursor = child
    deep_summary = _summary()
    deep_summary["recordings"][0]["captureConfig"] = nested
    respx.get(f"{BASE}/v1/sessions/{SESSION_ID}/timeline").mock(
        return_value=httpx.Response(200, content=json.dumps(deep_summary).encode())
    )
    with ReplayClient() as client:
        with pytest.raises(ReplaySchemaError, match="nesting limit"):
            client.get_summary(SESSION_ID)

    from cybernetics.replay.client import _parse_ndjson_events, _strict_json_loads

    with pytest.raises(ReplaySchemaError, match="strict JSON|nesting limit"):
        _strict_json_loads("[" * 2_000 + "0" + "]" * 2_000, "deep fixture")

    event = {
        "eventId": "evt_nonfinite",
        "channel": "world/pose",
        "type": "sample",
        "source": "fixture",
        "timeNs": "100",
        "payload": float("nan"),
    }
    with pytest.raises(ReplaySchemaError, match="strict JSON"):
        _parse_ndjson_events(json.dumps(event) + "\n", "chunk_nonfinite")
    overflow_json = json.dumps(event).replace("NaN", "1e999")
    with pytest.raises(ReplaySchemaError, match="non-finite number"):
        _parse_ndjson_events(overflow_json + "\n", "chunk_overflow")
    duplicate_key_json = json.dumps(event).replace(
        '"eventId": "evt_nonfinite"',
        '"eventId": "evt_first", "eventId": "evt_second"',
    ).replace("NaN", "0")
    with pytest.raises(ReplaySchemaError, match="strict JSON"):
        _parse_ndjson_events(duplicate_key_json + "\n", "chunk_duplicate")


def test_control_event_pagination_has_an_independent_page_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cybernetics.replay import client as replay_client_module

    monkeypatch.setattr(replay_client_module, "_MAX_CONTROL_EVENT_PAGES", 1)

    class FakeReplayClient(ReplayClient):
        def _request_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"items": [], "nextAfterId": "next_page"}

    client = FakeReplayClient(api_key="cp_live_test", http_client=object())
    with pytest.raises(ReplayError, match="control-event index exceeds 64 pages"):
        list(client._iter_control_events(SESSION_ID))


def test_raw_event_selection_exposes_match_count_and_truncation_direction() -> None:
    content = gzip.compress(b"fixture", mtime=0)
    chunk_data = _chunk(
        "chunk_selection",
        sequence=0,
        channel="world/pose",
        time_ns=200,
        content=content,
    )
    chunk_data.update({"eventCount": 2, "endTimeNs": "300"})
    chunk = ReplayChunk.from_api_dict(chunk_data)
    events = tuple(
        ReplayEvent.from_api_dict(
            {
                "eventId": f"evt_{time_ns}",
                "channel": "world/pose",
                "type": "sample",
                "source": "fixture",
                "timeNs": str(time_ns),
                "payload": {},
            }
        )
        for time_ns in (200, 300)
    )

    class FakeReplayClient(ReplayClient):
        def get_summary(self, session_id: str) -> ReplaySummary:
            return ReplaySummary.from_api_dict(_summary())

        def _iter_chunks(self, *args: Any, **kwargs: Any):
            yield chunk

        def _download_chunk_events(self, *args: Any, **kwargs: Any):
            return events, 7

    client = FakeReplayClient(api_key="cp_live_test", http_client=object())
    explicit = client.select_events(
        SESSION_ID,
        ReplayQuery(
            start_time_ns=100,
            end_time_ns=500,
            max_events=1,
            include_control_events=False,
        ),
    )
    assert [event.event_id for event in explicit.events] == ["evt_200"]
    assert explicit.matched_events == 2
    assert explicit.truncated is True
    assert explicit.truncation_reason == "max_events_after"
    assert explicit.to_dict()["page"] == {
        "returned": 1,
        "matched": 2,
        "truncated": True,
        "truncation_reason": "max_events_after",
    }

    latest = client.select_events(
        SESSION_ID,
        ReplayQuery(max_events=1, include_control_events=False),
    )
    assert [event.event_id for event in latest.events] == ["evt_300"]
    assert latest.truncation_reason == "max_events_before"


def test_agent_surfaces_sanitize_untrusted_labels_and_markdown(tmp_path: Path) -> None:
    response = _observation()
    malicious_id = "evt`]([bad](https://example.invalid))"
    malicious_media_id = "img`]([bad](https://example.invalid))"
    malicious_source = "data:image/jpeg;base64,VE9LRU4="
    response["events"][0].update(
        {
            "eventId": malicious_id,
            "source": malicious_source,
            "mediaIds": [malicious_media_id],
        }
    )
    response["images"][0].update(
        {
            "imageId": malicious_media_id,
            "eventId": malicious_id,
            "source": malicious_source,
        }
    )
    response["summary"] = "data:image/png;base64,SEVMTE8=\n# injected heading"
    response["quality"]["warnings"] = [
        "line one\n# attacker [click](https://example.invalid) Authorization: Token top-secret"
    ]
    observation = ReplayObservation.from_api_dict(response)

    provider_text = "\n".join(
        block["text"]
        for block in observation.to_openai_content()
        if block["type"] == "input_text"
    )
    assert "data:image" not in provider_text
    assert "top-secret" not in provider_text
    assert "\n# attacker" not in provider_text

    class FakeClient:
        def get_observation(self, *args: Any, **kwargs: Any) -> ReplayObservation:
            return observation

    client = ReplayClient(api_key="cp_live_test", http_client=object())
    client.get_observation = FakeClient().get_observation  # type: ignore[method-assign]
    bundle = client.export_agent_bundle(SESSION_ID, tmp_path / "safe-context")
    context = bundle.context_path.read_text()
    assert "data:image" not in context
    assert "top-secret" not in context
    assert "\n# injected heading" not in context
    assert "\n# attacker" not in context
    assert "[click](https://example.invalid)" not in context


def test_bundle_cleanup_does_not_mask_the_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cybernetics.replay import agent as replay_agent_module

    observation = ReplayObservation.from_api_dict(_observation(include_image=False))

    class FakeClient:
        def get_observation(self, *args: Any, **kwargs: Any) -> ReplayObservation:
            return observation

    client = ReplayClient(api_key="cp_live_test", http_client=object())
    client.get_observation = FakeClient().get_observation  # type: ignore[method-assign]

    def fail_build(*args: Any, **kwargs: Any) -> None:
        raise ValueError("primary failure")

    def fail_cleanup(*args: Any, **kwargs: Any) -> None:
        raise OSError("cleanup failure")

    monkeypatch.setattr(replay_agent_module, "_build_bundle_files", fail_build)
    monkeypatch.setattr(replay_agent_module.shutil, "rmtree", fail_cleanup)
    with pytest.raises(ValueError, match="primary failure"):
        client.export_agent_bundle(SESSION_ID, tmp_path / "cleanup")
