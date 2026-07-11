from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cybernetics.robotics.worldlines_policy_service as policy_service_module
from cybernetics.robotics import (
    WorldlinesPolicyConnection,
    WorldlinesPolicyConnectionError,
    WorldlinesPolicyServiceClient,
)
from cybernetics.robotics.worldlines_policy_service import _TensorCodec


def _connection(**overrides):
    data = {
        "schema_version": "worldlines-policy-connection/v1",
        "grant_id": "wlpg-fixture",
        "run_id": "eval-fixture",
        "deployment_id": "wlp-fixture",
        "session_id": "wlps-fixture",
        "transport": "unix_msgpack_shm_v1",
        "endpoint_address": "/tmp/worldlines-policy.sock",
        "shared_memory_root": "/tmp/worldlines-policy-shm",
        "grant_token": "grant-token-" + "x" * 32,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
    }
    data.update(overrides)
    return data


def test_worldlines_policy_connection_is_strict_and_redacts_grant() -> None:
    connection = WorldlinesPolicyConnection.from_dict(_connection())
    assert connection.deployment_id == "wlp-fixture"
    redacted = connection.redacted_dict()
    assert redacted["credential"] == "[REDACTED]"
    assert connection.grant_token not in json.dumps(redacted)
    assert connection.grant_token not in repr(connection)

    with pytest.raises(WorldlinesPolicyConnectionError, match="unknown fields"):
        WorldlinesPolicyConnection.from_dict(_connection(url="https://worker.invalid"))
    with pytest.raises(WorldlinesPolicyConnectionError, match="expired"):
        WorldlinesPolicyConnection.from_dict(
            _connection(expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat())
        )
    with pytest.raises(WorldlinesPolicyConnectionError, match="at least 32"):
        WorldlinesPolicyConnection.from_dict(_connection(grant_token="short"))
    with pytest.raises(WorldlinesPolicyConnectionError, match="normalized path"):
        WorldlinesPolicyConnection.from_dict(
            _connection(endpoint_address="/tmp//worldlines-policy.sock")
        )


def test_worldlines_policy_connection_file_must_be_private_regular_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy-connection.json"
    path.write_text(json.dumps(_connection()))
    path.chmod(0o600)
    assert WorldlinesPolicyConnection.load(path).grant_id == "wlpg-fixture"

    path.chmod(0o644)
    with pytest.raises(WorldlinesPolicyConnectionError, match="private regular file"):
        WorldlinesPolicyConnection.load(path)

    path.chmod(0o600)
    link = tmp_path / "policy-connection-link.json"
    link.symlink_to(path)
    with pytest.raises(WorldlinesPolicyConnectionError, match="private regular file"):
        WorldlinesPolicyConnection.load(link)


def test_sdk_tensor_codec_uses_binary_shared_memory_without_json_lists(
    tmp_path: Path,
) -> None:
    codec = _TensorCodec(tmp_path / "shm")
    camera = np.zeros((1, 512, 512, 3), dtype=np.uint8)
    encoded, attachments, stats = codec.encode({"rgb": camera})
    try:
        envelope = encoded["rgb"]
        assert envelope["__worldlines_tensor_v1__"] == "shm"
        assert "data" not in envelope
        assert stats.shared_memory_tensor_count == 1
        assert stats.shared_memory_bytes == camera.nbytes
        assert len(attachments) == 1
        assert stat.S_IMODE(attachments[0].stat().st_mode) == 0o600
        assert attachments[0].stat().st_uid == os.getuid()
    finally:
        codec.cleanup(attachments)
    assert list((tmp_path / "shm").iterdir()) == []


def test_sdk_tensor_codec_decodes_inline_action_tensor(tmp_path: Path) -> None:
    codec = _TensorCodec(tmp_path / "shm")
    values = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    decoded = codec.decode(
        {
            "values": {
                "__worldlines_tensor_v1__": "inline",
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "byte_length": values.nbytes,
                "data": values.tobytes(),
            }
        }
    )
    assert np.array_equal(decoded["values"], values)


@pytest.mark.parametrize(
    ("descriptor_override", "message"),
    [
        ({"session_id": "different-session"}, "session does not match"),
        ({"transport": "unexpected-transport"}, "transport does not match"),
    ],
)
def test_sdk_policy_client_rejects_descriptor_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    descriptor_override: dict[str, str],
    message: str,
) -> None:
    descriptor = {
        "protocol_version": "policy-service/v1",
        "session_id": "wlps-fixture",
        "policy_deployment_hash": "a" * 64,
        "policy_deployment_id": "wlp-fixture",
        "policy_revision": "fixture@1",
        "batch_size": 1,
        "max_horizon": 1,
        "state_model": "stateless",
        "reset_granularity": "environment",
        "deterministic": True,
        "observation_schema": {},
        "action_spec": {},
        "transport": "unix_msgpack_shm_v1",
    }
    descriptor.update(descriptor_override)

    class _FakeRpc:
        instance: "_FakeRpc | None" = None

        def __init__(self, *_args, **_kwargs) -> None:
            self.close_requested = False
            self.socket_closed = False
            _FakeRpc.instance = self

        def call(self, method: str, _payload):
            if method == "open":
                return descriptor
            if method == "close":
                self.close_requested = True
                return {"closed": True}
            raise AssertionError(f"unexpected method {method}")

        def close(self) -> None:
            self.socket_closed = True

    monkeypatch.setattr(policy_service_module, "_UnixPolicyRpcClient", _FakeRpc)
    connection = WorldlinesPolicyConnection.from_dict(_connection())
    policy = SimpleNamespace(
        source="worldlines",
        deployment_id="wlp-fixture",
        max_batch_size=1,
    )

    with pytest.raises(WorldlinesPolicyConnectionError, match=message):
        WorldlinesPolicyServiceClient(
            connection,
            policy,
            batch_size=1,
            sequence_ids=["lane-0"],
            seeds=[1],
        )

    assert _FakeRpc.instance is not None
    assert _FakeRpc.instance.close_requested is True
    assert _FakeRpc.instance.socket_closed is True
