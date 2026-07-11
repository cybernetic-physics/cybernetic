"""Direct Worldlines ``PolicyService/v1`` client and secret connection contract."""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import msgpack
import numpy as np

from .runtime_contracts import ActionChunk, PolicyDeploymentSpec
from .runtime_services import PolicyServiceDescriptor, RobotServiceContractError

WORLDLINES_POLICY_CONNECTION_VERSION = "worldlines-policy-connection/v1"
WORLDLINES_POLICY_TRANSPORT = "unix_msgpack_shm_v1"

MAX_FRAME_BYTES = 128 * 1024 * 1024
MAX_TENSOR_BYTES = 64 * 1024 * 1024
MAX_TOTAL_TENSOR_BYTES = 256 * 1024 * 1024
MAX_CONTAINER_ITEMS = 1_000_000
SHARED_MEMORY_THRESHOLD_BYTES = 256 * 1024

_FRAME_HEADER = struct.Struct(">Q")
_TENSOR_MARKER = "__worldlines_tensor_v1__"


class WorldlinesPolicyConnectionError(RobotServiceContractError):
    """A resolved policy connection or its secret file is invalid."""


class WorldlinesPolicyTransportError(RuntimeError):
    """The direct Worldlines transport failed or violated its wire contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class WorldlinesPolicyServiceError(RuntimeError):
    """The Worldlines policy service rejected an authenticated request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class WorldlinesPolicyConnection:
    """Control-plane-resolved secret binding kept outside ``robotics-job/v1``."""

    schema_version: str
    grant_id: str
    run_id: str
    deployment_id: str
    session_id: str
    transport: str
    endpoint_address: str
    shared_memory_root: str
    grant_token: str = field(repr=False)
    expires_at: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldlinesPolicyConnection":
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise WorldlinesPolicyConnectionError("policy connection must be an object")
        allowed = {
            "schema_version",
            "grant_id",
            "run_id",
            "deployment_id",
            "session_id",
            "transport",
            "endpoint_address",
            "shared_memory_root",
            "grant_token",
            "expires_at",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise WorldlinesPolicyConnectionError(f"policy connection has unknown fields {unknown}")
        schema_version = _string(value.get("schema_version"), "schema_version")
        if schema_version != WORLDLINES_POLICY_CONNECTION_VERSION:
            raise WorldlinesPolicyConnectionError(
                f"policy connection.schema_version must be {WORLDLINES_POLICY_CONNECTION_VERSION!r}"
            )
        transport = _string(value.get("transport"), "transport")
        if transport != WORLDLINES_POLICY_TRANSPORT:
            raise WorldlinesPolicyConnectionError(
                f"policy connection.transport must be {WORLDLINES_POLICY_TRANSPORT!r}"
            )
        endpoint_address = _absolute_normalized_path(
            value.get("endpoint_address"), "endpoint_address"
        )
        shared_memory_root = _absolute_normalized_path(
            value.get("shared_memory_root"), "shared_memory_root"
        )
        expires_at = _string(value.get("expires_at"), "expires_at")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise WorldlinesPolicyConnectionError(
                "policy connection.expires_at must be ISO-8601"
            ) from exc
        if expiry.tzinfo is None:
            raise WorldlinesPolicyConnectionError(
                "policy connection.expires_at must include a timezone"
            )
        if expiry.astimezone(UTC) <= datetime.now(UTC):
            raise WorldlinesPolicyConnectionError("policy connection grant is expired")
        grant_token = _string(value.get("grant_token"), "grant_token")
        if len(grant_token) < 32:
            raise WorldlinesPolicyConnectionError(
                "policy connection.grant_token must contain at least 32 characters"
            )
        return cls(
            schema_version=schema_version,
            grant_id=_string(value.get("grant_id"), "grant_id"),
            run_id=_string(value.get("run_id"), "run_id"),
            deployment_id=_string(value.get("deployment_id"), "deployment_id"),
            session_id=_string(value.get("session_id"), "session_id"),
            transport=transport,
            endpoint_address=endpoint_address,
            shared_memory_root=shared_memory_root,
            grant_token=grant_token,
            expires_at=expires_at,
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorldlinesPolicyConnection":
        connection_path = Path(path)
        try:
            before = connection_path.lstat()
            if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise WorldlinesPolicyConnectionError(
                    "policy connection file must be an owned private regular file"
                )
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(connection_path, flags)
        except OSError as exc:
            raise WorldlinesPolicyConnectionError(
                "policy connection file cannot be inspected"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_dev != metadata.st_dev
                or before.st_ino != metadata.st_ino
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077
                or metadata.st_size > 64 * 1024
            ):
                raise WorldlinesPolicyConnectionError(
                    "policy connection file must be an owned private regular file"
                )
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    raise WorldlinesPolicyConnectionError(
                        "policy connection file changed while being read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise WorldlinesPolicyConnectionError(
                    "policy connection file changed while being read"
                )
            payload = json.loads(b"".join(chunks).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorldlinesPolicyConnectionError(
                "policy connection file cannot be decoded"
            ) from exc
        finally:
            os.close(descriptor)
        return cls.from_dict(payload)

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grant_id": self.grant_id,
            "run_id": self.run_id,
            "deployment_id": self.deployment_id,
            "session_id": self.session_id,
            "transport": self.transport,
            "expires_at": self.expires_at,
            "credential": "[REDACTED]",
        }


@dataclass(frozen=True)
class _TransferStats:
    tensor_count: int = 0
    tensor_bytes: int = 0
    shared_memory_tensor_count: int = 0
    shared_memory_bytes: int = 0


class _TensorCodec:
    def __init__(self, shared_memory_root: str | Path) -> None:
        self.root = Path(shared_memory_root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self.root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise WorldlinesPolicyConnectionError(
                "policy shared-memory root must be an owned directory"
            )
        if metadata.st_mode & 0o077:
            os.chmod(self.root, 0o700)

    def encode(self, value: Any) -> tuple[Any, list[Path], _TransferStats]:
        attachments: list[Path] = []
        tensor_count = 0
        tensor_bytes = 0
        shared_count = 0
        shared_bytes = 0

        def visit(item: Any, depth: int) -> Any:
            nonlocal tensor_count, tensor_bytes, shared_count, shared_bytes
            if depth > 64:
                raise WorldlinesPolicyTransportError(
                    "INVALID_PAYLOAD", "container nesting exceeds 64"
                )
            tensor = _as_numpy(item)
            if tensor is not None:
                array = _validated_array(tensor)
                byte_length = int(array.nbytes)
                tensor_count += 1
                tensor_bytes += byte_length
                if tensor_bytes > MAX_TOTAL_TENSOR_BYTES:
                    raise WorldlinesPolicyTransportError(
                        "TENSOR_BUDGET_EXCEEDED",
                        f"tensor bytes exceed {MAX_TOTAL_TENSOR_BYTES}",
                    )
                shared = byte_length >= SHARED_MEMORY_THRESHOLD_BYTES and byte_length > 0
                envelope: dict[str, Any] = {
                    _TENSOR_MARKER: "shm" if shared else "inline",
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "byte_length": byte_length,
                }
                if shared:
                    attachment = self._write_attachment(array)
                    attachments.append(attachment)
                    envelope["name"] = attachment.name
                    shared_count += 1
                    shared_bytes += byte_length
                else:
                    envelope["data"] = array.tobytes(order="C")
                return envelope
            if isinstance(item, Mapping):
                if len(item) > MAX_CONTAINER_ITEMS or any(not isinstance(key, str) for key in item):
                    raise WorldlinesPolicyTransportError("INVALID_PAYLOAD", "mapping is invalid")
                return {key: visit(child, depth + 1) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                if len(item) > MAX_CONTAINER_ITEMS:
                    raise WorldlinesPolicyTransportError("INVALID_PAYLOAD", "array is too large")
                return [visit(child, depth + 1) for child in item]
            if isinstance(item, np.generic):
                return item.item()
            if item is None or isinstance(item, (bool, int, float, str, bytes)):
                return item
            raise WorldlinesPolicyTransportError(
                "INVALID_PAYLOAD", f"unsupported value type {type(item).__name__}"
            )

        try:
            encoded = visit(value, 0)
        except BaseException:
            self.cleanup(attachments)
            raise
        return (
            encoded,
            attachments,
            _TransferStats(
                tensor_count=tensor_count,
                tensor_bytes=tensor_bytes,
                shared_memory_tensor_count=shared_count,
                shared_memory_bytes=shared_bytes,
            ),
        )

    def decode(self, value: Any) -> Any:
        tensor_bytes = 0

        def visit(item: Any, depth: int) -> Any:
            nonlocal tensor_bytes
            if depth > 64:
                raise WorldlinesPolicyTransportError(
                    "INVALID_RESPONSE", "container nesting exceeds 64"
                )
            if isinstance(item, Mapping):
                if _TENSOR_MARKER in item:
                    mode = item.get(_TENSOR_MARKER)
                    if mode != "inline" or set(item) != {
                        _TENSOR_MARKER,
                        "dtype",
                        "shape",
                        "byte_length",
                        "data",
                    }:
                        raise WorldlinesPolicyTransportError(
                            "INVALID_TENSOR", "response tensor envelope is invalid"
                        )
                    try:
                        dtype = np.dtype(item["dtype"])
                        shape = tuple(int(size) for size in item["shape"])
                        byte_length = int(item["byte_length"])
                    except (KeyError, TypeError, ValueError, OverflowError) as exc:
                        raise WorldlinesPolicyTransportError(
                            "INVALID_TENSOR", "response tensor metadata is invalid"
                        ) from exc
                    expected = _tensor_size(dtype, shape)
                    data = item.get("data")
                    if (
                        byte_length != expected
                        or not isinstance(data, bytes)
                        or len(data) != expected
                    ):
                        raise WorldlinesPolicyTransportError(
                            "INVALID_TENSOR", "response tensor bytes do not match metadata"
                        )
                    tensor_bytes += expected
                    if tensor_bytes > MAX_TOTAL_TENSOR_BYTES:
                        raise WorldlinesPolicyTransportError(
                            "TENSOR_BUDGET_EXCEEDED",
                            f"tensor bytes exceed {MAX_TOTAL_TENSOR_BYTES}",
                        )
                    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
                return {key: visit(child, depth + 1) for key, child in item.items()}
            if isinstance(item, list):
                return [visit(child, depth + 1) for child in item]
            if item is None or isinstance(item, (bool, int, float, str, bytes)):
                return item
            raise WorldlinesPolicyTransportError(
                "INVALID_RESPONSE", f"unsupported value type {type(item).__name__}"
            )

        return visit(value, 0)

    @staticmethod
    def cleanup(attachments: Sequence[Path]) -> None:
        for attachment in attachments:
            try:
                attachment.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_attachment(self, array: np.ndarray) -> Path:
        path = self.root / f"wlt-{uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(array).cast("B")
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise WorldlinesPolicyTransportError(
                        "SHARED_MEMORY_WRITE_FAILED", "attachment write made no progress"
                    )
                written += count
            os.fchmod(descriptor, 0o600)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        return path


class _UnixPolicyRpcClient:
    def __init__(self, connection: WorldlinesPolicyConnection, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.connection = connection
        self.timeout_seconds = float(timeout_seconds)
        self.codec = _TensorCodec(connection.shared_memory_root)
        self._socket: socket.socket | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self.last_metrics: dict[str, Any] | None = None

    def call(self, method: str, payload: Mapping[str, Any]) -> Any:
        with self._lock:
            connection = self._connect()
            self._request_id += 1
            request_id = self._request_id
            encoded, attachments, stats = self.codec.encode(dict(payload))
            started = time.perf_counter()
            try:
                request_size = _send_frame(
                    connection,
                    {
                        "request_id": request_id,
                        "token": self.connection.grant_token,
                        "method": method,
                        "deadline_unix_ns": time.time_ns()
                        + int(self.timeout_seconds * 1_000_000_000),
                        "payload": encoded,
                    },
                )
                response, response_size = _receive_frame(connection)
            except TimeoutError as exc:
                self._disconnect()
                raise WorldlinesPolicyTransportError(
                    "DEADLINE_EXCEEDED",
                    f"policy call exceeded {self.timeout_seconds:g} seconds",
                ) from exc
            except (EOFError, ConnectionError, OSError, WorldlinesPolicyTransportError):
                self._disconnect()
                raise
            finally:
                self.codec.cleanup(attachments)
            self.last_metrics = {
                "round_trip_ms": (time.perf_counter() - started) * 1000,
                "request_frame_bytes": request_size,
                "response_frame_bytes": response_size,
                "tensor_count": stats.tensor_count,
                "tensor_bytes": stats.tensor_bytes,
                "shared_memory_tensor_count": stats.shared_memory_tensor_count,
                "shared_memory_bytes": stats.shared_memory_bytes,
            }
            if not isinstance(response, Mapping) or response.get("request_id") != request_id:
                self._disconnect()
                raise WorldlinesPolicyTransportError(
                    "INVALID_RESPONSE", "response request_id does not match"
                )
            if response.get("ok") is not True:
                error = response.get("error")
                if not isinstance(error, Mapping):
                    raise WorldlinesPolicyTransportError(
                        "INVALID_RESPONSE", "service error is malformed"
                    )
                raise WorldlinesPolicyServiceError(
                    str(error.get("code", "UNKNOWN")),
                    str(error.get("message", "")),
                )
            return self.codec.decode(response.get("result"))

    def close(self) -> None:
        with self._lock:
            self._disconnect()

    def _connect(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        deadline = time.monotonic() + min(self.timeout_seconds, 5)
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout_seconds)
            try:
                metadata = Path(self.connection.endpoint_address).lstat()
                if (
                    not stat.S_ISSOCK(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077
                ):
                    raise WorldlinesPolicyConnectionError(
                        "policy endpoint must be an owned private Unix socket"
                    )
                connection.connect(self.connection.endpoint_address)
                self._socket = connection
                return connection
            except FileNotFoundError:
                connection.close()
                if time.monotonic() >= deadline:
                    raise WorldlinesPolicyTransportError(
                        "CONNECT_TIMEOUT", "policy service socket did not appear"
                    ) from None
                time.sleep(0.01)
            except OSError:
                connection.close()
                raise

    def _disconnect(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


class WorldlinesPolicyServiceClient:
    """SDK ``PolicyServiceClient`` backed by a resolved Worldlines deployment."""

    def __init__(
        self,
        connection: WorldlinesPolicyConnection,
        policy: PolicyDeploymentSpec,
        *,
        batch_size: int,
        sequence_ids: Sequence[str],
        seeds: Sequence[int],
        timeout_seconds: float = 30,
    ) -> None:
        if policy.source != "worldlines":
            raise WorldlinesPolicyConnectionError(
                "WorldlinesPolicyServiceClient requires policy.source='worldlines'"
            )
        if connection.deployment_id != policy.deployment_id:
            raise WorldlinesPolicyConnectionError(
                "policy connection deployment does not match robotics job"
            )
        if batch_size <= 0 or batch_size > policy.max_batch_size:
            raise WorldlinesPolicyConnectionError(
                "policy connection batch size exceeds deployment capacity"
            )
        if (
            len(sequence_ids) != batch_size
            or any(
                not isinstance(sequence_id, str) or not sequence_id for sequence_id in sequence_ids
            )
            or len(set(sequence_ids)) != batch_size
        ):
            raise WorldlinesPolicyConnectionError(
                "policy connection requires one unique sequence id per batch lane"
            )
        self.connection = connection
        self.policy = policy
        self.batch_size = batch_size
        self.sequence_ids = list(sequence_ids)
        self.seeds = _policy_seeds(seeds, batch_size)
        self._last_step_ids = [-1] * batch_size
        self._rpc = _UnixPolicyRpcClient(connection, timeout_seconds=timeout_seconds)
        self._closed = False
        try:
            raw_descriptor = self._rpc.call(
                "open",
                {
                    "deployment_id": policy.deployment_id,
                    "session_id": connection.session_id,
                    "batch_size": batch_size,
                    "sequence_ids": self.sequence_ids,
                    "seeds": self.seeds,
                },
            )
            if not isinstance(raw_descriptor, Mapping):
                raise WorldlinesPolicyTransportError(
                    "INVALID_RESPONSE", "policy open descriptor is malformed"
                )
            descriptor = PolicyServiceDescriptor.from_dict(raw_descriptor)
            if descriptor.session_id != connection.session_id:
                raise WorldlinesPolicyConnectionError(
                    "policy open descriptor session does not match the resolved connection"
                )
            if descriptor.transport != WORLDLINES_POLICY_TRANSPORT:
                raise WorldlinesPolicyConnectionError(
                    "policy open descriptor transport does not match the resolved connection"
                )
            self._descriptor = descriptor
        except BaseException:
            with suppress(BaseException):
                self._rpc.call("close", {"session_id": connection.session_id})
            self._rpc.close()
            self._closed = True
            raise

    def describe(self) -> PolicyServiceDescriptor:
        return self._descriptor

    def reset(
        self,
        indices: Sequence[int] | None = None,
        seeds: Sequence[int] | None = None,
    ) -> None:
        normalized_indices = _policy_indices(indices, self.batch_size)
        next_seeds = list(self.seeds) if seeds is None else _policy_seeds(seeds, self.batch_size)
        self._rpc.call(
            "reset",
            {
                "session_id": self.connection.session_id,
                "indices": normalized_indices,
                "seeds": next_seeds,
            },
        )
        targets = range(self.batch_size) if normalized_indices is None else normalized_indices
        for index in targets:
            self.seeds[index] = next_seeds[index]
            self._last_step_ids[index] = -1

    def act(
        self,
        observation: Any,
        *,
        seed: int | Sequence[int] | None = None,
        step_ids: Sequence[int] | None = None,
    ) -> ActionChunk:
        normalized_steps = (
            [step_id + 1 for step_id in self._last_step_ids]
            if step_ids is None
            else _policy_step_ids(step_ids, self.batch_size)
        )
        if any(
            step_id <= previous
            for step_id, previous in zip(normalized_steps, self._last_step_ids, strict=True)
        ):
            raise WorldlinesPolicyConnectionError(
                "policy inference step ids must increase for every batch lane"
            )
        next_seeds = list(self.seeds)
        if seed is not None:
            if isinstance(seed, int) and not isinstance(seed, bool) and self.batch_size == 1:
                next_seeds = [seed]
            elif isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
                next_seeds = _policy_seeds(seed, self.batch_size)
            else:
                raise WorldlinesPolicyConnectionError(
                    "vector policy inference requires a seed array"
                )
        result = self._rpc.call(
            "infer",
            {
                "session_id": self.connection.session_id,
                "observation": _collate_observation(
                    observation,
                    self.policy,
                    batch_size=self.batch_size,
                ),
                "step_ids": normalized_steps,
                "seeds": next_seeds,
                "reset_mask": [False] * self.batch_size,
            },
        )
        if not isinstance(result, Mapping):
            raise WorldlinesPolicyTransportError(
                "INVALID_RESPONSE", "policy action chunk is malformed"
            )
        chunk_data = dict(result)
        auxiliary = dict(chunk_data.get("auxiliary") or {})
        if self._rpc.last_metrics is not None:
            auxiliary["transport"] = dict(self._rpc.last_metrics)
        chunk_data["auxiliary"] = auxiliary
        chunk = ActionChunk.from_dict(chunk_data)
        self.seeds = next_seeds
        self._last_step_ids = normalized_steps
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            self._rpc.call("close", {"session_id": self.connection.session_id})
        except BaseException as exc:  # noqa: BLE001 - close the owned socket regardless
            failure = exc
        self._rpc.close()
        self._closed = True
        if failure is not None:
            raise failure


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorldlinesPolicyConnectionError(
            f"policy connection.{where} must be a non-empty string"
        )
    return value


def _absolute_normalized_path(value: Any, where: str) -> str:
    path = _string(value, where)
    if (
        not path.startswith("/")
        or "\0" in path
        or ".." in Path(path).parts
        or os.path.normpath(path) != path
    ):
        raise WorldlinesPolicyConnectionError(
            f"policy connection.{where} must be an absolute normalized path"
        )
    return path


def _policy_indices(value: Sequence[int] | None, batch_size: int) -> list[int] | None:
    if value is None:
        return None
    normalized = list(value)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < batch_size
        for index in normalized
    ):
        raise WorldlinesPolicyConnectionError(
            f"policy reset indices must be integers in [0, {batch_size})"
        )
    if len(set(normalized)) != len(normalized):
        raise WorldlinesPolicyConnectionError("policy reset indices cannot contain duplicates")
    return normalized


def _policy_seeds(value: Sequence[int], batch_size: int) -> list[int]:
    if len(value) != batch_size or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in value
    ):
        raise WorldlinesPolicyConnectionError(
            "policy request requires one integer seed per batch lane"
        )
    return list(value)


def _policy_step_ids(value: Sequence[int], batch_size: int) -> list[int]:
    if len(value) != batch_size or any(
        isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0
        for step_id in value
    ):
        raise WorldlinesPolicyConnectionError(
            "policy inference requires one non-negative step id per batch lane"
        )
    return list(value)


def _collate_observation(
    observation: Any,
    policy: PolicyDeploymentSpec,
    *,
    batch_size: int,
) -> Any:
    if isinstance(observation, Mapping):
        return observation
    if (
        not isinstance(observation, Sequence)
        or isinstance(observation, (str, bytes, bytearray))
        or len(observation) != batch_size
        or any(not isinstance(item, Mapping) for item in observation)
    ):
        raise WorldlinesPolicyConnectionError(
            f"policy observation batch must contain {batch_size} objects"
        )
    expected_names = set(policy.observation_schema)
    if any(set(item) != expected_names for item in observation):
        raise WorldlinesPolicyConnectionError(
            "policy observation fields do not match the deployment schema"
        )
    collated: dict[str, Any] = {}
    for name, spec in policy.observation_schema.items():
        values = [item[name] for item in observation]
        if spec.dtype.lower() in {"str", "string", "utf8"}:
            if tuple(spec.shape) != (1,):
                raise WorldlinesPolicyConnectionError(
                    f"text observation {name!r} requires singleton shape [1]"
                )
            normalized_text: list[str] = []
            for value in values:
                if isinstance(value, str):
                    normalized_text.append(value)
                elif (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    and len(value) == 1
                    and isinstance(value[0], str)
                ):
                    normalized_text.append(value[0])
                else:
                    raise WorldlinesPolicyConnectionError(
                        f"text observation {name!r} must contain one string per lane"
                    )
            collated[name] = normalized_text
            continue
        arrays: list[np.ndarray] = []
        expected_shape = tuple(spec.shape)
        for value in values:
            tensor = _as_numpy(value)
            array = np.asarray(tensor if tensor is not None else value)
            if array.shape == () and int(np.prod(expected_shape)) == 1:
                array = array.reshape(expected_shape)
            if tuple(array.shape) != expected_shape:
                raise WorldlinesPolicyConnectionError(
                    f"policy observation {name!r} does not match shape {expected_shape}"
                )
            arrays.append(array)
        collated[name] = np.stack(arrays, axis=0)
    return collated


def _as_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if callable(detach) and callable(cpu):
        tensor = value.detach().cpu()
        numpy_method = getattr(tensor, "numpy", None)
        if callable(numpy_method):
            try:
                return numpy_method()
            except TypeError:
                float_method = getattr(tensor, "float", None)
                if callable(float_method):
                    return float_method().numpy()
                raise
    return None


def _validated_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    _tensor_size(array.dtype, tuple(array.shape))
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def _tensor_size(dtype: np.dtype[Any], shape: tuple[int, ...]) -> int:
    if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f"}:
        raise WorldlinesPolicyTransportError(
            "INVALID_TENSOR", f"tensor dtype {dtype} is unsupported"
        )
    if len(shape) > 8 or any(size < 0 for size in shape):
        raise WorldlinesPolicyTransportError("INVALID_TENSOR", "tensor shape is invalid")
    expected = int(dtype.itemsize)
    for size in shape:
        expected *= size
        if expected > MAX_TENSOR_BYTES:
            raise WorldlinesPolicyTransportError(
                "TENSOR_TOO_LARGE", f"tensor exceeds {MAX_TENSOR_BYTES} bytes"
            )
    return expected


def _pack_message(value: Any) -> bytes:
    try:
        payload = msgpack.packb(value, use_bin_type=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorldlinesPolicyTransportError("INVALID_PAYLOAD", str(exc)) from exc
    if len(payload) > MAX_FRAME_BYTES:
        raise WorldlinesPolicyTransportError(
            "FRAME_TOO_LARGE", f"frame exceeds {MAX_FRAME_BYTES} bytes"
        )
    return payload


def _unpack_message(payload: bytes) -> Any:
    try:
        return msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            max_str_len=MAX_FRAME_BYTES,
            max_bin_len=MAX_FRAME_BYTES,
            max_array_len=MAX_CONTAINER_ITEMS,
            max_map_len=MAX_CONTAINER_ITEMS,
        )
    except (
        ValueError,
        TypeError,
        msgpack.ExtraData,
        msgpack.FormatError,
        msgpack.StackError,
    ) as exc:
        raise WorldlinesPolicyTransportError("INVALID_FRAME", str(exc)) from exc


def _send_frame(connection: socket.socket, value: Any) -> int:
    payload = _pack_message(value)
    connection.sendall(_FRAME_HEADER.pack(len(payload)))
    connection.sendall(payload)
    return len(payload)


def _receive_frame(connection: socket.socket) -> tuple[Any, int]:
    header = _receive_exact(connection, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size > MAX_FRAME_BYTES:
        raise WorldlinesPolicyTransportError(
            "FRAME_TOO_LARGE", f"frame exceeds {MAX_FRAME_BYTES} bytes"
        )
    return _unpack_message(_receive_exact(connection, size)), size


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("policy service closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "WORLDLINES_POLICY_CONNECTION_VERSION",
    "WORLDLINES_POLICY_TRANSPORT",
    "WorldlinesPolicyConnection",
    "WorldlinesPolicyConnectionError",
    "WorldlinesPolicyServiceClient",
    "WorldlinesPolicyServiceError",
    "WorldlinesPolicyTransportError",
]
