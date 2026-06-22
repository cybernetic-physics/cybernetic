"""Internal client holder for managing AsyncCybernetics clients."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import threading
import time
import traceback
import weakref
from collections.abc import Coroutine, Generator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from typing import Any, Awaitable, Callable, Literal, TypeVar

import httpx

from cybernetics import types
from cybernetics._client import AsyncCybernetics
from cybernetics._exceptions import APIConnectionError, APIStatusError
from cybernetics._version import __version__ as worldlines_sdk_version
from cybernetics.lib.async_worldlines_provider import AsyncCyberneticsProvider
from cybernetics.lib.client_connection_pool_type import ClientConnectionPoolType
from cybernetics.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from cybernetics.lib.telemetry import Telemetry, init_telemetry, is_user_error
from cybernetics.lib.telemetry_provider import TelemetryProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")
SampleDispatchProfile = Literal["auto", "local_single_gpu", "workstation", "remote_large_backend"]
SampleDispatchRequestKind = Literal["sample", "compute_logprobs"]

MAX_REQUESTS_PER_HTTPX_CLIENT = 50
MAX_CONNECTION_ERROR_RETRIES = 16


class ClientConnectionPool:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        max_requests_per_client: int,
        constructor_kwargs: dict[str, Any],
    ):
        self._loop = loop
        self._max_requests_per_client = max_requests_per_client
        self._constructor_kwargs = constructor_kwargs
        self._clients: list[AsyncCybernetics] = []
        self._client_active_refcount: list[int] = []
        self._connection_error_retries_remaining: int = MAX_CONNECTION_ERROR_RETRIES

    @contextmanager
    def aclient(self) -> Generator[AsyncCybernetics, None, None]:
        assert _current_loop() is self._loop, (
            "AsyncCybernetics client called from incorrect event loop"
        )
        client_idx = -1
        for i, ref_count in enumerate(self._client_active_refcount):
            if ref_count < self._max_requests_per_client:
                client_idx = i
                break
        if client_idx == -1:
            self._clients.append(AsyncCybernetics(**self._constructor_kwargs))
            client_idx = len(self._clients) - 1
            self._client_active_refcount.append(0)

        self._client_active_refcount[client_idx] += 1
        try:
            yield self._clients[client_idx]
            if self._connection_error_retries_remaining < MAX_CONNECTION_ERROR_RETRIES:
                self._connection_error_retries_remaining += 1
        except APIStatusError as e:
            # This indicates request rejected by Cloudflare. Reset the connection and retry
            if e.status_code == 400 and e.response.headers.get("content-length", "0") == "0":
                # Ensure a new connection gets opened
                self._clients[client_idx] = AsyncCybernetics(**self._constructor_kwargs)
                if self._connection_error_retries_remaining > 0:
                    self._connection_error_retries_remaining -= 1
                    raise APIConnectionError(request=e.request) from e
            raise e
        finally:
            self._client_active_refcount[client_idx] -= 1


class InternalClientHolderThreadSingleton:
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started: bool = False
        self._lifecycle_lock: threading.Lock = threading.Lock()

    def _ensure_started(self):
        if self._started:
            return

        with self._lifecycle_lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
            self._thread.start()
            self._started = True

    def _background_thread_func(self):
        assert self._loop is not None, "Loop must not be None"
        self._loop.run_forever()

    def _set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Inject an external event loop (e.g. the sidecar subprocess loop).

        Must be called before any InternalClientHolder is created.
        Prevents _ensure_started from spawning a background thread — the
        caller's loop is used directly.
        """
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("Cannot set_loop after singleton has started")
            self._loop = loop
            self._started = True  # prevent _ensure_started from creating a thread

    def get_loop(self) -> asyncio.AbstractEventLoop:
        self._ensure_started()
        assert self._loop is not None, "Loop must not be None"
        return self._loop


_internal_client_holder_thread_singleton = InternalClientHolderThreadSingleton()


class _ShadowHolderSingleton:
    """Singleton to cache shadow InternalClientHolders by constructor args."""

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        # Key is (session_id, json-serialized kwargs)
        self._cache: dict[tuple[str, str], weakref.ref[InternalClientHolder]] = {}

    def get_or_create(self, session_id: str, kwargs: dict[str, Any]) -> InternalClientHolder:
        key = (session_id, json.dumps(kwargs, sort_keys=True))
        with self._lock:
            if key in self._cache:
                holder = self._cache[key]()
                if holder is not None:
                    return holder
            holder = InternalClientHolder(session_id=session_id, **kwargs)
            self._cache[key] = weakref.ref(holder)
            return holder


_shadow_holder_singleton = _ShadowHolderSingleton()


def _create_session_request(
    *,
    tags: list[str],
    user_metadata: dict[str, str] | None,
    sdk_version: str,
    project_id: str | None,
) -> types.CreateSessionRequest:
    request_kwargs: dict[str, Any] = {
        "tags": tags,
        "user_metadata": user_metadata or {},
        "sdk_version": sdk_version,
    }
    if project_id is not None:
        request_kwargs["project_id"] = project_id
    return types.CreateSessionRequest(**request_kwargs)


class BytesSemaphore:
    def __init__(self, max_bytes: int):
        self._bytes: int = max_bytes
        self._condition: asyncio.Condition = asyncio.Condition()
        self._release_task: asyncio.Task[None] | None = None

    async def _release(self):
        async with self._condition:
            self._condition.notify_all()

    @asynccontextmanager
    async def acquire(self, bytes: int):
        async with self._condition:
            while self._bytes < 0:
                await self._condition.wait()
        self._bytes -= bytes

        try:
            yield
        finally:
            self._bytes += bytes
            # Make sure the release task is never cancelled.
            self._release_task = asyncio.create_task(self._release())


class CountLimiter:
    def __init__(self):
        self._condition: asyncio.Condition = asyncio.Condition()
        self._inflight: int = 0

    def current_inflight(self) -> int:
        return self._inflight

    @asynccontextmanager
    async def acquire(self, limit: int):
        async with self._condition:
            while self._inflight >= limit:
                await self._condition.wait()
            self._inflight += 1

        try:
            yield
        finally:
            async with self._condition:
                self._inflight -= 1
                self._condition.notify_all()


class _SharedCountLimiterPoolSingleton:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._dispatch_limiters: dict[str, CountLimiter] = {}
        self._outstanding_limiters: dict[str, CountLimiter] = {}

    def get_dispatch_limiter(self, key: str) -> CountLimiter:
        with self._lock:
            return self._dispatch_limiters.setdefault(key, CountLimiter())

    def get_outstanding_limiter(self, key: str) -> CountLimiter:
        with self._lock:
            return self._outstanding_limiters.setdefault(key, CountLimiter())


class _SharedSampleBackoffState:
    def __init__(self) -> None:
        self.backoff_until: float | None = None
        self.backoff_streak: int = 0
        self.backoff_last_requested_at: float | None = None


class _SharedSampleBackoffPoolSingleton:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._states: dict[str, _SharedSampleBackoffState] = {}

    def get_state(self, key: str) -> _SharedSampleBackoffState:
        with self._lock:
            return self._states.setdefault(key, _SharedSampleBackoffState())


_shared_count_limiter_pool_singleton = _SharedCountLimiterPoolSingleton()
_shared_sample_backoff_pool_singleton = _SharedSampleBackoffPoolSingleton()


class _SharedSampleDispatcherPoolSingleton:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._dispatch_semaphores: dict[str, asyncio.Semaphore] = {}
        self._dispatch_bytes_semaphores: dict[str, BytesSemaphore] = {}

    def get_dispatch_semaphore(self, key: str) -> asyncio.Semaphore:
        with self._lock:
            return self._dispatch_semaphores.setdefault(key, asyncio.Semaphore(400))

    def get_dispatch_bytes_semaphore(self, key: str) -> BytesSemaphore:
        with self._lock:
            return self._dispatch_bytes_semaphores.setdefault(key, BytesSemaphore(5 * 1024 * 1024))


class _SharedSampleCoalescingWindowPoolSingleton:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._deadlines: dict[str, float] = {}

    def shared_wait_seconds(self, key: str, *, window_seconds: float, now: float) -> float:
        with self._lock:
            deadline = self._deadlines.get(key)
            if deadline is None or deadline <= now:
                deadline = now + window_seconds
                self._deadlines[key] = deadline
            return max(0.0, deadline - now)


_shared_sample_dispatcher_pool_singleton = _SharedSampleDispatcherPoolSingleton()
_shared_sample_coalescing_window_pool_singleton = _SharedSampleCoalescingWindowPoolSingleton()


class InternalClientHolder(AsyncCyberneticsProvider, TelemetryProvider):
    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        *,
        session_id: str | None = None,
        sample_dispatch_profile: SampleDispatchProfile = "auto",
        **kwargs: Any,
    ) -> None:
        self._constructor_kwargs = kwargs
        self._sample_dispatch_profile: SampleDispatchProfile = sample_dispatch_profile
        self._loop: asyncio.AbstractEventLoop = _internal_client_holder_thread_singleton.get_loop()
        self._client_pools: dict[ClientConnectionPoolType, ClientConnectionPool] = {}
        self._sample_backoff_until: float | None = None
        self._sample_dispatch_semaphore: asyncio.Semaphore | None = None
        self._sample_dispatch_throttled_limiter: CountLimiter = CountLimiter()
        self._sample_dispatch_bytes_semaphore: BytesSemaphore | None = None
        self._sample_dispatch_profile_lock: asyncio.Lock = asyncio.Lock()
        self._sample_dispatch_profile_loaded: bool = False
        self._sample_dispatch_profile_limit: int | None = None
        self._sample_dispatch_profile_limiter: CountLimiter = CountLimiter()
        self._sample_outstanding_profile_limit: int | None = None
        self._sample_outstanding_profile_limiter: CountLimiter = CountLimiter()
        self._sample_backoff_streak: int = 0
        self._sample_backoff_last_requested_at: float | None = None
        self._inflight_response_bytes_semaphore: BytesSemaphore = BytesSemaphore(5 * 1024 * 1024)
        self._training_client_lock: threading.Lock = threading.Lock()
        self._server_capabilities_cache: types.GetServerCapabilitiesResponse | None = None

        if session_id is not None:
            # Shadow mode: reuse existing session, can't create new clients
            self._session_id: str = session_id
            self._training_client_counter: int | None = None
            self._sampling_client_counter: int | None = None
        else:
            # Normal mode: create new session.
            # This blocks on .result() — must NOT be called from the event
            # loop thread (e.g. inside the sidecar subprocess).  Shadow
            # holders (session_id is not None) skip this path.
            if self._loop.is_running() and _current_loop() is self._loop:
                raise RuntimeError(
                    "Cannot create a new session from the event loop thread. "
                    "Use session_id= to create a shadow holder instead."
                )
            self._session_id = self.run_coroutine_threadsafe(
                self._create_session(user_metadata=user_metadata, project_id=project_id)
            ).result()
            self._training_client_counter = 0
            self._sampling_client_counter = 0

        if self._loop.is_running() and _current_loop() is self._loop:
            # Already on the event loop thread — .result() would deadlock.
            # Create the heartbeat task directly instead of via run_coroutine_threadsafe.
            self._session_heartbeat_task: asyncio.Task[None] = asyncio.create_task(
                self._session_heartbeat(self._session_id)
            )
        else:
            self._session_heartbeat_task = self.run_coroutine_threadsafe(
                self._start_heartbeat()
            ).result()
        self._telemetry: Telemetry | None = init_telemetry(self, session_id=self._session_id)

    @classmethod
    def get_shadow_holder(cls, session_id: str, kwargs: dict[str, Any]) -> InternalClientHolder:
        """Get or create a shadow holder from the singleton cache."""
        return _shadow_holder_singleton.get_or_create(session_id, kwargs)

    def _shared_sample_dispatch_semaphore(self) -> asyncio.Semaphore:
        semaphore = getattr(self, "_sample_dispatch_semaphore", None)
        if semaphore is not None:
            return semaphore
        if self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            semaphore = _shared_sample_dispatcher_pool_singleton.get_dispatch_semaphore(
                self._shared_sample_dispatch_scope_key()
            )
        else:
            semaphore = asyncio.Semaphore(400)
        self._sample_dispatch_semaphore = semaphore
        return semaphore

    def _shared_sample_dispatch_bytes_semaphore(self) -> BytesSemaphore:
        semaphore = getattr(self, "_sample_dispatch_bytes_semaphore", None)
        if semaphore is not None:
            return semaphore
        if self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            semaphore = _shared_sample_dispatcher_pool_singleton.get_dispatch_bytes_semaphore(
                self._shared_sample_dispatch_scope_key()
            )
        else:
            semaphore = BytesSemaphore(5 * 1024 * 1024)
        self._sample_dispatch_bytes_semaphore = semaphore
        return semaphore

    @asynccontextmanager
    async def _sample_dispatch_count_rate_limit(self):
        async with self._shared_sample_dispatch_semaphore():
            yield

    def _sample_dispatch_throttled_limit(
        self,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ) -> int:
        if self._sample_dispatch_profile == "remote_large_backend":
            return 10

        throttled_limit: int
        if self._sample_dispatch_profile == "local_single_gpu":
            throttled_limit = min(self._sample_dispatch_profile_limit or 2, 2)
        elif self._sample_dispatch_profile == "workstation":
            throttled_limit = min(self._sample_dispatch_profile_limit or 4, 4)
        elif (
            self._sample_dispatch_profile == "auto"
            and self._sample_dispatch_profile_limit is not None
        ):
            throttled_limit = min(self._sample_dispatch_profile_limit, 4)
        else:
            return 10

        if request_kind == "compute_logprobs":
            throttled_limit = max(1, throttled_limit - 1)

        backoff_streak = self._current_sample_backoff_streak()
        if backoff_streak >= 4:
            return max(1, throttled_limit - 1)
        if backoff_streak >= 3 and throttled_limit > 2:
            return throttled_limit - 1
        return throttled_limit

    @asynccontextmanager
    async def _sample_dispatch_count_throttled_rate_limit(
        self,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ):
        async with self._sample_dispatch_throttled_limiter.acquire(
            self._sample_dispatch_throttled_limit(request_kind=request_kind)
        ):
            yield

    def get_sample_dispatch_profile(self) -> SampleDispatchProfile:
        return self._sample_dispatch_profile

    @staticmethod
    def _is_local_sample_dispatch_profile(profile: SampleDispatchProfile) -> bool:
        return profile in {"auto", "local_single_gpu", "workstation"}

    def _sampling_dispatch_limit_from_capabilities(
        self,
        capabilities: types.GetServerCapabilitiesResponse | None,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ) -> int | None:
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return None
        if capabilities is None:
            return None
        if not capabilities.supports_local_client_dispatch_profiles:
            return None
        if capabilities.recommended_max_outstanding_sampling_requests is None:
            return None
        resolved_limit = max(1, capabilities.recommended_max_outstanding_sampling_requests)
        active_limit = capabilities.max_active_requests_per_subject
        if active_limit is not None:
            resolved_limit = min(resolved_limit, max(1, active_limit))
        if request_kind == "compute_logprobs":
            queue_limit = capabilities.max_queued_compute_logprobs_requests
            if queue_limit is not None:
                resolved_limit = min(resolved_limit, max(1, queue_limit))
        return resolved_limit

    def _sampling_outstanding_limit_from_capabilities(
        self,
        capabilities: types.GetServerCapabilitiesResponse | None,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ) -> int | None:
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return None
        if capabilities is None:
            return None
        if not capabilities.supports_local_client_dispatch_profiles:
            return None
        base_limit = capabilities.max_queued_requests_per_subject
        if base_limit is None:
            return None
        resolved_limit = max(1, base_limit)
        if request_kind == "compute_logprobs":
            queue_limit = capabilities.max_queued_compute_logprobs_requests
            if queue_limit is not None:
                resolved_limit = min(resolved_limit, max(1, queue_limit))
        return resolved_limit

    def _scheduler_state_subject_pending_count(
        self, scheduler_state: types.GetSchedulerStateResponse | None
    ) -> int:
        if scheduler_state is None or scheduler_state.queue_depths is None:
            return 0
        by_subject = scheduler_state.queue_depths.by_subject
        if by_subject is None:
            return 0
        return max(0, by_subject.get(self._session_id, 0))

    def _scheduler_state_matching_pending_count(
        self,
        scheduler_state: types.GetSchedulerStateResponse | None,
        *,
        request_kind: SampleDispatchRequestKind,
    ) -> int:
        if scheduler_state is None or scheduler_state.queue_depths is None:
            return 0
        by_request_class = scheduler_state.queue_depths.by_request_class
        if by_request_class is not None:
            return max(0, by_request_class.get(request_kind, 0))
        by_queue_class = scheduler_state.queue_depths.by_queue_class
        if by_queue_class is None:
            return 0
        queue_class = (
            "compute_logprobs" if request_kind == "compute_logprobs" else "interactive_sampling"
        )
        return max(0, by_queue_class.get(queue_class, 0))

    def _effective_sampling_outstanding_limit(
        self,
        *,
        base_limit: int,
        scheduler_state: types.GetSchedulerStateResponse | None,
        request_kind: SampleDispatchRequestKind = "sample",
    ) -> int:
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return base_limit
        effective_limit = base_limit
        pending_for_subject = self._scheduler_state_subject_pending_count(scheduler_state)
        if pending_for_subject > 0:
            effective_limit = max(1, effective_limit - pending_for_subject)
        if request_kind == "compute_logprobs":
            matching_pending = self._scheduler_state_matching_pending_count(
                scheduler_state,
                request_kind=request_kind,
            )
            if matching_pending > 0:
                effective_limit = max(1, effective_limit - matching_pending)
        return effective_limit

    def _effective_sample_dispatch_limit(
        self,
        *,
        base_limit: int,
        scheduler_state: types.GetSchedulerStateResponse | None,
        request_kind: SampleDispatchRequestKind,
    ) -> int:
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return base_limit
        matching_pending = self._scheduler_state_matching_pending_count(
            scheduler_state,
            request_kind=request_kind,
        )
        if matching_pending <= 0:
            return base_limit
        return max(1, base_limit - matching_pending)

    def _request_kind_overload_behavior_from_capabilities(
        self,
        capabilities: types.GetServerCapabilitiesResponse | None,
        *,
        request_kind: SampleDispatchRequestKind,
    ) -> str | None:
        if capabilities is None:
            return None
        if request_kind == "compute_logprobs":
            return capabilities.compute_logprobs_overload_behavior
        return capabilities.sampling_overload_behavior

    async def sample_request_extra_headers(
        self,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ) -> dict[str, str]:
        if request_kind == "sample":
            return {"X-Cybernetics-Sampling-Backpressure": "1"}

        capabilities = self._server_capabilities_cache
        if capabilities is None:
            capabilities = await self.get_server_capabilities_async()
        overload_behavior = self._request_kind_overload_behavior_from_capabilities(
            capabilities,
            request_kind=request_kind,
        )
        if overload_behavior == "best_effort":
            return {}
        return {"X-Cybernetics-Sampling-Backpressure": "1"}

    async def _resolve_sample_dispatch_profile_limit(self) -> int | None:
        if self._sample_dispatch_profile_loaded:
            return self._sample_dispatch_profile_limit

        async with self._sample_dispatch_profile_lock:
            if self._sample_dispatch_profile_loaded:
                return self._sample_dispatch_profile_limit

            capabilities = self._server_capabilities_cache
            if capabilities is None:
                capabilities = await self.get_server_capabilities_async()
            self._sample_dispatch_profile_limit = self._sampling_dispatch_limit_from_capabilities(
                capabilities
            )
            self._sample_outstanding_profile_limit = (
                self._sampling_outstanding_limit_from_capabilities(capabilities)
            )
            self._sample_dispatch_profile_loaded = True
            return self._sample_dispatch_profile_limit

    def _sample_coalescing_window_seconds_from_capabilities(
        self,
        capabilities: types.GetServerCapabilitiesResponse | None,
        *,
        request_kind: SampleDispatchRequestKind,
    ) -> float:
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return 0.0
        if capabilities is None:
            return 0.0
        if not capabilities.supports_local_client_dispatch_profiles:
            return 0.0
        if request_kind == "compute_logprobs":
            if capabilities.compute_logprobs_coalescing_window_ms is None:
                return 0.0
            return max(0.0, capabilities.compute_logprobs_coalescing_window_ms / 1000.0)
        if capabilities.sampling_coalescing_window_ms is None:
            return 0.0
        return max(0.0, capabilities.sampling_coalescing_window_ms / 1000.0)

    async def _maybe_wait_for_sample_coalescing_window(
        self, *, request_kind: SampleDispatchRequestKind
    ) -> None:
        if self._sample_backoff_requested_recently():
            return
        if not self._sample_dispatch_profile_loaded:
            await self._resolve_sample_dispatch_profile_limit()
        coalescing_window_seconds = self._sample_coalescing_window_seconds_from_capabilities(
            getattr(self, "_server_capabilities_cache", None),
            request_kind=request_kind,
        )
        if coalescing_window_seconds <= 0.0:
            return
        try:
            scheduler_state = await self.get_scheduler_state_async()
        except Exception:
            scheduler_state = None
        if (
            self._scheduler_state_matching_pending_count(
                scheduler_state,
                request_kind=request_kind,
            )
            > 0
        ):
            return
        wait_seconds = self._shared_sample_coalescing_wait_seconds(
            request_kind=request_kind,
            window_seconds=coalescing_window_seconds,
        )
        if wait_seconds <= 0.0:
            return
        await asyncio.sleep(wait_seconds)

    @asynccontextmanager
    async def _sample_dispatch_profile_rate_limit(self, limit: int):
        async with self._shared_sample_dispatch_profile_limiter().acquire(limit):
            yield

    async def _current_sample_dispatch_limit(
        self,
        *,
        request_kind: SampleDispatchRequestKind,
    ) -> int | None:
        capabilities = getattr(self, "_server_capabilities_cache", None)
        base_limit = self._sampling_dispatch_limit_from_capabilities(
            capabilities,
            request_kind=request_kind,
        )
        if base_limit is None:
            base_limit = getattr(self, "_sample_dispatch_profile_limit", None)
        if base_limit is None and hasattr(self, "get_server_capabilities_async"):
            try:
                capabilities = await self.get_server_capabilities_async()
            except Exception:
                capabilities = None
            base_limit = self._sampling_dispatch_limit_from_capabilities(
                capabilities,
                request_kind=request_kind,
            )
        if base_limit is None:
            return None
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return base_limit
        try:
            scheduler_state = await self.get_scheduler_state_async()
        except Exception:
            return base_limit
        return self._effective_sample_dispatch_limit(
            base_limit=base_limit,
            scheduler_state=scheduler_state,
            request_kind=request_kind,
        )

    @asynccontextmanager
    async def _sample_outstanding_profile_rate_limit(self, limit: int):
        async with self._shared_sample_outstanding_profile_limiter().acquire(limit):
            yield

    def _shared_sample_dispatch_scope_key(self) -> str:
        constructor_kwargs = getattr(self, "_constructor_kwargs", None)
        if constructor_kwargs is None:
            return json.dumps(
                {
                    "holder_id": id(self),
                    "sample_dispatch_profile": self._sample_dispatch_profile,
                },
                sort_keys=True,
            )
        base_url = constructor_kwargs.get("base_url")
        return json.dumps(
            {
                "base_url": base_url,
                "sample_dispatch_profile": self._sample_dispatch_profile,
            },
            sort_keys=True,
        )

    def _shared_sample_dispatch_profile_limiter(self) -> CountLimiter:
        limiter = getattr(self, "_sample_dispatch_profile_limiter", None)
        if limiter is not None:
            return limiter
        limiter = _shared_count_limiter_pool_singleton.get_dispatch_limiter(
            self._shared_sample_dispatch_scope_key()
        )
        self._sample_dispatch_profile_limiter = limiter
        return limiter

    def _shared_sample_coalescing_wait_seconds(
        self,
        *,
        request_kind: SampleDispatchRequestKind,
        window_seconds: float,
    ) -> float:
        if window_seconds <= 0.0:
            return 0.0
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return window_seconds
        now = time.monotonic()
        key = json.dumps(
            {
                "scope": self._shared_sample_dispatch_scope_key(),
                "request_kind": request_kind,
            },
            sort_keys=True,
        )
        return _shared_sample_coalescing_window_pool_singleton.shared_wait_seconds(
            key,
            window_seconds=window_seconds,
            now=now,
        )

    def _shared_sample_outstanding_profile_limiter(self) -> CountLimiter:
        limiter = getattr(self, "_sample_outstanding_profile_limiter", None)
        if limiter is not None:
            return limiter
        limiter = _shared_count_limiter_pool_singleton.get_outstanding_limiter(
            self._shared_sample_dispatch_scope_key()
        )
        self._sample_outstanding_profile_limiter = limiter
        return limiter

    async def _current_sample_outstanding_limit(
        self,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ) -> int | None:
        capabilities = self._server_capabilities_cache
        base_limit = self._sampling_outstanding_limit_from_capabilities(
            capabilities,
            request_kind=request_kind,
        )
        if base_limit is None:
            base_limit = getattr(self, "_sample_outstanding_profile_limit", None)
        if base_limit is None and hasattr(self, "get_server_capabilities_async"):
            try:
                capabilities = await self.get_server_capabilities_async()
            except Exception:
                capabilities = None
            base_limit = self._sampling_outstanding_limit_from_capabilities(
                capabilities,
                request_kind=request_kind,
            )
        if base_limit is None:
            return None
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return base_limit
        try:
            scheduler_state = await self.get_scheduler_state_async()
        except Exception:
            return base_limit
        return self._effective_sampling_outstanding_limit(
            base_limit=base_limit,
            scheduler_state=scheduler_state,
            request_kind=request_kind,
        )

    def _sample_local_outstanding_pressure_backoff_seconds(self) -> float:
        outstanding_limiter = getattr(self, "_sample_outstanding_profile_limiter", None)
        if outstanding_limiter is None:
            return 0.0
        outstanding_inflight = outstanding_limiter.current_inflight()
        if outstanding_inflight <= 1:
            return 0.0
        extra_outstanding = outstanding_inflight - 1
        return min(1.5, 0.25 * extra_outstanding)

    def _shared_sample_backoff_state(self) -> _SharedSampleBackoffState | None:
        constructor_kwargs = getattr(self, "_constructor_kwargs", None)
        if constructor_kwargs is None:
            return None
        return _shared_sample_backoff_pool_singleton.get_state(
            self._shared_sample_dispatch_scope_key()
        )

    def _current_sample_backoff_until(self) -> float | None:
        shared_state = self._shared_sample_backoff_state()
        if shared_state is not None and shared_state.backoff_until is not None:
            return shared_state.backoff_until
        return getattr(self, "_sample_backoff_until", None)

    def _current_sample_backoff_streak(self) -> int:
        shared_state = self._shared_sample_backoff_state()
        if shared_state is not None and shared_state.backoff_last_requested_at is not None:
            return shared_state.backoff_streak
        return getattr(self, "_sample_backoff_streak", 0)

    def _current_sample_backoff_last_requested_at(self) -> float | None:
        shared_state = self._shared_sample_backoff_state()
        if shared_state is not None and shared_state.backoff_last_requested_at is not None:
            return shared_state.backoff_last_requested_at
        return getattr(self, "_sample_backoff_last_requested_at", None)

    def _set_sample_backoff_state(
        self,
        *,
        backoff_until: float | None,
        backoff_streak: int,
        backoff_last_requested_at: float | None,
    ) -> None:
        self._sample_backoff_until = backoff_until
        self._sample_backoff_streak = backoff_streak
        self._sample_backoff_last_requested_at = backoff_last_requested_at
        shared_state = self._shared_sample_backoff_state()
        if shared_state is None:
            return
        shared_state.backoff_until = backoff_until
        shared_state.backoff_streak = backoff_streak
        shared_state.backoff_last_requested_at = backoff_last_requested_at

    def _sample_queue_state_backoff_seconds(self, queue_state: str) -> float:
        now = time.monotonic()
        backoff_last_requested_at = self._current_sample_backoff_last_requested_at()
        backoff_streak = self._current_sample_backoff_streak()
        if backoff_last_requested_at is None or now - backoff_last_requested_at > 5.0:
            backoff_streak = 0
        backoff_last_requested_at = now
        backoff_streak = min(backoff_streak + 1, 4)
        self._set_sample_backoff_state(
            backoff_until=self._current_sample_backoff_until(),
            backoff_streak=backoff_streak,
            backoff_last_requested_at=backoff_last_requested_at,
        )

        if self._sample_dispatch_profile == "local_single_gpu":
            base_seconds = 2.5 if queue_state == "paused_capacity" else 1.5
        elif self._sample_dispatch_profile == "workstation":
            base_seconds = 1.5 if queue_state == "paused_capacity" else 0.75
        elif self._sample_dispatch_profile == "remote_large_backend":
            base_seconds = 1.0
        elif self._sample_dispatch_profile_limit is not None:
            base_seconds = 2.0 if queue_state == "paused_capacity" else 1.0
        else:
            base_seconds = 1.0

        if self._sample_dispatch_profile in {"local_single_gpu", "workstation"} or (
            self._sample_dispatch_profile == "auto"
            and self._sample_dispatch_profile_limit is not None
        ):
            base_seconds += 0.5 * (backoff_streak - 1)
            base_seconds += self._sample_local_outstanding_pressure_backoff_seconds()

        return base_seconds

    def _sample_scheduler_wait_backoff_seconds(
        self, queue_state_reason: str | None
    ) -> float | None:
        if queue_state_reason not in {
            "fair_share_wait",
            "queue_class_wait",
            "coalescing_window",
            "residency_wait",
        }:
            return None
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return None
        if self._sample_dispatch_profile == "local_single_gpu":
            if queue_state_reason == "coalescing_window":
                return 0.2
            if queue_state_reason == "queue_class_wait":
                return 0.3
            if queue_state_reason == "residency_wait":
                return 0.25
            return 0.35
        if self._sample_dispatch_profile == "workstation":
            if queue_state_reason == "coalescing_window":
                return 0.1
            if queue_state_reason == "queue_class_wait":
                return 0.15
            if queue_state_reason == "residency_wait":
                return 0.12
            return 0.2
        if (
            self._sample_dispatch_profile == "auto"
            and self._sample_dispatch_profile_limit is not None
        ):
            if queue_state_reason == "coalescing_window":
                return 0.1
            if queue_state_reason == "queue_class_wait":
                return 0.15
            if queue_state_reason == "residency_wait":
                return 0.12
            return 0.2
        return None

    def note_sampling_queue_state(
        self,
        queue_state: str,
        queue_state_reason: str | None = None,
    ) -> None:
        if queue_state == "active":
            backoff_seconds = self._sample_scheduler_wait_backoff_seconds(queue_state_reason)
            if backoff_seconds is None:
                return
            requested_until = time.monotonic() + backoff_seconds
            current_until = self._current_sample_backoff_until()
            if current_until is None or requested_until > current_until:
                self._set_sample_backoff_state(
                    backoff_until=requested_until,
                    backoff_streak=self._current_sample_backoff_streak(),
                    backoff_last_requested_at=self._current_sample_backoff_last_requested_at(),
                )
            return

        if queue_state not in {"paused_capacity", "paused_rate_limit"}:
            return

        backoff_seconds = self._sample_queue_state_backoff_seconds(queue_state)
        requested_until = time.monotonic() + backoff_seconds
        current_until = self._current_sample_backoff_until()
        if current_until is None or requested_until > current_until:
            self._set_sample_backoff_state(
                backoff_until=requested_until,
                backoff_streak=self._current_sample_backoff_streak(),
                backoff_last_requested_at=self._current_sample_backoff_last_requested_at(),
            )

    def _sample_recent_backoff_window_seconds(self) -> float:
        if self._sample_dispatch_profile == "local_single_gpu":
            return 10.0
        if self._sample_dispatch_profile == "workstation":
            return 6.0
        if self._sample_dispatch_profile == "remote_large_backend":
            return 2.0
        if (
            self._sample_dispatch_profile == "auto"
            and self._sample_dispatch_profile_limit is not None
        ):
            return 6.0
        return 2.0

    def _sample_backoff_requested_recently(self) -> bool:
        backoff_until = self._current_sample_backoff_until()
        return (
            backoff_until is not None
            and time.monotonic() - backoff_until < self._sample_recent_backoff_window_seconds()
        )

    def sample_backoff_sleep_seconds(self) -> float:
        backoff_until = self._current_sample_backoff_until()
        if backoff_until is None:
            return 0.0
        remaining_seconds = max(0.0, backoff_until - time.monotonic())
        if remaining_seconds <= 0.0:
            return 0.0
        if not self._is_local_sample_dispatch_profile(self._sample_dispatch_profile):
            return min(1.0, remaining_seconds)
        capped_seconds = min(1.0, remaining_seconds)
        jitter_low = max(0.25, capped_seconds * 0.5)
        return min(remaining_seconds, random.uniform(jitter_low, capped_seconds))

    @asynccontextmanager
    async def _sample_dispatch_bytes_rate_limit(self, bytes: int):
        if self._sample_backoff_requested_recently():
            # Rate limit more aggressively if we received backoff response recently
            bytes *= 20
        async with self._shared_sample_dispatch_bytes_semaphore().acquire(bytes):
            yield

    @asynccontextmanager
    async def sample_dispatch_rate_limit(
        self,
        estimated_bytes_count: int,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ):
        await self._resolve_sample_dispatch_profile_limit()
        dispatch_profile_limit = await self._current_sample_dispatch_limit(
            request_kind=request_kind,
        )
        await self._maybe_wait_for_sample_coalescing_window(request_kind=request_kind)

        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(self._sample_dispatch_count_rate_limit())
            if dispatch_profile_limit is not None:
                await stack.enter_async_context(
                    self._sample_dispatch_profile_rate_limit(dispatch_profile_limit)
                )
            if self._sample_backoff_requested_recently():
                await stack.enter_async_context(
                    self._sample_dispatch_count_throttled_rate_limit(request_kind=request_kind)
                )
            await stack.enter_async_context(
                self._sample_dispatch_bytes_rate_limit(estimated_bytes_count)
            )

            yield

    @asynccontextmanager
    async def sample_request_rate_limit(
        self,
        estimated_bytes_count: int,
        *,
        request_kind: SampleDispatchRequestKind = "sample",
    ):
        await self._resolve_sample_dispatch_profile_limit()
        async with contextlib.AsyncExitStack() as stack:
            outstanding_limit = await self._current_sample_outstanding_limit(
                request_kind=request_kind,
            )
            if outstanding_limit is not None:
                await stack.enter_async_context(
                    self._sample_outstanding_profile_rate_limit(outstanding_limit)
                )
            await stack.enter_async_context(
                self.sample_dispatch_rate_limit(
                    estimated_bytes_count,
                    request_kind=request_kind,
                )
            )
            yield

    async def _session_heartbeat(self, session_id: str):
        SESSION_HEARTBEAT_PERIOD_SEC = 10
        SESSION_MISSED_HEARTBEAT_WARNING_THRESHOLD_SEC = 60 * 2
        last_heartbeat_time = time.monotonic()
        while True:
            await asyncio.sleep(SESSION_HEARTBEAT_PERIOD_SEC)

            last_exception: str | None = None
            try:
                with self.aclient(ClientConnectionPoolType.SESSION) as client:
                    await client.service.session_heartbeat(
                        session_id=session_id, max_retries=0, timeout=10
                    )
                last_heartbeat_time = time.monotonic()
            except Exception as e:
                last_exception = f"{type(e).__name__}: {str(e)}"
                pass
            if (
                time.monotonic() - last_heartbeat_time
                > SESSION_MISSED_HEARTBEAT_WARNING_THRESHOLD_SEC
            ):
                logger.warning(
                    f"Session heartbeat failed for {time.monotonic() - last_heartbeat_time} seconds for session {session_id}. Last exception: {last_exception}.\n"
                    + "Your connection may be unreliable or Cybernetics is down. If this persists, the session will be terminated."
                )

    async def _create_sampling_session(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> str:
        if model_path and not model_path.startswith("worldlines://"):
            raise ValueError("model_path must start with 'worldlines://'")
        # _create_sampling_session can only be called via a ServiceClient.
        # ServiceClient will never have a shadow holder, so we can safely assert.
        assert self._sampling_client_counter is not None
        sampling_session_seq_id = self._sampling_client_counter
        self._sampling_client_counter += 1
        with self.aclient(ClientConnectionPoolType.SESSION) as client:
            request = types.CreateSamplingSessionRequest(
                session_id=self._session_id,
                sampling_session_seq_id=sampling_session_seq_id,
                model_path=model_path,
                base_model=base_model,
            )
            result = await client.service.create_sampling_session(request=request)
            return result.sampling_session_id

    async def _start_heartbeat(self) -> asyncio.Task[None]:
        """Start the session heartbeat task."""
        return asyncio.create_task(self._session_heartbeat(self._session_id))

    async def _create_session(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
    ) -> str:
        if (tags_str := os.environ.get("CYBERNETICS_TAGS")) is not None:
            tags: set[str] = set(tags_str.split(","))
        else:
            tags = set()
        with self.aclient(ClientConnectionPoolType.SESSION) as client:
            request = _create_session_request(
                tags=list(tags),
                user_metadata=user_metadata or {},
                sdk_version=worldlines_sdk_version,
                project_id=project_id,
            )
            result = await client.service.create_session(request=request)
        if result.info_message:
            logger.info(result.info_message)
        if result.warning_message:
            logger.warning(result.warning_message)
        if result.error_message:
            logger.error(result.error_message)
        return result.session_id

    def _get_client_connection_pool(
        self, client_pool_type: ClientConnectionPoolType
    ) -> ClientConnectionPool:
        if client_pool_type not in self._client_pools:
            max_requests_per_client = (
                1
                if client_pool_type == ClientConnectionPoolType.TRAIN
                else MAX_REQUESTS_PER_HTTPX_CLIENT
            )
            self._client_pools[client_pool_type] = ClientConnectionPool(
                self.get_loop(), max_requests_per_client, self._constructor_kwargs
            )
        return self._client_pools[client_pool_type]

    def get_session_id(self) -> str:
        return self._session_id

    async def get_scheduler_state_async(self) -> types.GetSchedulerStateResponse:
        async def _send_request() -> types.GetSchedulerStateResponse:
            with self.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.service.get_scheduler_state()

        return await self.execute_with_retries(_send_request)

    def get_scheduler_state(self) -> AwaitableConcurrentFuture[types.GetSchedulerStateResponse]:
        return self.run_coroutine_threadsafe(self.get_scheduler_state_async())

    async def get_server_capabilities_async(
        self, force_refresh: bool = False
    ) -> types.GetServerCapabilitiesResponse:
        if self._server_capabilities_cache is not None and not force_refresh:
            return self._server_capabilities_cache

        async def _send_request() -> types.GetServerCapabilitiesResponse:
            with self.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.service.get_server_capabilities()

        capabilities = await self.execute_with_retries(_send_request)
        self._server_capabilities_cache = capabilities
        self._sample_dispatch_profile_loaded = False
        self._sample_outstanding_profile_limit = None
        return capabilities

    def get_server_capabilities(
        self, force_refresh: bool = False
    ) -> AwaitableConcurrentFuture[types.GetServerCapabilitiesResponse]:
        return self.run_coroutine_threadsafe(
            self.get_server_capabilities_async(force_refresh=force_refresh)
        )

    def server_supports_training(self) -> bool | None:
        return self.get_server_capabilities().result().supports_training

    async def server_supports_training_async(self) -> bool | None:
        capabilities = await self.get_server_capabilities()
        return capabilities.supports_training

    def server_supports_optimizer_restore(self) -> bool | None:
        return self.get_server_capabilities().result().supports_optimizer_restore

    async def server_supports_optimizer_restore_async(self) -> bool | None:
        capabilities = await self.get_server_capabilities()
        return capabilities.supports_optimizer_restore

    def server_supports_custom_loss_v2(self) -> bool | None:
        return self.get_server_capabilities().result().supports_custom_loss_v2

    async def server_supports_custom_loss_v2_async(self) -> bool | None:
        capabilities = await self.get_server_capabilities()
        return capabilities.supports_custom_loss_v2

    def get_training_client_id(self) -> int:
        # get_training_client_id can only be called via a ServiceClient.
        # ServiceClient will never have a shadow holder, so we can safely assert.
        assert self._training_client_counter is not None
        with self._training_client_lock:
            training_client_id = self._training_client_counter
            self._training_client_counter += 1
            return training_client_id

    def aclient(
        self, client_pool_type: ClientConnectionPoolType
    ) -> AbstractContextManager[AsyncCybernetics]:
        return self._get_client_connection_pool(client_pool_type).aclient()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def get_telemetry(self) -> Telemetry | None:
        return self._telemetry

    def run_coroutine_threadsafe(
        self,
        coro: Coroutine[Any, Any, T],
    ) -> AwaitableConcurrentFuture[T]:
        return AwaitableConcurrentFuture(asyncio.run_coroutine_threadsafe(coro, self.get_loop()))

    def close(self):
        self.run_coroutine_threadsafe(self._async_cleanup())
        if telemetry := getattr(self, "_telemetry", None):
            telemetry.stop()

    def __del__(self):
        self.close()

    async def _async_cleanup(self):
        if self._session_heartbeat_task:
            self._session_heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._session_heartbeat_task

    @staticmethod
    def _is_retryable_status_code(status_code: int) -> bool:
        return status_code in (408, 409, 429) or (500 <= status_code < 600)

    @staticmethod
    def _is_retryable_exception(exception: Exception) -> bool:
        RETRYABLE_EXCEPTIONS = (
            asyncio.TimeoutError,
            APIConnectionError,
            httpx.TimeoutException,
        )
        if isinstance(exception, RETRYABLE_EXCEPTIONS):
            return True
        if isinstance(exception, APIStatusError):
            return InternalClientHolder._is_retryable_status_code(exception.status_code)
        return False

    async def execute_with_retries(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        MAX_WAIT_TIME = 60 * 5
        start_time = time.time()
        attempt_count = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                is_retryable = self._is_retryable_exception(e)
                user_error = is_user_error(e)
                current_time = time.time()
                elapsed_time = current_time - start_time
                if telemetry := self.get_telemetry():
                    telemetry.log(
                        "InternalClientHolder.execute_with_retries.exception",
                        event_data={
                            "func": getattr(
                                func, "__qualname__", getattr(func, "__name__", type(func).__name__)
                            ),
                            "exception": str(e),
                            "exception_type": type(e).__name__,
                            "exception_stack": "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                            if e.__traceback__
                            else None,
                            "status_code": getattr(e, "status_code", None),
                            "is_retryable": is_retryable,
                            "is_user_error": user_error,
                            "attempt_count": attempt_count,
                            "start_time": start_time,
                            "current_time": current_time,
                            "elapsed_time": elapsed_time,
                        },
                        severity="WARNING" if is_retryable or user_error else "ERROR",
                    )
                if is_retryable and elapsed_time < MAX_WAIT_TIME:
                    # Apply exponential backoff
                    time_to_wait = min(2**attempt_count, 30)
                    attempt_count += 1
                    # Don't wait too long if we're almost at the max wait time
                    time_to_wait = min(time_to_wait, start_time + MAX_WAIT_TIME - current_time)
                    await asyncio.sleep(time_to_wait)
                    continue

                raise e

    def estimate_bytes_count_in_chunk(self, chunk: types.ModelInputChunk) -> int:
        if isinstance(chunk, types.ImageChunk):
            return len(chunk.data)
        if isinstance(chunk, types.ImageAssetPointerChunk):
            return len(chunk.location)
        return chunk.length * 10

    def estimate_bytes_count_in_model_input(self, model_input: types.ModelInput) -> int:
        return sum(self.estimate_bytes_count_in_chunk(chunk) for chunk in model_input.chunks)


def _current_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
