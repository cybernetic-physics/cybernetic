"""Synchronous Simulation Asset Rendering MVP client."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urlparse

from cybernetics.lib.credentials import resolve_api_key, resolve_base_url

from .errors import SimulationError
from .mcp import SessionMCPClient
from .packaging import (
    AssetPackage,
    AssetPackageError,
    detect_gaussian_splat_format,
    package_local_asset,
    validate_hosted_splat_ply,
)

DEFAULT_BASE_URL = "https://api.cyberneticphysics.com"
SIMULATION_ASSET_REF_SCHEMA_VERSION = "simulation-asset-ref/v1"
_READY_STATUSES = {"running", "idle"}
_TERMINAL_STATUSES = {"failed", "terminated", "stopped", "error", "snapshot_failed"}
_JOB_SUCCESS_STATUSES = {"completed", "succeeded"}
_JOB_TERMINAL_STATUSES = {
    "completed",
    "succeeded",
    "failed",
    "canceled",
    "cancelled",
}
_SPLAT_UPLOAD_CONTENT_TYPE = "application/octet-stream"
_STOP_ACCEPTED_STATUSES = (_TERMINAL_STATUSES - {"snapshot_failed"}) | {"stopping"}
_SESSION_TRANSIENT_STATUS_CODES = {502, 503, 504}
_SESSION_TRANSIENT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 10.0)
AssetRefKind = Literal["environment_version", "local_bundle", "catalog_asset"]


@dataclass(frozen=True)
class EnvironmentRef:
    env_id: str
    version_id: str | None = None

    @property
    def uri(self) -> str:
        if self.version_id:
            return f"cybernetics://envs/{self.env_id}/versions/{self.version_id}"
        return f"cybernetics://envs/{self.env_id}"


@dataclass(frozen=True)
class SimulationAssetRef:
    """Serializable bridge from sim asset ingestion into RobotTaskSpec.asset_refs."""

    uri: str
    asset_kind: str
    compatibility_status: str
    ref_kind: AssetRefKind
    schema_version: str = SIMULATION_ASSET_REF_SCHEMA_VERSION
    env_id: str | None = None
    version_id: str | None = None
    root_stage_relpath: str | None = None
    content_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "ref_kind": self.ref_kind,
            "uri": self.uri,
            "asset_kind": self.asset_kind,
            "compatibility_status": self.compatibility_status,
            "metadata": dict(self.metadata),
        }
        if self.env_id is not None:
            payload["env_id"] = self.env_id
        if self.version_id is not None:
            payload["version_id"] = self.version_id
        if self.root_stage_relpath is not None:
            payload["root_stage_relpath"] = self.root_stage_relpath
        if self.content_sha256 is not None:
            payload["content_sha256"] = self.content_sha256
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationAssetRef":
        schema = data.get("schema_version")
        if schema != SIMULATION_ASSET_REF_SCHEMA_VERSION:
            raise SimulationError(
                f"simulation asset ref schema_version must be "
                f"{SIMULATION_ASSET_REF_SCHEMA_VERSION!r}, got {schema!r}"
            )
        ref_kind = data.get("ref_kind")
        if ref_kind not in {"environment_version", "local_bundle", "catalog_asset"}:
            raise SimulationError(f"unsupported simulation asset ref kind: {ref_kind!r}")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SimulationError("simulation asset ref metadata must be an object")
        return cls(
            uri=_require_plain_str(data, "uri"),
            asset_kind=_require_plain_str(data, "asset_kind"),
            compatibility_status=_require_plain_str(data, "compatibility_status"),
            ref_kind=ref_kind,
            env_id=_optional_plain_str(data, "env_id"),
            version_id=_optional_plain_str(data, "version_id"),
            root_stage_relpath=_optional_plain_str(data, "root_stage_relpath"),
            content_sha256=_optional_plain_str(data, "content_sha256"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class SimImportResult:
    asset_ref: str
    environment: dict[str, Any] | None
    version: dict[str, Any] | None
    environment_ref: EnvironmentRef | None
    package: AssetPackage | None

    @property
    def env_id(self) -> str | None:
        return self.environment_ref.env_id if self.environment_ref else None

    @property
    def version_id(self) -> str | None:
        return self.environment_ref.version_id if self.environment_ref else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_ref": self.asset_ref,
            "environment": self.environment,
            "version": self.version,
            "environment_ref": self.environment_ref.uri if self.environment_ref else None,
            "package": self.package.to_dict() if self.package else None,
        }

    def to_asset_ref(self) -> SimulationAssetRef:
        if self.environment_ref is not None:
            version = self.version or {}
            package = self.package
            asset_kind = (
                package.asset_kind
                if package is not None
                else (_dict_str(version, "assetKind") or "environment")
            )
            metadata: dict[str, Any] = {}
            if self.environment and isinstance(self.environment.get("name"), str):
                metadata["environment_name"] = self.environment["name"]
            if isinstance(version.get("status"), str):
                metadata["version_status"] = version["status"]
            if package is not None:
                metadata["source_name"] = package.manifest.get("source", {}).get("name")
                metadata["file_count"] = package.file_count
            return SimulationAssetRef(
                ref_kind="environment_version",
                uri=self.environment_ref.uri,
                env_id=self.environment_ref.env_id,
                version_id=self.environment_ref.version_id,
                root_stage_relpath=_dict_str(version, "rootStageRelpath")
                or (package.root_stage_relpath if package else None),
                asset_kind=asset_kind,
                compatibility_status=(
                    package.compatibility_status if package else "ready_to_render"
                ),
                content_sha256=_dict_str(version, "contentSha256"),
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
        if self.package is not None:
            if self.package.temporary and not self.package.bundle_path.exists():
                raise SimulationError(
                    "local bundle asset refs require a kept bundle; pass "
                    "keep_bundle=True or bundle_path=... when importing non-renderable assets"
                )
            return SimulationAssetRef(
                ref_kind="local_bundle",
                uri=self.package.bundle_path.as_uri(),
                root_stage_relpath=self.package.root_stage_relpath,
                asset_kind=self.package.asset_kind,
                compatibility_status=self.package.compatibility_status,
                content_sha256=self.package.bundle_sha256,
                metadata={
                    "source_name": self.package.manifest.get("source", {}).get("name"),
                    "file_count": self.package.file_count,
                    "bundle_size_bytes": self.package.bundle_size_bytes,
                },
            )
        raise SimulationError("import result does not contain a simulation asset reference")


@dataclass(frozen=True)
class SimLaunchResult:
    session: dict[str, Any]
    session_id: str
    session_url: str
    viewer_url: str | None = None
    preview_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "session_id": self.session_id,
            "session_url": self.session_url,
            "viewer_url": self.viewer_url,
            "preview_url": self.preview_url,
        }


@dataclass(frozen=True)
class SimRenderResult:
    import_result: SimImportResult
    launch_result: SimLaunchResult | None
    preview_path: Path | None
    preview_url: str | None
    public_url: str | None
    launch_url: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "preview_url": self.preview_url,
            "public_url": self.public_url,
            "launch_url": self.launch_url,
            "import": self.import_result.to_dict(),
            "launch": self.launch_result.to_dict() if self.launch_result else None,
        }


class SimulationClient:
    """Import local simulation assets and launch hosted preview sessions.

    The MVP intentionally reuses the existing control-plane environment/session
    APIs instead of depending on the future first-class sim-asset endpoints.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            raise SimulationError(
                "No API key found. Run 'cybernetics auth login' or set CYBERNETICS_API_KEY."
            )
        self.api_key = resolved_key
        self.base_url = (resolve_base_url(base_url) or DEFAULT_BASE_URL).rstrip("/")
        self._owns_client = http_client is None
        if http_client is None:
            import httpx

            http_client = httpx.Client(base_url=self.base_url, timeout=180.0)
        self._client = http_client
        self._mcp_clients: set[SessionMCPClient] = set()

    def close(self) -> None:
        cleanup_error: Exception | None = None
        revocation_pending: set[SessionMCPClient] = set()
        for mcp_client in tuple(self._mcp_clients):
            try:
                mcp_client.close()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
                revocation_pending.add(mcp_client)
        self._mcp_clients = revocation_pending
        if self._owns_client and not revocation_pending:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self) -> "SimulationClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if not isinstance(exc, BaseException):
                raise
            exc.add_note(
                "SimulationClient cleanup failed after the primary error: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    def import_asset(
        self,
        asset_ref: str | Path,
        *,
        name: str | None = None,
        description: str | None = None,
        root_stage: str | None = None,
        notes: str | None = None,
        bundle_path: str | Path | None = None,
        keep_bundle: bool = False,
        require_renderable: bool = False,
        source_url: str | None = None,
    ) -> SimImportResult:
        ref = str(asset_ref)
        existing = parse_environment_ref(ref)
        if existing:
            return SimImportResult(
                asset_ref=ref,
                environment=None,
                version={"id": existing.version_id, "envId": existing.env_id}
                if existing.version_id
                else None,
                environment_ref=existing,
                package=None,
            )
        _reject_remote_ref(ref)

        package = package_local_asset(
            ref,
            root_stage=root_stage,
            output_path=bundle_path,
            source_url=source_url,
        )
        if require_renderable and package.compatibility_status != "ready_to_render":
            package.cleanup()
            if _is_gaussian_splat_kind(package.asset_kind):
                raise SimulationError(
                    f"{package.asset_kind} is {package.compatibility_status}; hosted Isaac "
                    "sessions render ParticleField USDZ, not raw splat files. Convert it first with "
                    "`cybernetics splat upload <file> --convert --wait`, then launch the "
                    "exported USDZ artifact."
                )
            raise SimulationError(
                f"{package.asset_kind} is {package.compatibility_status}; "
                "the MVP can render USD/USDZ assets and package robot descriptions for later conversion."
            )
        if keep_bundle:
            package = _with_persistent_bundle(package)

        # Gaussian splats upload as environment versions even though they are
        # needs_conversion: the bundle is the durable source artifact the
        # platform converts to ParticleField USDZ. Other non-renderable kinds keep the
        # package-local behavior.
        if package.compatibility_status != "ready_to_render" and not _is_gaussian_splat_kind(
            package.asset_kind
        ):
            if not keep_bundle and bundle_path is None:
                package.cleanup()
            return SimImportResult(
                asset_ref=ref,
                environment=None,
                version=None,
                environment_ref=None,
                package=package,
            )

        env_name = name or _default_environment_name(package)
        environment = self._request(
            "POST",
            "/v1/envs",
            json_body={
                "name": env_name,
                "description": description or f"Imported by cybernetics sim from {Path(ref).name}",
            },
        )
        env_id = _require_str(environment, "id")

        version_body: dict[str, Any] = {
            "notes": notes or f"Imported from {Path(ref).name}",
            "upload": {"type": "bundle_zip"},
        }
        # Splat bundles have no USD stage yet; omit the key instead of sending
        # null (the control plane treats rootStageRelpath as optional-string).
        if package.root_stage_relpath is not None:
            version_body["rootStageRelpath"] = package.root_stage_relpath
        version_response = self._request(
            "POST",
            f"/v1/envs/{env_id}/versions",
            json_body=version_body,
        )
        version = _require_dict(version_response, "version")
        version_id = _require_str(version, "id")
        upload = version_response.get("upload")
        if not isinstance(upload, dict):
            raise SimulationError("control plane did not return a bundle upload target")
        self._upload_bundle(upload, package.bundle_path)

        finalize_body: dict[str, Any] = {
            "contentSha256": package.bundle_sha256,
            "bundleSizeBytes": package.bundle_size_bytes,
        }
        if package.root_stage_relpath is not None:
            finalize_body["rootStageRelpath"] = package.root_stage_relpath
        finalized = self._request(
            "POST",
            f"/v1/envs/{env_id}/versions/{version_id}/finalize",
            json_body=finalize_body,
        )
        if not keep_bundle and bundle_path is None:
            package.cleanup()

        return SimImportResult(
            asset_ref=ref,
            environment=environment,
            version=finalized,
            environment_ref=EnvironmentRef(env_id=env_id, version_id=version_id),
            package=package,
        )

    def launch(
        self,
        asset_or_env: str | Path | SimImportResult | EnvironmentRef,
        *,
        name: str | None = None,
        root_stage: str | None = None,
        workspace_id: str | None = None,
        gpu_spec: str | None = None,
        runtime_provider: Literal["vast", "warm_pool"] | None = None,
        max_runtime_minutes: int | None = None,
        idle_timeout_minutes: int | None = None,
        wait: bool = False,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
    ) -> SimLaunchResult:
        imported = self._coerce_import_result(
            asset_or_env,
            name=name,
            root_stage=root_stage,
            require_renderable=True,
        )
        env_ref = imported.environment_ref
        if not env_ref:
            raise SimulationError("launch requires an environment reference")

        body: dict[str, Any] = {
            "name": name or _launch_name(imported),
            "envId": env_ref.env_id,
        }
        if env_ref.version_id:
            body["baseVersionId"] = env_ref.version_id
        if workspace_id:
            body["workspaceId"] = workspace_id
        if gpu_spec:
            body["gpuSpec"] = gpu_spec
        if runtime_provider:
            body["runtimeProvider"] = runtime_provider
        if max_runtime_minutes:
            body["maxRuntimeMinutes"] = max_runtime_minutes
        if idle_timeout_minutes:
            body["idleTimeoutMinutes"] = idle_timeout_minutes

        session = self._request("POST", "/v1/sessions", json_body=body)
        session_id = _session_id(session)
        if wait:
            session = self.wait_for_session(
                session_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )

        return SimLaunchResult(
            session=session,
            session_id=session_id,
            session_url=f"{self._site_base_url()}/sessions/{session_id}",
            viewer_url=_viewer_url(session),
        )

    def render(
        self,
        asset_ref: str | Path | SimImportResult | EnvironmentRef,
        *,
        name: str | None = None,
        root_stage: str | None = None,
        out: str | Path | None = None,
        wait: bool = False,
        keep_session: bool = True,
        public: bool = False,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
    ) -> SimRenderResult:
        if public:
            raise SimulationError("Public /sim artifact pages are not implemented in the MVP.")

        imported = self._coerce_import_result(
            asset_ref,
            name=name,
            root_stage=root_stage,
            require_renderable=True,
        )
        launch = self.launch(
            imported,
            name=name,
            wait=wait,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

        preview_url: str | None = None
        preview_path: Path | None = None
        status = "session_started"
        try:
            if wait:
                preview_url = self.preview_url(launch.session_id)
                status = "preview_ready"
                if out is not None:
                    preview_path = Path(out).expanduser().resolve()
                    self.download_preview(launch.session_id, preview_path)
        except Exception:
            if wait and not keep_session:
                try:
                    self.stop_session(launch.session_id)
                except SimulationError:
                    pass
            raise

        if wait and not keep_session:
            self.stop_session(launch.session_id)

        return SimRenderResult(
            import_result=imported,
            launch_result=launch,
            preview_path=preview_path,
            preview_url=preview_url,
            public_url=None,
            launch_url=launch.viewer_url or launch.session_url,
            status=status,
        )

    def wait_for_session(
        self,
        session_id: str,
        *,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """Wait for preview readiness, retrying only bounded transient gateway failures."""
        deadline = time.monotonic() + timeout_seconds
        last = "not checked"
        while True:
            path = f"/v1/sessions/{session_id}"
            operation = f"GET {path}"
            response = self._request_session_response(
                "GET",
                path,
                operation=operation,
                deadline=deadline,
            )
            session = _response_json_object(response, operation)
            status = str(session.get("status", "")).lower()
            if status in _TERMINAL_STATUSES:
                raise SimulationError(f"session {session_id} entered terminal status {status!r}")
            if status in _READY_STATUSES and _session_preview_ready(session):
                return session
            last = f"status={status or 'unknown'}"
            if time.monotonic() >= deadline:
                raise SimulationError(
                    f"session {session_id} was not ready after {timeout_seconds}s ({last})"
                )
            time.sleep(poll_interval_seconds)

    def preview_url(self, session_id: str, *, ttl_seconds: int = 3600, quality: int = 90) -> str:
        grant = self._request(
            "POST",
            f"/v1/sessions/{session_id}/cua-grant",
            json_body={"ttlSeconds": ttl_seconds},
        )
        neko_url = _require_str(grant, "neko_url")
        http_base = _neko_http_base(neko_url)
        login_token = self._neko_login_token(http_base, grant)
        query = urlencode({"quality": str(quality), "token": login_token})
        return f"{http_base}/api/shot.jpg?{query}"

    def download_preview(
        self,
        session_id: str,
        destination: str | Path,
        *,
        ttl_seconds: int = 3600,
        quality: int = 90,
    ) -> Path:
        url = self.preview_url(session_id, ttl_seconds=ttl_seconds, quality=quality)
        response = self._client.get(url)
        _raise_for_response(response, "GET CUA preview image")
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def stop_session(self, session_id: str) -> None:
        """Request session cleanup with bounded retries and verified idempotency."""
        path = f"/v1/sessions/{session_id}/stop"
        operation = f"POST {path}"
        response = self._request_session_response(
            "POST",
            path,
            operation=operation,
            json_body={},
        )
        if response.status_code == 409:
            verify_path = f"/v1/sessions/{session_id}"
            verify_operation = f"GET {verify_path} after stop conflict"
            verify_response = self._request_session_response(
                "GET",
                verify_path,
                operation=verify_operation,
            )
            session = _response_json_object(verify_response, verify_operation)
            status = str(session.get("status", "")).lower()
            if status in _STOP_ACCEPTED_STATUSES:
                return
        _raise_for_response(response, operation)

    def upload_splat(self, path: str | Path) -> dict[str, Any]:
        """Upload a local Gaussian splat file as a reconstruction input.

        Presigns via ``POST /v1/uploads/presign`` (``inputKind: "splat"``),
        uploads the file, and returns ``{"uploadId", "inputPrefix", "format"}``.
        The returned ``inputPrefix`` feeds :meth:`create_splat_convert_job`.
        """
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise SimulationError(f"splat path is not a file: {source}")
        splat_format = detect_gaussian_splat_format(source)
        if splat_format is None:
            raise SimulationError(
                f"{source.name} is not a recognized Gaussian splat (.ply with 3DGS "
                "vertex properties, .spz, .splat, or .ksplat)"
            )
        if splat_format != "ply":
            raise SimulationError(
                f"hosted splat conversion accepts only standard 3DGS .ply files; "
                f"got .{splat_format}"
            )
        try:
            validate_hosted_splat_ply(source)
        except AssetPackageError as exc:
            raise SimulationError(str(exc)) from exc

        presign = self._request(
            "POST",
            "/v1/uploads/presign",
            json_body={
                "files": [
                    {
                        "name": source.name,
                        "contentType": _SPLAT_UPLOAD_CONTENT_TYPE,
                        "size": source.stat().st_size,
                    }
                ],
                "inputKind": "splat",
            },
        )
        urls = presign.get("presignedUrls")
        if not isinstance(urls, list) or not urls or not isinstance(urls[0], dict):
            raise SimulationError("control plane did not return a presigned splat upload")
        self._upload_presigned_file(urls[0], source, _SPLAT_UPLOAD_CONTENT_TYPE)

        return {
            "uploadId": _require_str(presign, "uploadId"),
            "inputPrefix": _require_str(presign, "inputPrefix"),
            "format": splat_format,
            "name": source.name,
        }

    def create_splat_convert_job(
        self,
        input_uri: str,
        *,
        max_runtime_minutes: int = 30,
        max_hourly_price: float = 2.0,
        gpu_min_vram: int = 24,
    ) -> dict[str, Any]:
        """Create a splat→ParticleField-USDZ conversion job for an uploaded PLY.

        The job runs the export-only path of the reconstruction pipeline
        (checksum-pinned OpenUSD converter); COLMAP and training are skipped.
        """
        return self._request(
            "POST",
            "/v1/jobs",
            json_body={
                "inputUri": input_uri,
                "inputKind": "splat",
                "config": {"exportUsdz": True},
                "costGuardrails": {
                    "maxRuntimeMinutes": max_runtime_minutes,
                    "maxHourlyPrice": max_hourly_price,
                    "gpuMinVram": gpu_min_vram,
                },
            },
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 10.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = self.get_job(job_id)
            status = str(job.get("status", "")).lower()
            if status in _JOB_SUCCESS_STATUSES:
                return job
            if status in _JOB_TERMINAL_STATUSES:
                error = job.get("errorMessage") or "no error message"
                raise SimulationError(f"job {job_id} ended {status!r}: {error}")
            if time.monotonic() >= deadline:
                raise SimulationError(
                    f"job {job_id} still {status or 'unknown'!r} after {timeout_seconds}s"
                )
            time.sleep(poll_interval_seconds)

    def job_artifacts(self, job_id: str) -> dict[str, Any]:
        """Return ``{"artifacts": {...}, "downloadUrls": {...}}`` for a job."""
        return self._request("GET", f"/v1/jobs/{job_id}/artifacts")

    def _upload_presigned_file(
        self,
        upload: dict[str, Any],
        source: Path,
        content_type: str,
        *,
        upload_name: str | None = None,
    ) -> None:
        url = upload.get("url") or upload.get("putUrl")
        if not isinstance(url, str) or not url:
            raise SimulationError("presigned upload response did not include an upload URL")
        fields = upload.get("fields")
        if isinstance(fields, dict):
            with source.open("rb") as handle:
                response = self._client.post(
                    url,
                    data={str(k): str(v) for k, v in fields.items()},
                    files={"file": (upload_name or source.name, handle, content_type)},
                )
            _raise_for_response(response, f"POST {url}")
            return
        # Future-compatible fallback if the API switches to a real signed PUT.
        with source.open("rb") as handle:
            response = self._client.put(
                url,
                content=handle.read(),
                headers={"content-type": content_type},
            )
        _raise_for_response(response, f"PUT {url}")

    def mcp_session(
        self,
        session_id: str,
        *,
        ttl_seconds: int = 3600,
        name: str | None = None,
    ) -> SessionMCPClient:
        """Mint private, session-scoped credentials for ``isaac.*`` MCP calls."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise SimulationError("MCP session_id must be a non-empty string")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise SimulationError("MCP ttl_seconds must be an integer")
        if ttl_seconds < 1 or ttl_seconds > 24 * 3600:
            raise SimulationError("MCP ttl_seconds must be between 1 and 86400")
        if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 100):
            raise SimulationError(
                "MCP key name must be a non-empty string of at most 100 characters"
            )

        request_body: dict[str, Any] = {
            "sessionId": session_id,
            "ttlSeconds": ttl_seconds,
        }
        if name is not None:
            request_body["name"] = name
        grant = self._request(
            "POST",
            "/v1/api-keys/session-scoped",
            json_body=request_body,
        )
        key_id = _require_str(grant, "id")
        try:
            if _require_str(grant, "sessionId") != session_id:
                raise SimulationError("session-scoped key response targeted a different session")
            if _require_str(grant, "keyKind") != "session":
                raise SimulationError("session-scoped key response has an invalid key kind")
            scoped_key = _require_str(grant, "key")
        except SimulationError:
            try:
                self._revoke_mcp_key(key_id)
            except SimulationError:
                pass
            raise

        mcp_client = SessionMCPClient(
            base_url=self.base_url,
            http_client=self._client,
            session_id=session_id,
            key_id=key_id,
            scoped_key=scoped_key,
            revoke_key=self._revoke_mcp_key,
        )
        self._mcp_clients.add(mcp_client)
        return mcp_client

    def _revoke_mcp_key(self, key_id: str) -> None:
        self._request("DELETE", f"/v1/api-keys/{key_id}")

    def _neko_login_token(self, http_base: str, grant: dict[str, Any]) -> str:
        response = self._client.post(
            f"{http_base}/api/login",
            json={
                "username": _require_str(grant, "neko_username"),
                "password": _require_str(grant, "neko_password"),
            },
        )
        _raise_for_response(response, "POST CUA Neko login")
        body = response.json()
        if not isinstance(body, dict):
            raise SimulationError("CUA Neko login returned a non-object JSON response")
        return _require_str(body, "token")

    def _coerce_import_result(
        self,
        value: str | Path | SimImportResult | EnvironmentRef,
        *,
        name: str | None = None,
        root_stage: str | None = None,
        require_renderable: bool,
    ) -> SimImportResult:
        if isinstance(value, SimImportResult):
            return value
        if isinstance(value, EnvironmentRef):
            return SimImportResult(
                asset_ref=value.uri,
                environment=None,
                version={"id": value.version_id, "envId": value.env_id}
                if value.version_id
                else None,
                environment_ref=value,
                package=None,
            )
        return self.import_asset(
            value,
            name=name,
            root_stage=root_stage,
            require_renderable=require_renderable,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operation = f"{method} {path}"
        response = self._request_response(method, path, json_body=json_body)
        return _response_json_object(response, operation)

    def _request_response(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        return self._client.request(
            method,
            path,
            json=json_body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _request_session_response(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json_body: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> Any:
        """Run one session request with a small, explicit transient retry budget."""
        last_response: Any = None
        for attempt in range(1, len(_SESSION_TRANSIENT_BACKOFF_SECONDS) + 2):
            if attempt > 1 and deadline is not None and time.monotonic() >= deadline:
                _raise_transient_session_error(
                    last_response,
                    operation,
                    attempts=attempt - 1,
                    reason="retry deadline expired",
                )

            response = self._request_response(method, path, json_body=json_body)
            status_code = getattr(response, "status_code", 0)
            if status_code not in _SESSION_TRANSIENT_STATUS_CODES:
                return response
            last_response = response

            if attempt > len(_SESSION_TRANSIENT_BACKOFF_SECONDS):
                _raise_transient_session_error(
                    response,
                    operation,
                    attempts=attempt,
                    reason="retry budget exhausted",
                )

            delay = _SESSION_TRANSIENT_BACKOFF_SECONDS[attempt - 1]
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _raise_transient_session_error(
                        response,
                        operation,
                        attempts=attempt,
                        reason="retry deadline expired",
                    )
                delay = min(delay, remaining)
            time.sleep(delay)

        raise AssertionError("session retry loop exhausted without returning")

    def _upload_bundle(self, upload: dict[str, Any], bundle_path: Path) -> None:
        self._upload_presigned_file(
            upload, bundle_path, "application/zip", upload_name="bundle.zip"
        )

    def _site_base_url(self) -> str:
        parsed = urlparse(self.base_url)
        if parsed.hostname == "api.cyberneticphysics.com":
            return "https://cyberneticphysics.com"
        if parsed.hostname and parsed.hostname.endswith("-api.cyberneticphysics.com"):
            prefix = parsed.hostname.removesuffix("-api.cyberneticphysics.com")
            return f"https://{prefix}.cyberneticphysics.com"
        return self.base_url


def parse_environment_ref(value: str) -> EnvironmentRef | None:
    prefix = "cybernetics://envs/"
    if not value.startswith(prefix):
        return None
    parts = value[len(prefix) :].strip("/").split("/")
    if len(parts) == 1 and parts[0]:
        return EnvironmentRef(env_id=parts[0])
    if len(parts) == 3 and parts[0] and parts[1] == "versions" and parts[2]:
        return EnvironmentRef(env_id=parts[0], version_id=parts[2])
    raise SimulationError(
        "environment refs must look like cybernetics://envs/<env_id>/versions/<version_id>"
    )


def _is_gaussian_splat_kind(asset_kind: str) -> bool:
    return asset_kind.startswith("gaussian_splat_")


def _reject_remote_ref(value: str) -> None:
    if "://" in value:
        scheme = value.split("://", 1)[0]
        raise SimulationError(
            f"{scheme}:// asset refs are catalog-resolver work and are not supported by the SDK MVP yet"
        )


def _with_persistent_bundle(package: AssetPackage) -> AssetPackage:
    return AssetPackage(
        source_path=package.source_path,
        bundle_path=package.bundle_path,
        root_stage_relpath=package.root_stage_relpath,
        asset_kind=package.asset_kind,
        compatibility_status=package.compatibility_status,
        bundle_sha256=package.bundle_sha256,
        bundle_size_bytes=package.bundle_size_bytes,
        file_count=package.file_count,
        temporary=False,
        manifest=package.manifest,
    )


def _require_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise SimulationError(f"response is missing object field {key!r}")
    return item


def _require_str(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise SimulationError(f"response is missing string field {key!r}")
    return item


def _require_plain_str(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise SimulationError(f"simulation asset ref is missing string field {key!r}")
    return item


def _optional_plain_str(value: dict[str, Any], key: str) -> str | None:
    item = value.get(key)
    return item if isinstance(item, str) and item else None


def _dict_str(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _session_id(session: dict[str, Any]) -> str:
    value = session.get("sessionId") or session.get("id")
    if not isinstance(value, str) or not value:
        raise SimulationError("session response did not include sessionId")
    return value


def _viewer_url(session: dict[str, Any]) -> str | None:
    access = session.get("access")
    if isinstance(access, dict) and isinstance(access.get("viewerUrl"), str):
        return access["viewerUrl"]
    viewer_url = session.get("viewerUrl")
    return viewer_url if isinstance(viewer_url, str) else None


def _bridge_ready(session: dict[str, Any]) -> bool:
    if session.get("isaac_extension_ready") is True:
        return True
    bridge = session.get("bridge_status") or session.get("bridgeStatus") or {}
    return isinstance(bridge, dict) and bridge.get("isaac_extension_ready") is True


def _session_preview_ready(session: dict[str, Any]) -> bool:
    if _bridge_ready(session):
        return True
    if session.get("runtimeStatus") == "running":
        return True
    return _viewer_url(session) is not None


def _neko_http_base(neko_url: str) -> str:
    if neko_url.startswith("wss://"):
        return "https://" + neko_url[len("wss://") :]
    if neko_url.startswith("ws://"):
        return "http://" + neko_url[len("ws://") :]
    return neko_url.rstrip("/")


def _raise_for_response(response: Any, operation: str) -> None:
    status_code = getattr(response, "status_code", 0)
    if status_code < 400:
        return
    text = getattr(response, "text", "")
    raise SimulationError(f"{operation} failed with HTTP {status_code}: {text[:300]}")


def _response_json_object(response: Any, operation: str) -> dict[str, Any]:
    _raise_for_response(response, operation)
    if response.status_code == 204 or not response.content:
        return {}
    body = response.json()
    if not isinstance(body, dict):
        raise SimulationError(f"{operation} returned a non-object JSON response")
    return body


def _raise_transient_session_error(
    response: Any,
    operation: str,
    *,
    attempts: int,
    reason: str,
) -> None:
    status_code = getattr(response, "status_code", 0)
    text = str(getattr(response, "text", ""))
    raise SimulationError(
        f"{operation} failed with transient HTTP {status_code} after {attempts} attempts "
        f"({reason}): {text[:300]}"
    )


def _default_environment_name(package: AssetPackage) -> str:
    return f"sim-{package.source_path.stem or 'asset'}"


def _launch_name(imported: SimImportResult) -> str:
    if imported.package:
        return f"sim-render-{imported.package.source_path.stem or 'asset'}"
    if imported.environment_ref:
        return f"sim-render-{imported.environment_ref.env_id}"
    return "sim-render"
