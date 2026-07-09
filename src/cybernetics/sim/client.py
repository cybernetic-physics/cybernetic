"""Synchronous Simulation Asset Rendering MVP client."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from cybernetics.lib.credentials import resolve_api_key, resolve_base_url

from .packaging import AssetPackage, package_local_asset

DEFAULT_BASE_URL = "https://api.cyberneticphysics.com"
SIMULATION_ASSET_REF_SCHEMA_VERSION = "simulation-asset-ref/v1"
_READY_STATUSES = {"running", "idle"}
_TERMINAL_STATUSES = {"failed", "terminated", "stopped", "error", "snapshot_failed"}
AssetRefKind = Literal["environment_version", "local_bundle", "catalog_asset"]


class SimulationError(RuntimeError):
    """A simulation asset operation failed."""


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

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "SimulationClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

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
            raise SimulationError(
                f"{package.asset_kind} is {package.compatibility_status}; "
                "the MVP can render USD/USDZ assets and package robot descriptions for later conversion."
            )
        if keep_bundle:
            package = _with_persistent_bundle(package)

        if package.compatibility_status != "ready_to_render":
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
                "description": description
                or f"Imported by cybernetics sim from {Path(ref).name}",
            },
        )
        env_id = _require_str(environment, "id")

        version_response = self._request(
            "POST",
            f"/v1/envs/{env_id}/versions",
            json_body={
                "notes": notes or f"Imported from {Path(ref).name}",
                "rootStageRelpath": package.root_stage_relpath,
                "upload": {"type": "bundle_zip"},
            },
        )
        version = _require_dict(version_response, "version")
        version_id = _require_str(version, "id")
        upload = version_response.get("upload")
        if not isinstance(upload, dict):
            raise SimulationError("control plane did not return a bundle upload target")
        self._upload_bundle(upload, package.bundle_path)

        finalized = self._request(
            "POST",
            f"/v1/envs/{env_id}/versions/{version_id}/finalize",
            json_body={
                "contentSha256": package.bundle_sha256,
                "bundleSizeBytes": package.bundle_size_bytes,
                "rootStageRelpath": package.root_stage_relpath,
            },
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
            raise SimulationError(
                "Public /sim artifact pages are not implemented in the MVP."
            )

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
        if wait:
            preview_url = self.preview_url(launch.session_id)
            status = "preview_ready"
            if out is not None:
                preview_path = Path(out).expanduser().resolve()
                self.download_preview(launch.session_id, preview_path)

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
        deadline = time.monotonic() + timeout_seconds
        last = "not checked"
        while True:
            session = self._request("GET", f"/v1/sessions/{session_id}")
            status = str(session.get("status", "")).lower()
            if status in _TERMINAL_STATUSES:
                raise SimulationError(f"session {session_id} entered terminal status {status!r}")
            if status in _READY_STATUSES and _bridge_ready(session):
                return session
            last = f"status={status or 'unknown'}"
            if time.monotonic() >= deadline:
                raise SimulationError(f"session {session_id} was not ready after {timeout_seconds}s ({last})")
            time.sleep(poll_interval_seconds)

    def preview_url(self, session_id: str, *, ttl_seconds: int = 3600, quality: int = 90) -> str:
        grant = self._request(
            "POST",
            f"/v1/sessions/{session_id}/cua-grant",
            json_body={"ttlSeconds": ttl_seconds},
        )
        neko_url = _require_str(grant, "neko_url")
        http_base = _neko_http_base(neko_url)
        return f"{http_base}/api/shot.jpg?quality={quality}"

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
        _raise_for_response(response, f"GET {url}")
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def stop_session(self, session_id: str) -> None:
        self._request("POST", f"/v1/sessions/{session_id}/stop", json_body={})

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
                version={"id": value.version_id, "envId": value.env_id} if value.version_id else None,
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
        response = self._client.request(
            method,
            path,
            json=json_body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        _raise_for_response(response, f"{method} {path}")
        if response.status_code == 204 or not response.content:
            return {}
        body = response.json()
        if not isinstance(body, dict):
            raise SimulationError(f"{method} {path} returned a non-object JSON response")
        return body

    def _upload_bundle(self, upload: dict[str, Any], bundle_path: Path) -> None:
        url = upload.get("url") or upload.get("putUrl")
        if not isinstance(url, str) or not url:
            raise SimulationError("control plane upload response did not include an upload URL")

        fields = upload.get("fields")
        if isinstance(fields, dict):
            with bundle_path.open("rb") as handle:
                response = self._client.post(
                    url,
                    data={str(k): str(v) for k, v in fields.items()},
                    files={"file": ("bundle.zip", handle, "application/zip")},
                )
            _raise_for_response(response, f"POST {url}")
            return

        # Future-compatible fallback if the API switches to a real signed PUT.
        with bundle_path.open("rb") as handle:
            response = self._client.put(
                url,
                content=handle.read(),
                headers={"content-type": "application/zip"},
            )
        _raise_for_response(response, f"PUT {url}")

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


def _default_environment_name(package: AssetPackage) -> str:
    return f"sim-{package.source_path.stem or 'asset'}"


def _launch_name(imported: SimImportResult) -> str:
    if imported.package:
        return f"sim-render-{imported.package.source_path.stem or 'asset'}"
    if imported.environment_ref:
        return f"sim-render-{imported.environment_ref.env_id}"
    return "sim-render"
