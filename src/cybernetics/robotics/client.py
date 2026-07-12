"""Synchronous client for managed robotics evaluations and asset bundles."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional
from zipfile import ZipFile

from cybernetics.lib.credentials import resolve_api_key, resolve_base_url

from .experiments import (
    ROBOTICS_CATALOG_SCHEMA_VERSION,
    RoboticsBenchmarkTemplate,
    RoboticsPreflight,
    experiment_request,
)
from .runtime_contracts import AssetBundleRef, RoboticsJobSpec

DEFAULT_BASE_URL = "https://api.cyberneticphysics.com"


class RobotEvalsError(RuntimeError):
    """A managed robotics API operation failed."""


class RobotEvalsClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            raise RobotEvalsError(
                "No API key found. Run 'cybernetics auth login' or set CYBERNETICS_API_KEY."
            )
        self.api_key = resolved_key
        self.base_url = (resolve_base_url(base_url) or DEFAULT_BASE_URL).rstrip("/")
        self._owns_client = http_client is None
        if http_client is None:
            import httpx

            http_client = httpx.Client(base_url=self.base_url, timeout=180.0)
        self._client = http_client

    def submit(
        self,
        job: RoboticsJobSpec,
        *,
        budget_usd_limit: Optional[float] = None,
        expected_job_hash: Optional[str] = None,
        ci_context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job": job.to_dict(),
            "ciContext": dict(ci_context or {}),
        }
        if budget_usd_limit is not None:
            if budget_usd_limit <= 0:
                raise RobotEvalsError("budget_usd_limit must be positive")
            payload["budgetUsdLimit"] = float(budget_usd_limit)
        if expected_job_hash is not None:
            if expected_job_hash != job.job_hash():
                raise RobotEvalsError("expected_job_hash does not match the normalized job")
            payload["expectedJobHash"] = expected_job_hash
        return self._request("POST", "/v1/eval/runs", json_body=payload)

    def list_benchmarks(self) -> list[RoboticsBenchmarkTemplate]:
        body = self._request("GET", "/v1/eval/robotics/catalog")
        if body.get("schemaVersion") != ROBOTICS_CATALOG_SCHEMA_VERSION:
            raise RobotEvalsError("robotics benchmark catalog has an unsupported schema version")
        return [
            RoboticsBenchmarkTemplate.from_dict(item)
            for item in _object_list(body.get("items"), "list robotics benchmarks")
        ]

    def get_benchmark(self, benchmark_id: str) -> RoboticsBenchmarkTemplate:
        for benchmark in self.list_benchmarks():
            if benchmark.id == benchmark_id:
                return benchmark
        raise RobotEvalsError(f"robotics benchmark not found: {benchmark_id}")

    def compose(
        self,
        benchmark_id: str,
        *,
        policy_id: Optional[str] = None,
        episode_start: int = 0,
        episodes: Optional[int] = None,
        root_seed: Optional[int] = None,
        vector_width: Optional[int] = None,
        max_steps: Optional[int] = None,
        job_name: Optional[str] = None,
        video: Optional[bool] = None,
        observations: Optional[bool] = None,
        actions: Optional[bool] = None,
        predictions: Optional[bool] = None,
        failure_clips: Optional[bool] = None,
        dataset_export: Optional[str] = None,
        budget_usd_limit: Optional[float] = None,
    ) -> RoboticsPreflight:
        """Compose and preflight one benchmark episode shard on the control plane."""

        benchmark = self.get_benchmark(benchmark_id)
        policy = benchmark.policy(policy_id)
        experiment = experiment_request(
            benchmark_id=benchmark.id,
            policy_id=policy.id,
            episode_start=episode_start,
            episodes=episodes,
            root_seed=root_seed,
            vector_width=vector_width,
            max_steps=max_steps,
            job_name=job_name,
            video=video,
            observations=observations,
            actions=actions,
            predictions=predictions,
            failure_clips=failure_clips,
            dataset_export=dataset_export,
        )
        payload: dict[str, Any] = {"experiment": experiment}
        if budget_usd_limit is not None:
            if budget_usd_limit <= 0:
                raise RobotEvalsError("budget_usd_limit must be positive")
            payload["budgetUsdLimit"] = float(budget_usd_limit)
        return RoboticsPreflight.from_dict(
            self._request("POST", "/v1/eval/robotics/preflight", json_body=payload)
        )

    def preflight(
        self,
        job: RoboticsJobSpec,
        *,
        budget_usd_limit: Optional[float] = None,
    ) -> RoboticsPreflight:
        payload: dict[str, Any] = {"job": job.to_dict()}
        if budget_usd_limit is not None:
            if budget_usd_limit <= 0:
                raise RobotEvalsError("budget_usd_limit must be positive")
            payload["budgetUsdLimit"] = float(budget_usd_limit)
        return RoboticsPreflight.from_dict(
            self._request("POST", "/v1/eval/robotics/preflight", json_body=payload)
        )

    def submit_preflight(
        self,
        preflight: RoboticsPreflight,
        *,
        budget_usd_limit: Optional[float] = None,
        ci_context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        job = preflight.require_launchable_job()
        return self.submit(
            job,
            budget_usd_limit=budget_usd_limit,
            expected_job_hash=preflight.job_hash,
            ci_context=ci_context,
        )

    def list_runs(self) -> list[dict[str, Any]]:
        body = self._request("GET", "/v1/eval/runs")
        return _object_list(body.get("items"), "list eval runs")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/eval/runs/{run_id}")

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        body = self.get_run(run_id)
        return _object_list(body.get("events"), "list eval run events")

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        body = self.get_run(run_id)
        return _object_list(body.get("artifacts"), "list eval run artifacts")

    def cancel(self, run_id: str) -> None:
        self._request("POST", f"/v1/eval/runs/{run_id}/cancel", json_body={})

    def artifact_download_url(self, run_id: str, artifact_id: int) -> str:
        body = self._request("GET", f"/v1/eval/runs/{run_id}/artifacts/{int(artifact_id)}/download")
        url = body.get("url")
        if not isinstance(url, str) or not url:
            raise RobotEvalsError("artifact download response did not include a URL")
        return url

    def upload_asset_bundle(
        self,
        path: str | Path,
        *,
        name: Optional[str] = None,
        media_type: str = "application/zip",
        source: Optional[Mapping[str, Any]] = None,
        license: Optional[str] = None,
        unpack: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AssetBundleRef:
        bundle_path = Path(path).resolve()
        if not bundle_path.is_file():
            raise RobotEvalsError(f"asset bundle does not exist: {bundle_path}")
        size = bundle_path.stat().st_size
        digest = _sha256_file(bundle_path)
        bundle_metadata = dict(metadata or {})
        if unpack:
            if media_type not in {"application/zip", "application/x-zip-compressed"}:
                raise RobotEvalsError("unpack=True currently requires a ZIP asset bundle")
            bundle_metadata.update(_inspect_zip(bundle_path))
            bundle_metadata.update({"unpack": True, "archive_format": "zip"})
        create_payload: dict[str, Any] = {
            "name": name or bundle_path.name,
            "mediaType": media_type,
            "sizeBytes": size,
            "contentSha256": digest,
            "source": dict(source or {"kind": "local", "name": bundle_path.name}),
            "metadata": bundle_metadata,
        }
        if license is not None:
            create_payload["license"] = license
        body = self._request(
            "POST",
            "/v1/eval/asset-bundles",
            json_body=create_payload,
        )
        upload = body.get("upload")
        bundle_id = body.get("id")
        uri = body.get("uri")
        if (
            not isinstance(upload, Mapping)
            or not isinstance(bundle_id, str)
            or not isinstance(uri, str)
        ):
            raise RobotEvalsError("asset bundle create response is incomplete")
        self._upload(upload, bundle_path, media_type)
        finalized = self._request(
            "POST", f"/v1/eval/asset-bundles/{bundle_id}/finalize", json_body={}
        )
        verified_digest = finalized.get("contentSha256")
        if not isinstance(verified_digest, str):
            raise RobotEvalsError("asset bundle finalize response omitted the verified SHA-256")
        if verified_digest != digest:
            raise RobotEvalsError("asset bundle server SHA-256 does not match the local file")
        verified_metadata = finalized.get("metadata")
        if isinstance(verified_metadata, Mapping):
            bundle_metadata = dict(verified_metadata)
        return AssetBundleRef(
            uri=uri,
            content_sha256=digest,
            media_type=media_type,
            size_bytes=size,
            source=dict(source or {"kind": "local", "name": bundle_path.name}),
            license=license,
            metadata=bundle_metadata,
        )

    def list_asset_bundles(self) -> list[dict[str, Any]]:
        body = self._request("GET", "/v1/eval/asset-bundles")
        return _object_list(body.get("items"), "list robotics asset bundles")

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "RobotEvalsClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
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
            raise RobotEvalsError(f"{method} {path} returned non-object JSON")
        return body

    def _upload(self, upload: Mapping[str, Any], path: Path, media_type: str) -> None:
        url = upload.get("url")
        fields = upload.get("fields")
        if not isinstance(url, str) or not url:
            raise RobotEvalsError("asset upload response did not include a URL")
        if not isinstance(fields, Mapping):
            raise RobotEvalsError("asset upload response did not include presigned POST fields")
        with path.open("rb") as handle:
            response = self._client.post(
                url,
                data={str(key): str(value) for key, value in fields.items()},
                files={"file": (path.name, handle, media_type)},
            )
        _raise_for_response(response, f"POST {url}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_zip(path: Path) -> dict[str, int]:
    file_count = 0
    packed_size = 0
    unpacked_size = 0
    paths: set[str] = set()
    with ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            if item.flag_bits & 0x1:
                raise RobotEvalsError(f"asset ZIP contains an encrypted entry: {item.filename}")
            if "\\" in item.filename or "\x00" in item.filename:
                raise RobotEvalsError(f"asset ZIP contains unsafe path: {item.filename}")
            relpath = PurePosixPath(item.filename)
            parts = item.filename.split("/")
            if (
                relpath.is_absolute()
                or ".." in relpath.parts
                or any(not part or part == "." for part in parts)
                or any(any(ord(char) < 32 for char in part) for part in parts)
            ):
                raise RobotEvalsError(f"asset ZIP contains unsafe path: {item.filename}")
            normalized = relpath.as_posix()
            if normalized in paths:
                raise RobotEvalsError(f"asset ZIP contains a duplicate path: {item.filename}")
            paths.add(normalized)
            mode = (item.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RobotEvalsError(f"asset ZIP contains a symlink: {item.filename}")
            file_count += 1
            packed_size += item.compress_size
            unpacked_size += item.file_size
            if file_count > 100_000:
                raise RobotEvalsError("asset ZIP contains more than 100000 files")
            if unpacked_size > 200 * 1024 * 1024 * 1024:
                raise RobotEvalsError("asset ZIP expands beyond 200 GiB")
            ratio = item.file_size / max(1, item.compress_size)
            if ratio > 200:
                raise RobotEvalsError(f"asset ZIP compression ratio exceeds 200:1: {item.filename}")
    if file_count == 0:
        raise RobotEvalsError("asset ZIP contains no files")
    if unpacked_size / max(1, packed_size) > 200:
        raise RobotEvalsError("asset ZIP aggregate compression ratio exceeds 200:1")
    return {
        "file_count": file_count,
        "packed_size_bytes": packed_size,
        "unpacked_size_bytes": unpacked_size,
    }


def _raise_for_response(response: Any, operation: str) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:
        detail = ""
        try:
            body = response.json()
            detail = str(body.get("message") or body.get("error") or "")
        except Exception:
            detail = str(getattr(response, "text", ""))[:500]
        raise RobotEvalsError(f"{operation} failed ({response.status_code}): {detail}") from exc


def _object_list(value: Any, operation: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise RobotEvalsError(f"{operation} returned an invalid item list")
    return [dict(item) for item in value]
