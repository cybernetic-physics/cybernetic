"""Commands for uploading and converting Gaussian splats."""

from __future__ import annotations

from typing import Any

import click

from cybernetics.sim import SimulationClient, SimulationError

from ..context import CLIContext
from ..exceptions import CyberneticsCliError
from ..output import OutputBase

_GUARDRAIL_OPTIONS = (
    click.option(
        "--max-runtime-minutes",
        default=30,
        show_default=True,
        type=int,
        help="Conversion job runtime guardrail.",
    ),
    click.option(
        "--max-hourly-price",
        default=2.0,
        show_default=True,
        type=float,
        help="Max GPU $/hour for the conversion job.",
    ),
    click.option(
        "--gpu-min-vram",
        default=24,
        show_default=True,
        type=int,
        help="Minimum GPU VRAM (GB) for the conversion job.",
    ),
)


class SplatOutput(OutputBase):
    def __init__(self, title: str, payload: dict[str, Any]) -> None:
        self.title = title
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def get_title(self) -> str | None:
        return self.title

    def get_table_columns(self) -> list[str]:
        return ["Field", "Value"]

    def get_table_rows(self) -> list[list[str]]:
        return [[key, _format_value(value)] for key, value in self.payload.items()]


def _with_guardrail_options(command):
    for option in reversed(_GUARDRAIL_OPTIONS):
        command = option(command)
    return command


@click.group()
def cli() -> None:
    """Upload standard 3DGS PLY splats and convert them to ParticleField USDZ."""


@cli.command("upload")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--convert/--no-convert",
    default=True,
    show_default=True,
    help="Create a splat→ParticleField-USDZ conversion job after uploading.",
)
@click.option("--wait", is_flag=True, help="Wait for the conversion job and print artifacts.")
@click.option("--timeout-seconds", default=1800.0, show_default=True, type=float)
@_with_guardrail_options
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def upload_command(
    ctx: CLIContext | None,
    path: str,
    convert: bool,
    wait: bool,
    timeout_seconds: float,
    max_runtime_minutes: int,
    max_hourly_price: float,
    gpu_min_vram: int,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Upload a standard 3DGS Gaussian splat (.ply)."""
    client = _client(api_key=api_key, base_url=base_url)
    try:
        try:
            uploaded = client.upload_splat(path)
            payload: dict[str, Any] = dict(uploaded)
            if convert:
                job = client.create_splat_convert_job(
                    uploaded["inputPrefix"],
                    max_runtime_minutes=max_runtime_minutes,
                    max_hourly_price=max_hourly_price,
                    gpu_min_vram=gpu_min_vram,
                )
                payload["job_id"] = job.get("jobId")
                payload["job_status"] = job.get("status")
                if wait:
                    job = client.wait_for_job(
                        _require(job, "jobId"), timeout_seconds=timeout_seconds
                    )
                    payload["job_status"] = job.get("status")
                    payload.update(_artifact_fields(client, _require(job, "jobId")))
        except SimulationError as exc:
            raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    SplatOutput("Gaussian Splat Upload", payload).print((ctx or CLIContext()).format)


@cli.command("convert")
@click.argument("input_uri")
@click.option("--wait", is_flag=True, help="Wait for the conversion job and print artifacts.")
@click.option("--timeout-seconds", default=1800.0, show_default=True, type=float)
@_with_guardrail_options
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def convert_command(
    ctx: CLIContext | None,
    input_uri: str,
    wait: bool,
    timeout_seconds: float,
    max_runtime_minutes: int,
    max_hourly_price: float,
    gpu_min_vram: int,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Create a conversion job for an already-uploaded splat input URI."""
    client = _client(api_key=api_key, base_url=base_url)
    try:
        try:
            job = client.create_splat_convert_job(
                input_uri,
                max_runtime_minutes=max_runtime_minutes,
                max_hourly_price=max_hourly_price,
                gpu_min_vram=gpu_min_vram,
            )
            payload = {"job_id": job.get("jobId"), "job_status": job.get("status")}
            if wait:
                job = client.wait_for_job(_require(job, "jobId"), timeout_seconds=timeout_seconds)
                payload["job_status"] = job.get("status")
                payload.update(_artifact_fields(client, _require(job, "jobId")))
        except SimulationError as exc:
            raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    SplatOutput("Gaussian Splat Convert", payload).print((ctx or CLIContext()).format)


@cli.command("status")
@click.argument("job_id")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def status_command(
    ctx: CLIContext | None,
    job_id: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Show a conversion job's status (and artifacts once it succeeds)."""
    client = _client(api_key=api_key, base_url=base_url)
    try:
        try:
            job = client.get_job(job_id)
            payload = {
                "job_id": job.get("jobId"),
                "job_status": job.get("status"),
                "error_message": job.get("errorMessage"),
            }
            if str(job.get("status", "")).lower() in {"completed", "succeeded"}:
                payload.update(_artifact_fields(client, job_id))
        except SimulationError as exc:
            raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    SplatOutput("Gaussian Splat Job", payload).print((ctx or CLIContext()).format)


def _artifact_fields(client: SimulationClient, job_id: str) -> dict[str, Any]:
    artifacts = client.job_artifacts(job_id)
    downloads = artifacts.get("downloadUrls")
    fields: dict[str, Any] = {"artifacts": artifacts.get("artifacts")}
    if isinstance(downloads, dict) and downloads.get("usdz"):
        fields["usdz_download_url"] = downloads["usdz"]
    return fields


def _require(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CyberneticsCliError(f"response is missing {key!r}")
    return value


def _client(*, api_key: str | None, base_url: str | None) -> SimulationClient:
    try:
        return SimulationClient(api_key=api_key, base_url=base_url)
    except SimulationError as exc:
        raise CyberneticsCliError(str(exc)) from exc


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, sort_keys=True)
    return str(value)
