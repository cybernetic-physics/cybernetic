"""Commands for importing and launching simulation assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from cybernetics.sim import (
    AssetPackageError,
    SimImportResult,
    SimLaunchResult,
    SimRenderResult,
    SimulationClient,
    SimulationError,
    inspect_local_asset,
)

from ..context import CLIContext
from ..exceptions import CyberneticsCliError
from ..output import OutputBase


class SimOutput(OutputBase):
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


@click.group()
def cli() -> None:
    """Import, render, and launch simulation assets."""


@cli.command("inspect")
@click.argument("asset_ref")
@click.option("--root-stage", default=None, help="Root stage path inside the asset folder.")
@click.pass_obj
def inspect_command(ctx: CLIContext | None, asset_ref: str, root_stage: str | None) -> None:
    """Inspect a local simulation asset without uploading it."""
    try:
        inspection = inspect_local_asset(asset_ref, root_stage=root_stage)
    except AssetPackageError as exc:
        raise CyberneticsCliError(str(exc)) from exc
    SimOutput("Simulation Asset", inspection.to_dict()).print((ctx or CLIContext()).format)


@cli.command("import")
@click.argument("asset_ref")
@click.option("--name", default=None, help="Environment display name.")
@click.option("--description", default=None, help="Environment description.")
@click.option("--root-stage", default=None, help="Root USD stage path inside the asset folder.")
@click.option("--notes", default=None, help="Environment version notes.")
@click.option("--bundle-path", default=None, type=click.Path(dir_okay=False), help="Keep bundle zip here.")
@click.option("--keep-bundle", is_flag=True, help="Do not remove the generated bundle zip.")
@click.option("--source-url", default=None, help="Explicit source URL to keep as safe provenance.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def import_command(
    ctx: CLIContext | None,
    asset_ref: str,
    name: str | None,
    description: str | None,
    root_stage: str | None,
    notes: str | None,
    bundle_path: str | None,
    keep_bundle: bool,
    source_url: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Package and upload a local USD/USDZ or Gaussian-splat asset as an environment version.

    Gaussian splats upload as needs_conversion source bundles; convert them to
    a renderable NuRec USDZ with `cybernetics splat upload --convert`.
    """
    client = _client(api_key=api_key, base_url=base_url)
    try:
        try:
            result = client.import_asset(
                asset_ref,
                name=name,
                description=description,
                root_stage=root_stage,
                notes=notes,
                bundle_path=bundle_path,
                keep_bundle=keep_bundle or bundle_path is not None,
                source_url=source_url,
            )
        except (AssetPackageError, SimulationError) as exc:
            raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    _print_import(result, ctx)


@cli.command("launch")
@click.argument("asset_ref")
@click.option("--name", default=None, help="Session display name.")
@click.option("--root-stage", default=None, help="Root USD stage path inside the asset folder.")
@click.option("--gpu-spec", default=None, help="GPU selector passed to the session API.")
@click.option("--max-runtime-minutes", default=None, type=int, help="Session max runtime.")
@click.option("--idle-timeout-minutes", default=None, type=int, help="Session idle timeout.")
@click.option("--wait", is_flag=True, help="Wait until the session and Isaac bridge are ready.")
@click.option("--timeout-seconds", default=900.0, show_default=True, type=float)
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def launch_command(
    ctx: CLIContext | None,
    asset_ref: str,
    name: str | None,
    root_stage: str | None,
    gpu_spec: str | None,
    max_runtime_minutes: int | None,
    idle_timeout_minutes: int | None,
    wait: bool,
    timeout_seconds: float,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Launch a hosted Isaac session from a local asset or environment ref."""
    client = _client(api_key=api_key, base_url=base_url)
    try:
        try:
            result = client.launch(
                asset_ref,
                name=name,
                root_stage=root_stage,
                gpu_spec=gpu_spec,
                max_runtime_minutes=max_runtime_minutes,
                idle_timeout_minutes=idle_timeout_minutes,
                wait=wait,
                timeout_seconds=timeout_seconds,
            )
        except (AssetPackageError, SimulationError) as exc:
            raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    _print_launch(result, ctx)


@cli.command("render")
@click.argument("asset_ref")
@click.option("--name", default=None, help="Environment/session display name.")
@click.option("--root-stage", default=None, help="Root USD stage path inside the asset folder.")
@click.option("--out", default=None, type=click.Path(dir_okay=False), help="Write preview JPG here.")
@click.option("--wait", is_flag=True, help="Wait for the live session before creating a preview URL.")
@click.option(
    "--stop-after-preview",
    is_flag=True,
    help="Stop the render session after creating the preview.",
)
@click.option(
    "--public",
    "public_requested",
    is_flag=True,
    help="Reserved for public /sim artifacts; not available in the MVP.",
)
@click.option("--timeout-seconds", default=900.0, show_default=True, type=float)
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def render_command(
    ctx: CLIContext | None,
    asset_ref: str,
    name: str | None,
    root_stage: str | None,
    out: str | None,
    wait: bool,
    stop_after_preview: bool,
    public_requested: bool,
    timeout_seconds: float,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Create a hosted render session and optional live preview JPG."""
    if public_requested:
        raise CyberneticsCliError(
            "Public /sim artifact pages are not implemented in the MVP.",
            "Run without --public to create an authenticated launch/preview session.",
        )
    client = _client(api_key=api_key, base_url=base_url)
    try:
        try:
            result = client.render(
                asset_ref,
                name=name,
                root_stage=root_stage,
                out=out,
                wait=wait,
                keep_session=not stop_after_preview,
                timeout_seconds=timeout_seconds,
            )
        except (AssetPackageError, SimulationError) as exc:
            raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    _print_render(result, ctx)


def _client(*, api_key: str | None, base_url: str | None) -> SimulationClient:
    try:
        return SimulationClient(api_key=api_key, base_url=base_url)
    except SimulationError as exc:
        raise CyberneticsCliError(str(exc)) from exc


def _print_import(result: SimImportResult, ctx: CLIContext | None) -> None:
    if result.package and result.package.compatibility_status != "ready_to_render":
        # Splat bundles upload before conversion; keep the real status visible.
        status = result.package.compatibility_status
    else:
        status = "ready"
    payload = {
        "status": status,
        "asset_ref": result.asset_ref,
        "environment_ref": result.environment_ref.uri if result.environment_ref else None,
        "env_id": result.env_id,
        "version_id": result.version_id,
        "asset_kind": result.package.asset_kind if result.package else None,
        "root_stage_relpath": result.package.root_stage_relpath if result.package else None,
        "bundle_path": str(result.package.bundle_path) if result.package else None,
    }
    try:
        payload["simulation_asset_ref"] = result.to_asset_ref().to_dict()
    except SimulationError:
        payload["simulation_asset_ref"] = None
    SimOutput("Simulation Import", payload).print((ctx or CLIContext()).format)


def _print_launch(result: SimLaunchResult, ctx: CLIContext | None) -> None:
    payload = {
        "session_id": result.session_id,
        "session_url": result.session_url,
        "viewer_url": result.viewer_url,
        "preview_url": result.preview_url,
        "status": result.session.get("status"),
    }
    SimOutput("Simulation Launch", payload).print((ctx or CLIContext()).format)


def _print_render(result: SimRenderResult, ctx: CLIContext | None) -> None:
    payload = {
        "status": result.status,
        "preview_path": str(result.preview_path) if result.preview_path else None,
        "preview_url": result.preview_url,
        "launch_url": result.launch_url,
        "public_url": result.public_url,
        "environment_ref": (
            result.import_result.environment_ref.uri
            if result.import_result.environment_ref
            else None
        ),
        "session_id": result.launch_result.session_id if result.launch_result else None,
    }
    SimOutput("Simulation Render", payload).print((ctx or CLIContext()).format)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, sort_keys=True)
    return str(value)
