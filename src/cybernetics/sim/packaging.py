"""Local simulation asset packaging for Cybernetic Physics environments.

This module owns the local-file boundary for ``cybernetics sim``. It accepts a
file or directory, infers the root simulation artifact, and emits a bounded zip
bundle that the control plane can store as an environment version.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

CompatibilityStatus = Literal["ready_to_render", "needs_root_stage", "needs_conversion"]

USD_STAGE_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
ROBOT_DESCRIPTION_EXTENSIONS = {".urdf", ".xacro", ".sdf", ".world", ".mjcf", ".xml"}
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache"}
MANIFEST_NAME = "cybernetics_sim_asset_manifest.json"

_ROOT_NAME_PRIORITY = (
    "scene",
    "root",
    "stage",
    "world",
    "environment",
    "env",
    "main",
    "warehouse",
)


class AssetPackageError(ValueError):
    """The local asset could not be safely packaged."""


@dataclass(frozen=True)
class AssetInspection:
    source_path: Path
    root_relpath: str | None
    asset_kind: str
    compatibility_status: CompatibilityStatus
    file_count: int
    total_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "root_relpath": self.root_relpath,
            "asset_kind": self.asset_kind,
            "compatibility_status": self.compatibility_status,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }


@dataclass(frozen=True)
class AssetPackage:
    source_path: Path
    bundle_path: Path
    root_stage_relpath: str | None
    asset_kind: str
    compatibility_status: CompatibilityStatus
    bundle_sha256: str
    bundle_size_bytes: int
    file_count: int
    temporary: bool
    manifest: dict[str, Any]

    def cleanup(self) -> None:
        if self.temporary:
            try:
                self.bundle_path.unlink()
            except FileNotFoundError:
                pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "bundle_path": str(self.bundle_path),
            "root_stage_relpath": self.root_stage_relpath,
            "asset_kind": self.asset_kind,
            "compatibility_status": self.compatibility_status,
            "bundle_sha256": self.bundle_sha256,
            "bundle_size_bytes": self.bundle_size_bytes,
            "file_count": self.file_count,
            "manifest": self.manifest,
        }


def inspect_local_asset(path: str | os.PathLike[str], root_stage: str | None = None) -> AssetInspection:
    source = Path(path).expanduser().resolve()
    files = _collect_files(source)
    root_relpath = _resolve_root_relpath(source, files, root_stage)
    asset_kind = _asset_kind(root_relpath)
    return AssetInspection(
        source_path=source,
        root_relpath=root_relpath,
        asset_kind=asset_kind,
        compatibility_status=_compatibility(root_relpath),
        file_count=len(files),
        total_size_bytes=sum(file.stat().st_size for file in files),
    )


def package_local_asset(
    path: str | os.PathLike[str],
    *,
    root_stage: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    source_url: str | None = None,
) -> AssetPackage:
    source = Path(path).expanduser().resolve()
    bundle_path, temporary = _resolve_output_path(source, output_path)
    files = _collect_files(source, exclude_paths={bundle_path})
    root_relpath = _resolve_root_relpath(source, files, root_stage)
    asset_kind = _asset_kind(root_relpath)
    compatibility = _compatibility(root_relpath)
    manifest = _build_manifest(
        source,
        files,
        root_relpath,
        asset_kind,
        compatibility,
        source_url=source_url,
    )

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = _archive_name(source, file_path)
            zf.write(file_path, arcname)
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return AssetPackage(
        source_path=source,
        bundle_path=bundle_path,
        root_stage_relpath=root_relpath if compatibility == "ready_to_render" else None,
        asset_kind=asset_kind,
        compatibility_status=compatibility,
        bundle_sha256=_sha256_file(bundle_path),
        bundle_size_bytes=bundle_path.stat().st_size,
        file_count=len(files),
        temporary=temporary,
        manifest=manifest,
    )


def _collect_files(source: Path, *, exclude_paths: set[Path] | None = None) -> list[Path]:
    excludes = {path.expanduser().resolve() for path in (exclude_paths or set())}
    if not source.exists():
        raise AssetPackageError(f"asset path does not exist: {source}")
    if source.is_symlink():
        raise AssetPackageError(f"asset path must not be a symlink: {source}")
    if source.is_file():
        if source in excludes:
            raise AssetPackageError("output bundle path must not overwrite the source asset")
        return [source]
    if not source.is_dir():
        raise AssetPackageError(f"asset path must be a file or directory: {source}")

    files: list[Path] = []
    for root, dirs, names in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        root_path = Path(root)
        for name in sorted(names):
            file_path = root_path / name
            if file_path.is_symlink():
                raise AssetPackageError(f"asset bundle must not contain symlinks: {file_path}")
            if file_path.resolve() in excludes:
                continue
            if file_path.is_file():
                files.append(file_path)
    if not files:
        raise AssetPackageError(f"asset directory contains no files: {source}")
    return files


def _resolve_root_relpath(source: Path, files: list[Path], explicit: str | None) -> str | None:
    if explicit:
        rel = _clean_relpath(explicit)
        candidate = (source.parent / rel) if source.is_file() else (source / rel)
        if source.is_file() and rel != source.name:
            raise AssetPackageError(
                f"--root-stage for a single file must be {source.name!r}, got {rel!r}"
            )
        if not candidate.exists():
            raise AssetPackageError(f"root stage does not exist in asset bundle: {rel}")
        return rel

    candidates = sorted((_archive_name(source, file) for file in files), key=_root_sort_key)
    for rel in candidates:
        if Path(rel).suffix.lower() in USD_STAGE_EXTENSIONS:
            return rel
    for rel in candidates:
        if _is_robot_description_relpath(rel):
            return rel
    return None


def _archive_name(source: Path, file_path: Path) -> str:
    if source.is_file():
        rel = file_path.name
    else:
        rel = file_path.relative_to(source).as_posix()
    return _clean_relpath(rel)


def _clean_relpath(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute():
        raise AssetPackageError(f"bundle path must be relative: {value}")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise AssetPackageError(f"bundle path must not contain traversal: {value}")
    return path.as_posix()


def _root_sort_key(relpath: str) -> tuple[int, int, str]:
    path = PurePosixPath(relpath)
    stem = path.stem.lower()
    try:
        name_rank = _ROOT_NAME_PRIORITY.index(stem)
    except ValueError:
        name_rank = len(_ROOT_NAME_PRIORITY)
    return (len(path.parts), name_rank, relpath.lower())


def _asset_kind(root_relpath: str | None) -> str:
    if root_relpath is None:
        return "asset_bundle"
    suffix = Path(root_relpath).suffix.lower()
    if suffix == ".usdz":
        return "usdz_package"
    if suffix in {".usd", ".usda", ".usdc"}:
        return "usd_stage"
    if suffix == ".urdf":
        return "urdf_robot"
    if suffix == ".xacro":
        return "xacro_robot"
    if suffix in {".sdf", ".world"}:
        return "sdf_world"
    if suffix == ".mjcf" or PurePosixPath(root_relpath).name.lower() == "model.xml":
        return "mjcf_model"
    return "asset_bundle"


def _compatibility(root_relpath: str | None) -> CompatibilityStatus:
    if root_relpath is None:
        return "needs_root_stage"
    suffix = Path(root_relpath).suffix.lower()
    if suffix in USD_STAGE_EXTENSIONS:
        return "ready_to_render"
    if _is_robot_description_relpath(root_relpath):
        return "needs_conversion"
    return "needs_root_stage"


def _is_robot_description_relpath(relpath: str) -> bool:
    suffix = Path(relpath).suffix.lower()
    name = PurePosixPath(relpath).name.lower()
    return suffix in ROBOT_DESCRIPTION_EXTENSIONS and (suffix != ".xml" or name == "model.xml")


def _resolve_output_path(
    source: Path, output_path: str | os.PathLike[str] | None
) -> tuple[Path, bool]:
    if output_path is not None:
        return Path(output_path).expanduser().resolve(), False
    fd, name = tempfile.mkstemp(prefix=f"{source.stem or 'asset'}-", suffix=".bundle.zip")
    os.close(fd)
    return Path(name), True


def _build_manifest(
    source: Path,
    files: list[Path],
    root_relpath: str | None,
    asset_kind: str,
    compatibility: CompatibilityStatus,
    *,
    source_url: str | None = None,
) -> dict[str, Any]:
    source_info = {"type": "local", "name": source.name}
    if source_url:
        source_info["url"] = source_url
    return {
        "schema": "cybernetics.sim.asset-bundle/v1",
        "source": source_info,
        "root_stage_relpath": root_relpath,
        "asset_kind": asset_kind,
        "compatibility_status": compatibility,
        "files": [
            {
                "path": _archive_name(source, file),
                "size_bytes": file.stat().st_size,
                "sha256": _sha256_file(file),
            }
            for file in files
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
