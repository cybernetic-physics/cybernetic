"""Local simulation asset packaging for Cybernetic Physics environments.

This module owns the local-file boundary for ``cybernetics sim``. It accepts a
file or directory, infers the root simulation artifact, and emits a bounded zip
bundle that the control plane can store as an environment version.

Gaussian splats (3DGS ``.ply`` plus the ``.spz``/``.splat``/``.ksplat``
containers) are recognized as first-class local assets. They package like any
other bundle but stay ``needs_conversion``. The hosted conversion API currently
accepts only standard 3DGS PLY and emits an OpenUSD ParticleField USDZ.
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
# Binary Gaussian-splat containers are identified by extension alone; ``.ply``
# is ambiguous (mesh vs 3DGS) and requires header sniffing.
GAUSSIAN_SPLAT_BINARY_EXTENSIONS = {".spz", ".splat", ".ksplat"}
# Per-vertex properties emitted by the INRIA 3DGS trainer (and everything
# downstream of it: 3DGRUT, gsplat, Postshot, SuperSplat exports). A PLY with
# all of these is a Gaussian splat, not a mesh scan.
_GAUSSIAN_PLY_MARKER_PROPS = frozenset({"f_dc_0", "opacity", "scale_0", "rot_0"})
_PLY_HEADER_MAX_BYTES = 64 * 1024
HOSTED_SPLAT_MAX_BYTES = 256 * 1024 * 1024
HOSTED_SPLAT_MAX_GAUSSIANS = 1_000_000
HOSTED_SPLAT_MAX_VERTEX_PROPERTIES = 62
_HOSTED_SPLAT_SUPPORTED_TYPES = frozenset({"float", "float32"})
_HOSTED_SPLAT_REQUIRED_PROPS = frozenset(
    {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
)
_HOSTED_SPLAT_ALLOWED_PROPS = (
    _HOSTED_SPLAT_REQUIRED_PROPS
    | frozenset({"nx", "ny", "nz"})
    | frozenset(f"f_rest_{index}" for index in range(45))
)
_HOSTED_SPLAT_VALID_REST_COUNTS = frozenset({0, 9, 24, 45})
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
class HostedSplatPly:
    size_bytes: int
    gaussian_count: int
    property_count: int
    spherical_harmonics_degree: int


def validate_hosted_splat_ply(path: Path) -> HostedSplatPly:
    """Fail before presign unless ``path`` meets the hosted converter contract."""
    if path.suffix.lower() != ".ply" or not path.is_file():
        raise AssetPackageError("hosted splat conversion requires one .ply file")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise AssetPackageError("hosted splat PLY must be non-empty")
    if size_bytes > HOSTED_SPLAT_MAX_BYTES:
        raise AssetPackageError("hosted splat PLY must not exceed 256 MiB")
    try:
        with path.open("rb") as handle:
            raw = handle.read(_PLY_HEADER_MAX_BYTES)
    except OSError as exc:
        raise AssetPackageError(f"could not read splat PLY: {path.name}") from exc
    marker_offsets = [
        offset
        for marker in (b"end_header\n", b"end_header\r\n")
        if (offset := raw.find(marker)) >= 0
    ]
    if not marker_offsets:
        raise AssetPackageError("splat PLY header is missing end_header within 64 KiB")
    try:
        lines = raw[: min(marker_offsets)].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise AssetPackageError("splat PLY header must be ASCII") from exc
    if not lines or lines[0].strip() != "ply":
        raise AssetPackageError("splat input is missing the PLY magic header")

    ply_format = ""
    gaussian_count: int | None = None
    properties: list[str] = []
    in_vertices = False
    for raw_line in lines[1:]:
        parts = raw_line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            ply_format = parts[1]
        elif parts[0] == "element":
            in_vertices = len(parts) == 3 and parts[1] == "vertex"
            if in_vertices:
                if gaussian_count is not None:
                    raise AssetPackageError("splat PLY must declare exactly one vertex element")
                count_token = parts[2]
                if not count_token.isascii() or not count_token.isdigit():
                    raise AssetPackageError(
                        "splat PLY Gaussian count must be an ASCII decimal integer"
                    )
                if len(count_token) > len(str(HOSTED_SPLAT_MAX_GAUSSIANS)):
                    raise AssetPackageError("splat PLY Gaussian count exceeds parser limits")
                try:
                    gaussian_count = int(count_token)
                except ValueError as exc:
                    raise AssetPackageError(
                        "splat PLY Gaussian count exceeds parser limits"
                    ) from exc
        elif parts[0] == "property" and in_vertices:
            if (
                len(parts) != 3
                or parts[1] == "list"
                or parts[1] not in _HOSTED_SPLAT_SUPPORTED_TYPES
            ):
                raise AssetPackageError(
                    "hosted splat vertex properties must be scalar float or float32"
                )
            properties.append(parts[2])

    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise AssetPackageError(f"unsupported splat PLY format: {ply_format or 'missing'}")
    if gaussian_count is None or gaussian_count <= 0:
        raise AssetPackageError("splat PLY must contain at least one Gaussian")
    if gaussian_count > HOSTED_SPLAT_MAX_GAUSSIANS:
        raise AssetPackageError("hosted splat PLY must not exceed 1,000,000 Gaussians")
    if len(properties) > HOSTED_SPLAT_MAX_VERTEX_PROPERTIES:
        raise AssetPackageError("hosted splat PLY must not exceed 62 vertex properties")
    if len(set(properties)) != len(properties):
        raise AssetPackageError("splat PLY vertex property names must be unique")
    unsupported = sorted(set(properties).difference(_HOSTED_SPLAT_ALLOWED_PROPS))
    if unsupported:
        raise AssetPackageError(f"splat PLY contains unsupported vertex properties: {unsupported}")
    missing = sorted(_HOSTED_SPLAT_REQUIRED_PROPS.difference(properties))
    if missing:
        raise AssetPackageError(f"splat PLY is missing required 3DGS properties: {missing}")
    try:
        rest_indices = sorted(
            int(name.removeprefix("f_rest_")) for name in properties if name.startswith("f_rest_")
        )
    except ValueError as exc:
        raise AssetPackageError("splat PLY f_rest suffixes must be integers") from exc
    if len(rest_indices) not in _HOSTED_SPLAT_VALID_REST_COUNTS or rest_indices != list(
        range(len(rest_indices))
    ):
        raise AssetPackageError(
            "splat PLY f_rest properties must be contiguous for SH degree 0, 1, 2, or 3"
        )
    degree = {0: 0, 9: 1, 24: 2, 45: 3}[len(rest_indices)]
    return HostedSplatPly(size_bytes, gaussian_count, len(properties), degree)


def detect_gaussian_splat_format(path: Path) -> str | None:
    """Return the Gaussian-splat container format for ``path``, or ``None``.

    ``.spz``/``.splat``/``.ksplat`` are splat-only extensions. ``.ply`` is
    sniffed: only files whose vertex element carries the 3DGS training
    properties (``f_dc_0``, ``opacity``, ``scale_0``, ``rot_0``) qualify —
    a photogrammetry mesh PLY does not.
    """
    suffix = path.suffix.lower()
    if suffix in GAUSSIAN_SPLAT_BINARY_EXTENSIONS:
        return suffix[1:]
    if suffix != ".ply":
        return None
    header = _parse_ply_header(path)
    if header is None:
        return None
    properties, _ = header
    return "ply" if properties >= _GAUSSIAN_PLY_MARKER_PROPS else None


def _parse_ply_header(path: Path) -> tuple[frozenset[str], int | None] | None:
    """Parse a PLY header into (vertex property names, vertex count).

    Returns ``None`` when the file is not a parseable PLY. Reads at most
    ``_PLY_HEADER_MAX_BYTES`` so multi-GB splats stay cheap to inspect.
    """
    try:
        with path.open("rb") as handle:
            raw = handle.read(_PLY_HEADER_MAX_BYTES)
    except OSError:
        return None
    end = raw.find(b"end_header")
    if not raw.startswith(b"ply") or end < 0:
        return None
    lines = raw[:end].decode("ascii", errors="replace").splitlines()

    properties: set[str] = set()
    vertex_count: int | None = None
    in_vertex_element = False
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex_element:
                try:
                    vertex_count = int(parts[2])
                except ValueError:
                    vertex_count = None
        elif parts[0] == "property" and in_vertex_element and len(parts) >= 3:
            properties.add(parts[-1])
    return frozenset(properties), vertex_count


def _splat_manifest_info(root_path: Path | None) -> dict[str, Any] | None:
    """Manifest/inspection metadata for a splat root artifact, if it is one."""
    if root_path is None:
        return None
    splat_format = detect_gaussian_splat_format(root_path)
    if splat_format is None:
        return None
    info: dict[str, Any] = {"format": splat_format}
    if splat_format == "ply":
        header = _parse_ply_header(root_path)
        if header is not None and header[1] is not None:
            info["gaussian_count"] = header[1]
    return info


@dataclass(frozen=True)
class AssetInspection:
    source_path: Path
    root_relpath: str | None
    asset_kind: str
    compatibility_status: CompatibilityStatus
    file_count: int
    total_size_bytes: int
    splat: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "root_relpath": self.root_relpath,
            "asset_kind": self.asset_kind,
            "compatibility_status": self.compatibility_status,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "splat": self.splat,
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


def inspect_local_asset(
    path: str | os.PathLike[str], root_stage: str | None = None
) -> AssetInspection:
    source = Path(path).expanduser().resolve()
    files = _collect_files(source)
    root_relpath = _resolve_root_relpath(source, files, root_stage)
    root_path = _root_path(source, root_relpath)
    asset_kind = _asset_kind(root_relpath, root_path)
    return AssetInspection(
        source_path=source,
        root_relpath=root_relpath,
        asset_kind=asset_kind,
        compatibility_status=_compatibility(root_relpath, root_path),
        file_count=len(files),
        total_size_bytes=sum(file.stat().st_size for file in files),
        splat=_splat_manifest_info(root_path),
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
    root_path = _root_path(source, root_relpath)
    asset_kind = _asset_kind(root_relpath, root_path)
    compatibility = _compatibility(root_relpath, root_path)
    manifest = _build_manifest(
        source,
        files,
        root_relpath,
        asset_kind,
        compatibility,
        source_url=source_url,
        splat_info=_splat_manifest_info(root_path),
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

    by_relpath = {_archive_name(source, file): file for file in files}
    candidates = sorted(by_relpath, key=_root_sort_key)
    for rel in candidates:
        if Path(rel).suffix.lower() in USD_STAGE_EXTENSIONS:
            return rel
    for rel in candidates:
        if _is_robot_description_relpath(rel):
            return rel
    for rel in candidates:
        if detect_gaussian_splat_format(by_relpath[rel]) is not None:
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


def _root_path(source: Path, root_relpath: str | None) -> Path | None:
    if root_relpath is None:
        return None
    return source if source.is_file() else source / root_relpath


def _asset_kind(root_relpath: str | None, root_path: Path | None = None) -> str:
    if root_relpath is None:
        return "asset_bundle"
    splat_format = detect_gaussian_splat_format(root_path) if root_path else None
    if splat_format is not None:
        return f"gaussian_splat_{splat_format}"
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


def _compatibility(root_relpath: str | None, root_path: Path | None = None) -> CompatibilityStatus:
    if root_relpath is None:
        return "needs_root_stage"
    suffix = Path(root_relpath).suffix.lower()
    if suffix in USD_STAGE_EXTENSIONS:
        return "ready_to_render"
    if _is_robot_description_relpath(root_relpath):
        return "needs_conversion"
    if root_path is not None and detect_gaussian_splat_format(root_path) is not None:
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
    splat_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_info = {"type": "local", "name": source.name}
    if source_url:
        source_info["url"] = source_url
    manifest: dict[str, Any] = {
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
    if splat_info is not None:
        manifest["splat"] = splat_info
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
