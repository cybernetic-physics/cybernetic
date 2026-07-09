from __future__ import annotations

import json
import zipfile

import pytest

from cybernetics.sim import AssetPackageError, inspect_local_asset, package_local_asset


def test_package_local_usd_folder_builds_bundle_with_manifest(tmp_path) -> None:
    scene = tmp_path / "scene.usd"
    scene.write_text("#usda 1.0\n")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "cube.usdc").write_bytes(b"PXR-USDC")

    package = package_local_asset(tmp_path, output_path=tmp_path / "bundle.zip")

    assert package.root_stage_relpath == "scene.usd"
    assert package.asset_kind == "usd_stage"
    assert package.compatibility_status == "ready_to_render"
    assert package.bundle_size_bytes > 0
    assert len(package.bundle_sha256) == 64

    with zipfile.ZipFile(package.bundle_path) as zf:
        assert {"scene.usd", "assets/cube.usdc", "cybernetics_sim_asset_manifest.json"} <= set(
            zf.namelist()
        )
        manifest = json.loads(zf.read("cybernetics_sim_asset_manifest.json"))
    assert manifest["root_stage_relpath"] == "scene.usd"
    assert manifest["compatibility_status"] == "ready_to_render"
    assert manifest["source"] == {"type": "local", "name": tmp_path.name}
    assert str(tmp_path) not in json.dumps(manifest)


def test_package_excludes_preexisting_output_bundle_inside_source(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"old bundle")

    package = package_local_asset(tmp_path, output_path=bundle_path)

    with zipfile.ZipFile(package.bundle_path) as zf:
        assert "scene.usd" in zf.namelist()
        assert "bundle.zip" not in zf.namelist()


def test_manifest_keeps_explicit_source_url_without_local_absolute_path(tmp_path) -> None:
    scene = tmp_path / "scene.usd"
    scene.write_text("#usda 1.0\n")

    package = package_local_asset(
        tmp_path,
        output_path=tmp_path / "bundle.zip",
        source_url="https://example.test/research/scene",
    )

    manifest_json = json.dumps(package.manifest)
    assert package.manifest["source"] == {
        "type": "local",
        "name": tmp_path.name,
        "url": "https://example.test/research/scene",
    }
    assert str(tmp_path) not in manifest_json


def test_robot_description_is_classified_as_conversion_needed(tmp_path) -> None:
    robot = tmp_path / "robot.urdf"
    robot.write_text("<robot name='demo'></robot>\n")

    inspection = inspect_local_asset(robot)

    assert inspection.root_relpath == "robot.urdf"
    assert inspection.asset_kind == "urdf_robot"
    assert inspection.compatibility_status == "needs_conversion"


def test_package_rejects_unsafe_root_stage(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    with pytest.raises(AssetPackageError, match="traversal"):
        package_local_asset(tmp_path, root_stage="../scene.usd")
