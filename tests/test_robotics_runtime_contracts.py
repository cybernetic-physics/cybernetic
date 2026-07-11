from __future__ import annotations

import builtins
import importlib
import math
import sys
from typing import Any, Mapping

import pytest

from cybernetics.robotics import (
    ActionChunk,
    AssetBundleRef,
    AssetMountSpec,
    EnvironmentPackageSpec,
    RobotContractError,
    RoboticsJobSpec,
)


def action_spec_dict() -> dict[str, Any]:
    return {
        "representation": "discrete",
        "tensor": {
            "name": "navigation_action",
            "semantic": "native",
            "dtype": "int64",
            "shape": [1],
            "bounds": [0, 3],
            "rate_hz": 2.0,
        },
        "control_hz": 2.0,
        "horizon": 1,
    }


def environment_dict(*, vectorized: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "robot-environment-package/v1",
        "package_id": "internnav-habitat-r2r",
        "simulator": "habitat-lab",
        "simulator_version": "0.3.3",
        "source_repo": "https://github.com/cybernetic-physics/InternNav.git",
        "source_ref": "7a5c62400ac45b313d9b709c740b64191556a242",
        "runtime_image": "ghcr.io/cybernetic-physics/internnav@sha256:" + "a" * 64,
        "factory": {
            "kind": "python",
            "target": "cyberphys_eval.adapters.internnav:create_habitat_env",
            "kwargs": {"config": "/runtime/config.py"},
        },
        "resources": {
            "cpu_cores": 4,
            "memory_gb": 16,
            "disk_gb": 40,
            "gpu_count": 0,
            "timeout_seconds": 3600,
        },
        "asset_mounts": [
            {
                "mount_path": "/runtime/assets/r2r",
                "read_only": True,
                "ref": {
                    "schema_version": "asset-bundle-ref/v1",
                    "uri": "cybernetics://artifacts/r2r",
                    "content_sha256": "b" * 64,
                    "media_type": "application/zip",
                    "size_bytes": 1024,
                    "metadata": {"unpack": True, "archive_format": "zip"},
                },
            }
        ],
        "observation_schema": {
            "rgb": {
                "name": "rgb",
                "semantic": "rgb",
                "dtype": "uint8",
                "shape": [480, 640, 3],
            },
            "instruction": {
                "name": "instruction",
                "semantic": "instruction",
                "dtype": "utf8",
                "shape": [1],
            },
        },
        "action_spec": action_spec_dict(),
        "supports_vectorization": vectorized,
        "default_vector_width": 2 if vectorized else 1,
        "metadata": {"benchmark": "r2r"},
    }


def job_dict(*, vectorized: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "robotics-job/v1",
        "job_name": "internnav-r2r-smoke",
        "environment": environment_dict(vectorized=vectorized),
        "policy": {
            "policy_id": "fixture-nav",
            "source": "fixture",
            "revision": "fixture-v1",
            "action_spec": action_spec_dict(),
            "config": {"action": 1},
        },
        "episodes": {
            "count": 2,
            "root_seed": 41,
            "vector_width": 2 if vectorized else 1,
            "max_steps": 8,
        },
        "recording": {
            "video": True,
            "cameras": ["rgb"],
            "dataset_export": "jsonl",
        },
        "evaluation": {
            "behavior": "vision_language_navigation",
            "primary_metric": "success",
            "checks": {
                "navigation_success": {
                    "metric": "success",
                    "operator": ">=",
                    "value": 1.0,
                }
            },
            "native_metrics": ["success", "spl", "distance_to_goal"],
        },
        "metadata": {"robot_id": "habitat-agent"},
    }


def test_robotics_job_round_trips_and_resolves_seed_tree() -> None:
    job = RoboticsJobSpec.from_dict(job_dict())

    assert RoboticsJobSpec.from_dict(job.to_dict()) == job
    assert job.episodes.resolved_seeds() == [41, 42]
    assert len(job.job_hash()) == 64
    assert job.environment.default_vector_width == 2
    assert "gpu_type" not in job.to_dict()["resources"]
    assert "normalization_id" not in job.to_dict()["policy"]["action_spec"]
    assert "units" not in job.to_dict()["environment"]["action_spec"]["tensor"]


def test_environment_package_requires_pinned_image_digest() -> None:
    data = environment_dict()
    data["runtime_image"] = "ghcr.io/cybernetic-physics/internnav:latest"

    with pytest.raises(RobotContractError, match="OCI sha256"):
        EnvironmentPackageSpec.from_dict(data)


def test_non_vector_environment_rejects_vector_job() -> None:
    data = job_dict(vectorized=False)
    data["episodes"]["vector_width"] = 2

    with pytest.raises(RobotContractError, match="non-vector"):
        RoboticsJobSpec.from_dict(data)


def test_asset_bundle_and_mount_reject_path_traversal() -> None:
    bundle = AssetBundleRef.from_dict(
        {
            "schema_version": "asset-bundle-ref/v1",
            "uri": "cybernetics://artifacts/hm3d",
            "content_sha256": "c" * 64,
            "media_type": "application/zip",
            "size_bytes": 1024,
        }
    )
    assert bundle.to_dict()["content_sha256"] == "c" * 64

    with pytest.raises(RobotContractError, match="canonical child"):
        AssetMountSpec.from_dict({"mount_path": "/runtime/../escape", "ref": bundle.to_dict()})

    with pytest.raises(RobotContractError, match="read_only"):
        AssetMountSpec.from_dict(
            {
                "mount_path": "/runtime/assets/data",
                "read_only": False,
                "ref": bundle.to_dict(),
            }
        )


def test_environment_package_rejects_asset_mounts_outside_reserved_root_or_overlapping() -> None:
    data = environment_dict()
    ref = data["asset_mounts"][0]["ref"]
    data["asset_mounts"][0]["mount_path"] = "/usr/local/bin/cyberphys-eval"
    with pytest.raises(RobotContractError, match="canonical child"):
        EnvironmentPackageSpec.from_dict(data)

    data = environment_dict()
    data["asset_mounts"].append(
        {"mount_path": "/runtime/assets/r2r/scenes", "read_only": True, "ref": ref}
    )
    with pytest.raises(RobotContractError, match="duplicate or overlapping"):
        EnvironmentPackageSpec.from_dict(data)


def test_action_chunk_rejects_more_actions_than_requested() -> None:
    with pytest.raises(RobotContractError, match="cannot exceed"):
        ActionChunk.from_dict(
            {
                "values": [[1], [2]],
                "representation": "discrete",
                "requested_horizon": 1,
                "produced_horizon": 2,
            }
        )


def test_action_chunk_preserves_timing_mask_and_auxiliary_metadata() -> None:
    data = {
        "values": [[1]],
        "representation": "discrete",
        "requested_horizon": 1,
        "produced_horizon": 1,
        "timestamps": [1.25],
        "valid_mask": [True],
        "inference_latency_ms": 0.0,
        "auxiliary": {"processor": "pinned"},
    }

    assert ActionChunk.from_dict(data).to_dict() == data


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("episodes", "count"), 1.5, "integer"),
        (("episodes", "max_steps"), math.inf, "finite"),
        (("environment", "supports_vectorization"), "false", "boolean"),
        (("recording", "video"), "false", "boolean"),
        (("evaluation", "checks", "navigation_success", "required"), "true", "boolean"),
    ],
)
def test_robotics_job_rejects_coerced_scalar_values(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    data = job_dict()
    cursor: dict[str, Any] = data
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value

    with pytest.raises(RobotContractError, match=message):
        RoboticsJobSpec.from_dict(data)


def test_tensor_specs_require_dimensions_and_ordered_finite_bounds() -> None:
    empty_shape = job_dict()
    empty_shape["environment"]["action_spec"]["tensor"]["shape"] = []
    with pytest.raises(RobotContractError, match="at least one dimension"):
        RoboticsJobSpec.from_dict(empty_shape)

    inverted_bounds = job_dict()
    inverted_bounds["environment"]["action_spec"]["tensor"]["bounds"] = [3, 0]
    with pytest.raises(RobotContractError, match="minimum cannot exceed"):
        RoboticsJobSpec.from_dict(inverted_bounds)

    misspelled_frame = job_dict()
    misspelled_frame["environment"]["action_spec"]["tensor"]["coordinate_frame"] = "robot"
    with pytest.raises(RobotContractError, match="unknown fields"):
        RoboticsJobSpec.from_dict(misspelled_frame)


def test_evaluation_requires_at_least_one_check() -> None:
    data = job_dict()
    data["evaluation"]["checks"] = {}

    with pytest.raises(RobotContractError, match="at least one check"):
        RoboticsJobSpec.from_dict(data)


def test_robotics_import_does_not_import_simulator_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = {"gymnasium", "habitat", "internnav", "isaaclab", "lerobot", "mujoco", "omni"}
    attempted: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and name.split(".", 1)[0] in blocked:
            attempted.append(name)
            raise AssertionError(f"simulator import attempted: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module in (
        "cybernetics.robotics.gymnasium",
        "cybernetics.robotics.runtime_contracts",
    ):
        sys.modules.pop(module, None)
        importlib.import_module(module)

    assert attempted == []
