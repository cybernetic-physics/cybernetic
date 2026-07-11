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
    RobotContractError,
    RoboticsJobSpec,
    SimulatorPackageSpec,
    TaskPackageSpec,
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


def observation_schema_dict() -> dict[str, Any]:
    return {
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
    }


def simulator_dict(*, vectorized: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "robot-simulator-package/v1",
        "package_id": "sim-habitat-0.3.3",
        "simulator": "habitat-lab",
        "simulator_version": "0.3.3",
        "source_repo": "https://github.com/cybernetic-physics/InternNav.git",
        "source_ref": "7a5c62400ac45b313d9b709c740b64191556a242",
        "runtime_image": "ghcr.io/cybernetic-physics/internnav@sha256:" + "a" * 64,
        "service_entrypoint": "cyberphys-sim-service",
        "factory": {
            "kind": "python",
            "target": "cyberphys_eval.adapters.internnav:create_habitat_env",
            "kwargs": {},
        },
        "resources": {
            "cpu_cores": 4,
            "memory_gb": 16,
            "disk_gb": 40,
            "gpu_count": 0,
            "timeout_seconds": 3600,
        },
        "supports_vectorization": vectorized,
        "default_vector_width": 2 if vectorized else 1,
        "capabilities": ["rgb", "reset", "snapshot", "vector_step"],
        "supported_asset_formats": ["glb", "json"],
        "mount_roots": ["/runtime/assets"],
        "license": "MIT",
        "metadata": {"service_protocol": "sim-service/v1"},
    }


def task_dict() -> dict[str, Any]:
    return {
        "schema_version": "robot-task-package/v1",
        "package_id": "task-internnav-r2r-val-unseen",
        "task_id": "internnav-r2r",
        "revision": "r2r-val-unseen-v1",
        "source_repo": "https://github.com/cybernetic-physics/InternNav.git",
        "source_ref": "7a5c62400ac45b313d9b709c740b64191556a242",
        "compatible_simulators": ["sim-habitat-0.3.3", "habitat-lab"],
        "required_capabilities": ["rgb", "reset"],
        "embodiment_id": "habitat-agent",
        "observation_schema": observation_schema_dict(),
        "action_spec": action_spec_dict(),
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
        "adapter_config": {"config_path": "/runtime/config.py"},
        "dataset": {"dataset_id": "r2r", "split": "val_unseen", "revision": "v1"},
        "native_metrics": ["success", "spl", "distance_to_goal"],
        "license": "R2R research license",
        "metadata": {"benchmark": "r2r"},
    }


def job_dict(*, vectorized: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "robotics-job/v1",
        "job_name": "internnav-r2r-smoke",
        "simulator": simulator_dict(vectorized=vectorized),
        "task": task_dict(),
        "policy": {
            "schema_version": "robot-policy-deployment/v1",
            "deployment_id": "wlp-fixture-nav",
            "model_id": "fixture-nav",
            "source": "fixture",
            "runtime_family": "fixture",
            "revision": "fixture-v1",
            "embodiment_id": "habitat-agent",
            "observation_schema": observation_schema_dict(),
            "action_spec": action_spec_dict(),
            "resources": {
                "cpu_cores": 1,
                "memory_gb": 1,
                "disk_gb": 1,
                "gpu_count": 0,
                "timeout_seconds": 3600,
            },
            "max_batch_size": 2,
            "max_horizon": 1,
            "state_model": "stateless",
            "reset_granularity": "environment",
            "deterministic": True,
            "default_action_selection": {
                "execution_horizon": 1,
                "queue_threshold": 0,
                "overlap": "latest",
            },
            "config": {"action": 1},
        },
        "rollout": {
            "episodes": 2,
            "root_seed": 41,
            "vector_width": 2 if vectorized else 1,
            "max_steps": 8,
            "control_rate_hz": 2,
            "action_selection": {
                "execution_horizon": 1,
                "queue_threshold": 0,
                "overlap": "latest",
            },
        },
        "placement": {
            "topology": "colocated_required",
            "simulator_resources": simulator_dict(vectorized=vectorized)["resources"],
            "policy_resources": {
                "cpu_cores": 1,
                "memory_gb": 1,
                "disk_gb": 1,
                "gpu_count": 0,
                "timeout_seconds": 3600,
            },
            "coordinator_resources": {
                "cpu_cores": 1,
                "memory_gb": 1,
                "disk_gb": 1,
                "gpu_count": 0,
                "timeout_seconds": 3600,
            },
            "gpu_sharing": False,
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
    assert job.rollout.resolved_seeds() == [41, 42]
    assert len(job.job_hash()) == 64
    assert job.simulator.default_vector_width == 2
    assert "gpu_type" not in job.runtime_resources().to_dict()
    assert "normalization_id" not in job.to_dict()["policy"]["action_spec"]
    assert "units" not in job.to_dict()["task"]["action_spec"]["tensor"]


def test_robotics_job_v1_has_cross_language_canonical_hash_and_composed_boundaries() -> None:
    job = RoboticsJobSpec.from_dict(job_dict())
    serialized = job.to_dict()

    assert job.job_hash() == "a76036151879cb85ffb5c97d3cb31f1a3aa0deda839d41e25611eff0b2137490"
    assert {"environment", "episodes", "resources"}.isdisjoint(serialized)
    assert {
        "asset_mounts",
        "observation_schema",
        "action_spec",
        "policy_id",
        "checkpoint_ref",
    }.isdisjoint(serialized["simulator"])
    assert serialized["task"]["license"] == "R2R research license"
    assert serialized["policy"]["deployment_id"] == "wlp-fixture-nav"


def test_worldlines_policy_endpoints_and_credentials_are_not_manifest_fields() -> None:
    data = job_dict()
    data["policy"]["source"] = "worldlines"
    data["policy"]["config"] = {"url": "https://policy.invalid", "token": "secret"}

    with pytest.raises(RobotContractError, match="control-plane resolved"):
        RoboticsJobSpec.from_dict(data)


def test_policy_default_action_selection_cannot_exceed_deployment_horizon() -> None:
    data = job_dict()
    data["policy"]["default_action_selection"]["execution_horizon"] = 2

    with pytest.raises(RobotContractError, match="exceeds max_horizon"):
        RoboticsJobSpec.from_dict(data)


def test_simulator_package_requires_pinned_image_digest() -> None:
    data = simulator_dict()
    data["runtime_image"] = "ghcr.io/cybernetic-physics/internnav:latest"

    with pytest.raises(RobotContractError, match="OCI sha256"):
        SimulatorPackageSpec.from_dict(data)


def test_non_vector_environment_rejects_vector_job() -> None:
    data = job_dict(vectorized=False)
    data["rollout"]["vector_width"] = 2

    with pytest.raises(RobotContractError, match="non-vector"):
        RoboticsJobSpec.from_dict(data)


def test_placement_must_preserve_a_package_required_gpu_type() -> None:
    data = job_dict()
    data["simulator"]["resources"].update({"gpu_count": 1, "gpu_type": "RTX_4090"})
    data["placement"]["simulator_resources"]["gpu_count"] = 1
    data["placement"]["simulator_resources"].pop("gpu_type", None)

    with pytest.raises(RobotContractError, match="gpu_type"):
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


def test_task_package_rejects_asset_mounts_outside_reserved_root_or_overlapping() -> None:
    data = task_dict()
    ref = data["asset_mounts"][0]["ref"]
    data["asset_mounts"][0]["mount_path"] = "/usr/local/bin/cyberphys-eval"
    with pytest.raises(RobotContractError, match="canonical child"):
        TaskPackageSpec.from_dict(data)

    data = task_dict()
    data["asset_mounts"].append(
        {"mount_path": "/runtime/assets/r2r/scenes", "read_only": True, "ref": ref}
    )
    with pytest.raises(RobotContractError, match="duplicate or overlapping"):
        TaskPackageSpec.from_dict(data)


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
        (("rollout", "episodes"), 1.5, "integer"),
        (("rollout", "max_steps"), math.inf, "finite"),
        (("simulator", "supports_vectorization"), "false", "boolean"),
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
    empty_shape["task"]["action_spec"]["tensor"]["shape"] = []
    with pytest.raises(RobotContractError, match="at least one dimension"):
        RoboticsJobSpec.from_dict(empty_shape)

    inverted_bounds = job_dict()
    inverted_bounds["task"]["action_spec"]["tensor"]["bounds"] = [3, 0]
    with pytest.raises(RobotContractError, match="minimum cannot exceed"):
        RoboticsJobSpec.from_dict(inverted_bounds)

    misspelled_frame = job_dict()
    misspelled_frame["task"]["action_spec"]["tensor"]["coordinate_frame"] = "robot"
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
