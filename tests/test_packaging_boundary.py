"""Secret-boundary gate: the published distribution ships the client SDK only.

This is the required pre-publish proof that no hosted-backend, infrastructure, or
secret artifact can leak into the ``cybernetic-physics`` wheel/sdist. It builds
both distributions from scratch and asserts:

  * the only importable package is ``cybernetics`` (no ``worldlines_backend``,
    no second top-level package), and
  * not a single member matches the denylist (backend tree, deploy/skypilot/vast
    manifests, Dockerfiles, env files, TF vars, private keys, or the internal
    ``wld-`` key prefix).

The member checks are deliberately written to be exact: top-level packages come
from ``parts[0]`` (not a fragile ``split('/')[1]`` that crashes on root files),
and ``wld-`` is matched as the literal key-prefix token on the file *basename*
(not an over-broad regex that would also flag ``worldlines``).
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import PurePosixPath

import pytest

PROJECT_ROOT = PurePosixPath(__file__).parent.parent
IMPORT_PACKAGE = "cybernetics"

# Substring tokens (matched anywhere in the POSIX member path, case-insensitive).
DENY_SUBSTRINGS = ("worldlines_backend", "skypilot")
# Path-prefix tokens (matched as a leading path component).
DENY_PREFIX_DIRS = ("deploy",)
# Basename predicates: (label, predicate over the lowercased basename).
DENY_BASENAME_RULES: tuple[tuple[str, "callable[[str], bool]"], ...] = (
    ("*vast*.yaml", lambda b: "vast" in b and b.endswith((".yaml", ".yml"))),
    ("sky-config*", lambda b: b.startswith("sky-config")),
    (".skyignore", lambda b: b == ".skyignore"),
    (
        "Dockerfile",
        lambda b: b == "dockerfile" or b.startswith("dockerfile.") or b.endswith(".dockerfile"),
    ),
    (".env*", lambda b: b == ".env" or b.startswith(".env.")),
    ("*.tfvars", lambda b: b.endswith(".tfvars")),
    ("*.pem", lambda b: b.endswith(".pem")),
    ("*.key", lambda b: b.endswith(".key")),
    # The internal key prefix is ``wld-`` (e.g. wld-abc...); match it as a literal
    # token in the basename. NOT a broad ``wld`` regex (which would flag worldlines).
    ("wld- key prefix", lambda b: "wld-" in b),
)


def _denylist_violations(members: list[str]) -> list[str]:
    violations: list[str] = []
    for name in members:
        path = PurePosixPath(name)
        low = name.lower()
        base = path.name.lower()
        for token in DENY_SUBSTRINGS:
            if token in low:
                violations.append(f"{name} (matched substring '{token}')")
        for top in DENY_PREFIX_DIRS:
            if path.parts and path.parts[0].lower() == top:
                violations.append(f"{name} (matched leading dir '{top}/')")
        for label, predicate in DENY_BASENAME_RULES:
            if predicate(base):
                violations.append(f"{name} (matched '{label}')")
    return violations


def _import_packages(members: list[str]) -> set[str]:
    """Top-level importable packages shipped by the distribution.

    A package is a top-level directory containing modules; a top-level ``.py``
    file is a single-module package. Metadata dirs and root non-module files
    (README, license, etc.) are not import packages.
    """

    tops: set[str] = set()
    for name in members:
        parts = PurePosixPath(name).parts
        if not parts:
            continue
        top = parts[0]
        if top.endswith((".dist-info", ".data")) or top == "":
            continue
        if len(parts) == 1:
            if top.endswith(".py"):
                tops.add(top[:-3])
            continue
        tops.add(top)
    return tops


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    out = tmp_path_factory.mktemp("dist")
    # --no-isolation: build deps (hatchling, hatch-fancy-pypi-readme) are in the dev
    # group, so the build is hermetic and needs no network.
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out), str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"build failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    assert len(sdists) == 1, f"expected exactly one sdist, got {sdists}"

    with zipfile.ZipFile(wheels[0]) as zf:
        wheel_members = zf.namelist()
    with tarfile.open(sdists[0]) as tf:
        # strip the top-level "<name>-<version>/" sdist prefix for member checks
        sdist_members = [m.split("/", 1)[1] if "/" in m else m for m in tf.getnames()]

    return {"wheel": wheel_members, "sdist": sdist_members}


def test_only_cybernetics_import_package_ships(built_distributions: dict[str, list[str]]) -> None:
    assert _import_packages(built_distributions["wheel"]) == {IMPORT_PACKAGE}


def test_wheel_has_no_denylisted_members(built_distributions: dict[str, list[str]]) -> None:
    violations = _denylist_violations(built_distributions["wheel"])
    assert not violations, "denylisted members leaked into the wheel:\n" + "\n".join(violations)


def test_sdist_has_no_denylisted_members(built_distributions: dict[str, list[str]]) -> None:
    violations = _denylist_violations(built_distributions["sdist"])
    assert not violations, "denylisted members leaked into the sdist:\n" + "\n".join(violations)


def test_denylist_helpers_are_not_overbroad() -> None:
    """Guard against the historical bugs: false positives + crash on root files."""

    # 'worldlines' (the kept URI scheme / *_worldlines_path names) must NOT trip wld-.
    assert not _denylist_violations(["cybernetics/cli/commands/checkpoint.py"])
    assert not _denylist_violations(
        ["cybernetics/types/checkpoint.py"]
    )  # contains 'worldlines://' literals
    # A real backend/secret member IS caught.
    assert _denylist_violations(["worldlines_backend/app.py"])
    assert _denylist_violations(["deploy/skypilot/train.sky.yaml"])
    assert _denylist_violations(["secrets/prod.tfvars"])
    assert _denylist_violations(["keys/wld-abc123.key"])
    # Root-level file (no '/') must not crash _import_packages.
    assert _import_packages(["cybernetics/__init__.py", "README.md"]) == {IMPORT_PACKAGE}
