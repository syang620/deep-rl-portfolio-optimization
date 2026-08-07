"""Certification-container build metadata and runtime identity verification."""

from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.governance import (
    GIT_COMMIT_PATTERN,
    OCI_DIGEST_PATTERN,
    GovernanceError,
    read_json,
    sha256_file,
    validate_container_identity,
)

RUNTIME_IDENTITY_PATH = Path("/opt/portfolio-rl/runtime_identity.json")
PR22_MERGE_SHA = "f53f64afeff1638302977b1f7b30979f488fcd43"


def build_certification_image(
    *,
    repository_root: Path,
    runtime_git_sha: str,
    base_image: str,
    platform: str,
    output_oci_archive: Path,
    output_identity: Path,
    output_runtime_identity: Path,
    image_name: str,
    built_at: datetime,
) -> dict[str, Any]:
    """Build an OCI archive from one exact Git tree without pushing it."""
    root = repository_root.resolve()
    if built_at.tzinfo is None:
        raise GovernanceError("container build timestamp must be timezone-aware")
    if not GIT_COMMIT_PATTERN.fullmatch(runtime_git_sha):
        raise GovernanceError("runtime Git SHA must be a full commit identifier")
    _git(root, "cat-file", "-e", f"{runtime_git_sha}^{{commit}}")
    if _git_text(root, "rev-parse", "origin/main") != runtime_git_sha:
        raise GovernanceError(
            "runtime Git SHA must equal the merged origin/main commit"
        )
    if "@sha256:" not in base_image:
        raise GovernanceError("base image must be pinned by OCI digest")
    base_digest = "sha256:" + base_image.rsplit("@sha256:", 1)[1]
    if not OCI_DIGEST_PATTERN.fullmatch(base_digest):
        raise GovernanceError("base image digest is invalid")
    if any(
        path.exists()
        for path in (output_oci_archive, output_identity, output_runtime_identity)
    ):
        raise FileExistsError("container build outputs are create-only")
    output_oci_archive.parent.mkdir(parents=True, exist_ok=True)
    output_identity.parent.mkdir(parents=True, exist_ok=True)
    output_runtime_identity.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase3b-container-") as temp_name:
        context = Path(temp_name) / "context"
        context.mkdir()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", runtime_git_sha],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        source_tar = Path(temp_name) / "source.tar"
        source_tar.write_bytes(archive)
        with tarfile.open(source_tar) as tar:
            for member in tar.getmembers():
                destination = (context / member.name).resolve()
                try:
                    destination.relative_to(context)
                except ValueError as exc:
                    raise GovernanceError("Git archive contains an unsafe path") from exc
            tar.extractall(context)
        runtime_payload = {
            "schema_version": 1,
            "runtime_git_sha": runtime_git_sha,
            "pr22_merge_sha": PR22_MERGE_SHA,
            "pyproject_sha256": sha256_file(context / "pyproject.toml"),
            "dependency_lock_sha256": sha256_file(
                context / "requirements-phase3b.lock"
            ),
        }
        runtime_file = context / "runtime_identity.json"
        runtime_file.write_text(
            json.dumps(runtime_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        runtime_file_sha256 = sha256_file(runtime_file)
        runtime_file_bytes = runtime_file.read_bytes()
        command = [
            "docker",
            "buildx",
            "build",
            "--platform",
            platform,
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"RUNTIME_GIT_SHA={runtime_git_sha}",
            "--build-arg",
            f"PR22_MERGE_SHA={PR22_MERGE_SHA}",
            "--build-arg",
            f"PYPROJECT_SHA256={runtime_payload['pyproject_sha256']}",
            "--build-arg",
            f"DEPENDENCY_LOCK_SHA256={runtime_payload['dependency_lock_sha256']}",
            "--output",
            f"type=oci,dest={output_oci_archive.resolve()}",
            "-f",
            str(context / "Dockerfile.phase3b"),
            str(context),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise GovernanceError(
                f"certification image build failed: {result.stderr.strip()}"
            )
    image_digest = _oci_manifest_digest(output_oci_archive)
    dependency_lock = _oci_annotation(
        output_oci_archive, "org.portfolio-rl.dependency-lock-sha256"
    )
    python_version = _oci_annotation(
        output_oci_archive, "org.portfolio-rl.python-version"
    )
    identity = {
        "schema_version": 2,
        "image_reference": f"{image_name}@{image_digest}",
        "image_digest": image_digest,
        "git_commit": runtime_git_sha,
        "input_schema_version": "phase3b_normalized_features_v1",
        "data_source_contract_version": "point-in-time-v1",
        "built_at": built_at.astimezone(UTC).isoformat(),
        "base_image_digest": base_digest,
        "dependency_lock_sha256": dependency_lock,
        "platform": platform,
        "pr22_merge_sha": PR22_MERGE_SHA,
        "pyproject_sha256": runtime_payload["pyproject_sha256"],
        "python_version": python_version,
        "runtime_identity_file_sha256": runtime_file_sha256,
    }
    validate_container_identity(identity, runtime_git_sha)
    output_runtime_identity.write_bytes(runtime_file_bytes)
    output_identity.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def verify_runtime_identity(
    *,
    repository_root: Path,
    embedded_identity_path: Path,
    container_identity_path: Path,
    identity_package_path: Path | None = None,
) -> str:
    """Require repository, embedded image, container, and approved package SHAs to match."""
    root = repository_root.resolve()
    repository_sha = _git_text(root, "rev-parse", "HEAD")
    embedded = read_json(embedded_identity_path)
    container = read_json(container_identity_path)
    embedded_sha = str(embedded.get("runtime_git_sha", ""))
    validate_container_identity(container, repository_sha)
    if container.get("schema_version") == 2:
        expected_embedded = {
            "schema_version",
            "runtime_git_sha",
            "pr22_merge_sha",
            "pyproject_sha256",
            "dependency_lock_sha256",
        }
        if set(embedded) != expected_embedded:
            raise GovernanceError("embedded runtime identity schema mismatch")
        if sha256_file(embedded_identity_path) != container["runtime_identity_file_sha256"]:
            raise GovernanceError("embedded runtime identity file hash mismatch")
        if (
            embedded["pr22_merge_sha"] != PR22_MERGE_SHA
            or container["pr22_merge_sha"] != PR22_MERGE_SHA
        ):
            raise GovernanceError("embedded PR 22 provenance mismatch")
        bindings = {
            "pyproject_sha256": sha256_file(root / "pyproject.toml"),
            "dependency_lock_sha256": sha256_file(
                root / "requirements-phase3b.lock"
            ),
        }
        for key, expected in bindings.items():
            if embedded[key] != expected or container[key] != expected:
                raise GovernanceError(f"runtime {key} mismatch")
    observed = {repository_sha, embedded_sha, str(container["git_commit"])}
    if identity_package_path is not None:
        finalized = read_json(identity_package_path / "finalized_identity.json")
        observed.add(
            str(finalized.get("provenance", {}).get("identity_tooling_merge_sha", ""))
        )
        if (
            finalized.get("provenance", {}).get("container_digest")
            != container["image_digest"]
        ):
            raise GovernanceError("identity package container digest mismatch")
    if len(observed) != 1:
        raise GovernanceError("repository, embedded, and approved runtime SHAs differ")
    return repository_sha


def _oci_manifest_digest(path: Path) -> str:
    with tarfile.open(path) as archive:
        index = json.load(archive.extractfile("index.json"))
    manifests = [
        row
        for row in index["manifests"]
        if row.get("platform", {}).get("os") != "unknown"
    ]
    if len(manifests) != 1 or not OCI_DIGEST_PATTERN.fullmatch(manifests[0]["digest"]):
        raise GovernanceError("OCI archive does not contain one authoritative manifest")
    return manifests[0]["digest"]


def _oci_annotation(path: Path, key: str) -> str:
    with tarfile.open(path) as archive:
        index = json.load(archive.extractfile("index.json"))
        descriptor = next(
            row
            for row in index["manifests"]
            if row.get("platform", {}).get("os") != "unknown"
        )
        manifest = json.load(
            archive.extractfile(f"blobs/sha256/{descriptor['digest'].split(':')[1]}")
        )
        config_digest = manifest["config"]["digest"].split(":", 1)[1]
        config = json.load(archive.extractfile(f"blobs/sha256/{config_digest}"))
    value = config.get("config", {}).get("Labels", {}).get(key)
    if not isinstance(value, str) or not value:
        raise GovernanceError(f"OCI image is missing required annotation: {key}")
    return value


def _git(root: Path, *args: str) -> None:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise GovernanceError(f"Git command failed: git {' '.join(args)}")


def _git_text(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise GovernanceError(f"Git command failed: git {' '.join(args)}")
    return result.stdout.strip()
