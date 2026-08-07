"""Build a create-only Phase 3B certification OCI archive from an exact merge SHA."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.governance import GovernanceError, read_yaml, resolve_path
from portfolio_rl.phase3b.runtime_identity import build_certification_image


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/phase3b/container_build.yaml")
    parser.add_argument("--runtime-git-sha", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    config = read_yaml(resolve_path(root, args.config))
    if (
        set(config)
        != {
            "schema_version",
            "status",
            "base_image",
            "python_version",
            "platform",
            "image_name",
            "output_oci_archive",
        "output_identity",
        "output_runtime_identity",
            "push",
        }
        or config["schema_version"] != 1
    ):
        raise GovernanceError("container build config schema mismatch")
    if config["status"] != "ready_for_build" or config["push"] is not False:
        raise GovernanceError(
            "certification image build must be approved locally with push disabled"
        )
    if str(config["python_version"]) != "3.12.12":
        raise GovernanceError("certification Python version mismatch")
    identity = build_certification_image(
        repository_root=root,
        runtime_git_sha=args.runtime_git_sha,
        base_image=str(config["base_image"]),
        platform=str(config["platform"]),
        output_oci_archive=resolve_path(root, config["output_oci_archive"]),
        output_identity=resolve_path(root, config["output_identity"]),
        output_runtime_identity=resolve_path(root, config["output_runtime_identity"]),
        image_name=str(config["image_name"]),
        built_at=datetime.now(UTC),
    )
    print(f"container_identity: {resolve_path(root, config['output_identity'])}")
    print(
        "runtime_identity: "
        f"{resolve_path(root, config['output_runtime_identity'])}"
    )
    print(f"oci_image_digest: {identity['image_digest']}")
    print("image_pushed: false")
    print("official_certification_started: false")


if __name__ == "__main__":
    main()
