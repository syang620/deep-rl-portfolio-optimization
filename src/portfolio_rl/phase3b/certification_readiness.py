"""Fail-closed readiness checks that never start Phase 3B certification."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from portfolio_rl.phase3b.certification_authorization import (
    ApprovedCertificationAuthorization,
    verify_finalized_certification_authorization,
)
from portfolio_rl.phase3b.governance import GovernanceError, read_json, read_yaml
from portfolio_rl.phase3b.identity_approval import (
    _registry_state,
    _test_access_audit,
    verify_identity_approval,
)
from portfolio_rl.phase3b.public_identities import inspect_public_identities
from portfolio_rl.phase3b.runtime_identity import verify_runtime_identity


def check_certification_readiness(
    *,
    repository_root: Path,
    identity_package_path: Path,
    authorization_package_path: Path,
    container_identity_path: Path,
    embedded_identity_path: Path,
    certification_id: str,
    decision_date: date | None = None,
    cycle_number: int | None = None,
) -> list[str]:
    """Return every blocking reason; an empty list is the only ready state."""
    root = repository_root.resolve()
    blockers: list[str] = []
    try:
        approved = verify_identity_approval(
            repository_root=root,
            package_path=identity_package_path,
            require_current_evidence=False,
        )
    except (GovernanceError, FileNotFoundError) as exc:
        return [f"finalized identity package: {exc}"]
    try:
        authorization = verify_finalized_certification_authorization(
            repository_root=root,
            package_path=authorization_package_path,
            require_current_evidence=False,
        )
    except (GovernanceError, FileNotFoundError) as exc:
        return [f"finalized certification authorization: {exc}"]
    if authorization.identity_package_path != approved.package_path:
        blockers.append("authorization references a different identity package")
    if authorization.identity != approved.identity:
        blockers.append("authorization identity differs from finalized identity")
    if authorization.certification_id != certification_id:
        blockers.append("authorization certification ID mismatch")
    if decision_date is not None and not (
        authorization.approved_start_date
        <= decision_date
        <= authorization.approved_end_date
    ):
        blockers.append(
            "decision date is outside the approved certification start window"
        )
    try:
        verify_runtime_identity(
            repository_root=root,
            embedded_identity_path=embedded_identity_path,
            container_identity_path=container_identity_path,
            identity_package_path=approved.package_path,
        )
        packaged_container = read_json(
            approved.package_path / "candidate/container_identity.json"
        )
        if packaged_container != read_json(container_identity_path):
            blockers.append("runtime container identity differs from approved package")
    except (GovernanceError, FileNotFoundError) as exc:
        blockers.append(f"runtime identity: {exc}")
    try:
        reconciliation = read_json(
            approved.package_path / "evidence/scaler_reconciliation.json"
        )
        if (
            reconciliation.get("reconciled") is not True
            or reconciliation.get("refit_performed") is not False
        ):
            blockers.append("frozen scaler reconciliation is not valid")
        for key in (
            "maximum_asset_normalization_error",
            "maximum_global_normalization_error",
            "maximum_model_matrix_error",
        ):
            if float(reconciliation.get(key, float("inf"))) > 1e-12:
                blockers.append(f"frozen scaler reconciliation failed: {key}")
    except (GovernanceError, OSError, TypeError, ValueError) as exc:
        blockers.append(f"scaler reconciliation: {exc}")
    try:
        inspect_public_identities(
            service_signing_key=approved.package_path
            / "candidate/public_keys/service_signing.pub",
            performance_sealing_key=approved.package_path
            / "candidate/public_keys/performance_sealing.pub",
            approver_keys={
                role: approved.package_path / f"candidate/public_keys/{role}.pub"
                for role in (
                    "portfolio_manager",
                    "independent_reviewer",
                    "data_operations_custodian",
                )
            },
        )
    except GovernanceError as exc:
        blockers.append(f"public identities: {exc}")
    try:
        _test_access_audit(root)
    except GovernanceError as exc:
        blockers.append(f"test-access audit: {exc}")
    try:
        _registry_state(root, root / "artifacts/phase3b/registration", "holdout")
    except GovernanceError as exc:
        blockers.append(f"holdout registry: {exc}")
    conflicts, completed_cycles = _certification_conflicts(
        root, certification_id, approved.identity.identity_sha256
    )
    blockers.extend(conflicts)
    if cycle_number is not None and cycle_number != completed_cycles + 1:
        blockers.append(
            f"official cycle number must be {completed_cycles + 1} for current registry state"
        )
    for name in ("execution", "operations", "access_control"):
        try:
            if (
                read_yaml(root / f"configs/phase3b/{name}.yaml").get("status")
                != "draft"
            ):
                blockers.append(f"tracked {name} template is not draft")
        except GovernanceError as exc:
            blockers.append(f"tracked {name} template: {exc}")
    return blockers


def require_certification_readiness(
    **kwargs: object,
) -> ApprovedCertificationAuthorization:
    """Raise with all blockers, otherwise return the verified authorization."""
    blockers = check_certification_readiness(**kwargs)  # type: ignore[arg-type]
    if blockers:
        raise GovernanceError("certification readiness failed: " + "; ".join(blockers))
    return verify_finalized_certification_authorization(
        repository_root=Path(kwargs["repository_root"]),
        package_path=Path(kwargs["authorization_package_path"]),
        require_current_evidence=False,
    )


def _certification_conflicts(
    root: Path, certification_id: str, identity_sha: str
) -> tuple[list[str], int]:
    blockers: list[str] = []
    completed_cycles: set[int] = set()
    registry = root / "artifacts/phase3b/certification"
    if not registry.exists():
        return blockers, 0
    for path in sorted(registry.rglob("*.json")):
        try:
            payload = read_json(path)
        except GovernanceError as exc:
            blockers.append(
                f"unverifiable certification registry artifact {path.name}: {exc}"
            )
            continue
        if payload.get("official") is not True:
            continue
        if payload.get("certification_id") != certification_id:
            blockers.append(
                "official certification already started for another certification ID"
            )
        if payload.get("identity_sha256") != identity_sha:
            blockers.append(
                "official certification registry contains a conflicting identity"
            )
        cycle = payload.get("cycle_number")
        if payload.get("certification_id") == certification_id and isinstance(
            cycle, int
        ):
            completed_cycles.add(cycle)
    if completed_cycles and completed_cycles != set(
        range(1, max(completed_cycles) + 1)
    ):
        blockers.append("official certification registry has nonconsecutive cycles")
    return blockers, len(completed_cycles)
