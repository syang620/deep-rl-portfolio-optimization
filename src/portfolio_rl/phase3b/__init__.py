"""Governance interfaces for the independent Phase 3B forward holdout."""

from portfolio_rl.phase3b.certification import (
    CertificationIdentity,
    CertificationStatus,
    reconstruct_certification_status,
    verify_certification_authorization,
)
from portfolio_rl.phase3b.close_processor import (
    CloseProcessingResult,
    StrategyCloseState,
    process_market_close,
)
from portfolio_rl.phase3b.holdout_registry import (
    RegisteredHoldout,
    RegistrationError,
    prepare_registration_challenge,
    register_forward_holdout,
    verify_holdout_registration,
)
from portfolio_rl.phase3b.identity_approval import (
    ApprovedRuntimeIdentity,
    finalize_identity_approval,
    prepare_identity_approval,
    sign_identity_approval,
    verify_identity_approval,
)
from portfolio_rl.phase3b.sealed_ledger import (
    append_sealed_entry,
    verify_sealed_ledger,
)
from portfolio_rl.phase3b.shadow_runner import (
    generate_shadow_decision,
    verify_shadow_decision,
    write_shadow_decision,
)

__all__ = [
    "ApprovedRuntimeIdentity",
    "CertificationIdentity",
    "CertificationStatus",
    "CloseProcessingResult",
    "RegisteredHoldout",
    "RegistrationError",
    "StrategyCloseState",
    "append_sealed_entry",
    "finalize_identity_approval",
    "generate_shadow_decision",
    "prepare_identity_approval",
    "prepare_registration_challenge",
    "process_market_close",
    "reconstruct_certification_status",
    "register_forward_holdout",
    "sign_identity_approval",
    "verify_certification_authorization",
    "verify_holdout_registration",
    "verify_identity_approval",
    "verify_sealed_ledger",
    "verify_shadow_decision",
    "write_shadow_decision",
]
