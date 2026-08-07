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
    "CertificationIdentity",
    "CertificationStatus",
    "CloseProcessingResult",
    "RegisteredHoldout",
    "RegistrationError",
    "StrategyCloseState",
    "append_sealed_entry",
    "generate_shadow_decision",
    "prepare_registration_challenge",
    "process_market_close",
    "reconstruct_certification_status",
    "register_forward_holdout",
    "verify_certification_authorization",
    "verify_holdout_registration",
    "verify_sealed_ledger",
    "verify_shadow_decision",
    "write_shadow_decision",
]
