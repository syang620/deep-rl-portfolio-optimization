"""Governance interfaces for the independent Phase 3B forward holdout."""

from portfolio_rl.phase3b.holdout_registry import (
    RegisteredHoldout,
    RegistrationError,
    prepare_registration_challenge,
    register_forward_holdout,
    verify_holdout_registration,
)
from portfolio_rl.phase3b.shadow_runner import (
    generate_shadow_decision,
    verify_shadow_decision,
    write_shadow_decision,
)

__all__ = [
    "RegisteredHoldout",
    "RegistrationError",
    "generate_shadow_decision",
    "prepare_registration_challenge",
    "register_forward_holdout",
    "verify_holdout_registration",
    "verify_shadow_decision",
    "write_shadow_decision",
]
