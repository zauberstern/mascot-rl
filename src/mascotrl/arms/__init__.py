"""Spectrum study arms package."""

from mascotrl.arms.spec import (
    ALLOWED_ARM_IDS,
    ALLOWED_DELTA_MODES,
    ALLOWED_RESIDUAL_MODELS,
    ALLOWED_SYNTHETIC_PRETRAIN,
    ALLOWED_TRAIN_DISTRIBUTIONS,
    EQUITY_LABEL_STEM,
    OPTION_LABEL_STEM,
    ArmSpec,
    arm_spec_from_cfg,
    default_arm_spec,
    fingerprint_arm_spec,
    require_explicit_arm,
)

__all__ = [
    "ALLOWED_ARM_IDS",
    "ALLOWED_DELTA_MODES",
    "ALLOWED_RESIDUAL_MODELS",
    "ALLOWED_SYNTHETIC_PRETRAIN",
    "ALLOWED_TRAIN_DISTRIBUTIONS",
    "EQUITY_LABEL_STEM",
    "OPTION_LABEL_STEM",
    "ArmSpec",
    "arm_spec_from_cfg",
    "default_arm_spec",
    "fingerprint_arm_spec",
    "require_explicit_arm",
]
