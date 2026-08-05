"""Spectrum study arm specifications (options / equities / overlay)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

ALLOWED_ARM_IDS = frozenset({"opt", "eq", "mix"})
ALLOWED_DELTA_MODES = frozenset({"soft", "joint", "option_block", "off"})
# Spectrum train worlds (evidence-gated); see src/spectrum/registry.py.
ALLOWED_TRAIN_DISTRIBUTIONS = frozenset(
    {
        "historical",
        "rbergomi",
        "gbm",
        "heston",
        "garch",
        "sabr",
        "hybrid_pretrain_finetune",
    }
)
ALLOWED_RESIDUAL_MODELS = frozenset({"ff4", "ipca3", "none"})
ALLOWED_SYNTHETIC_PRETRAIN = frozenset({"off", "ablation"})

OPTION_LABEL_STEM = "dh_ret_lagdelta"
EQUITY_LABEL_STEM = "stk_ret"

SyntheticPretrain = Literal["off", "ablation"]
TrainDistribution = Literal[
    "historical",
    "rbergomi",
    "gbm",
    "heston",
    "garch",
    "sabr",
    "hybrid_pretrain_finetune",
]
ResidualModel = Literal["ff4", "ipca3", "none"]

@dataclass(frozen=True)
class ArmSpec:
    """Tradable-object arm for the spectrum study.

    Holds the information set constant and varies which instruments may be held.
    Slot layout is always ``[option_0 .. option_{n-1}, equity_0 .. equity_{m-1}]``.
    """

    id: str
    option_slots: int
    equity_slots: int
    delta_mode: str = "soft"
    option_label_stem: str = OPTION_LABEL_STEM
    equity_label_stem: str = EQUITY_LABEL_STEM
    train_distribution: str = "rbergomi"
    residual_model: str = "none"
    friction_spec_id: str = "legacy"
    synthetic_pretrain: str = "off"
    feature_allowlist_hash: str = ""
    # Spectrum overlay may train; only opt/eq may stamp residual-alpha claims.
    alpha_claim: bool = True

    def __post_init__(self) -> None:
        if self.id not in ALLOWED_ARM_IDS:
            raise ValueError(f"unknown arm id={self.id!r}; allowed={sorted(ALLOWED_ARM_IDS)}")
        if self.delta_mode not in ALLOWED_DELTA_MODES:
            raise ValueError(
                f"unknown delta_mode={self.delta_mode!r}; allowed={sorted(ALLOWED_DELTA_MODES)}"
            )
        if self.train_distribution not in ALLOWED_TRAIN_DISTRIBUTIONS:
            raise ValueError(
                f"unknown train_distribution={self.train_distribution!r}; "
                f"allowed={sorted(ALLOWED_TRAIN_DISTRIBUTIONS)}"
            )
        if self.residual_model not in ALLOWED_RESIDUAL_MODELS:
            raise ValueError(
                f"unknown residual_model={self.residual_model!r}; "
                f"allowed={sorted(ALLOWED_RESIDUAL_MODELS)}"
            )
        if self.synthetic_pretrain not in ALLOWED_SYNTHETIC_PRETRAIN:
            raise ValueError(
                f"unknown synthetic_pretrain={self.synthetic_pretrain!r}; "
                f"allowed={sorted(ALLOWED_SYNTHETIC_PRETRAIN)}"
            )
        if self.option_slots < 0 or self.equity_slots < 0:
            raise ValueError("slot counts must be non-negative")
        if self.id == "opt" and self.equity_slots != 0:
            raise ValueError("arm opt requires equity_slots=0")
        if self.id == "eq" and self.option_slots != 0:
            raise ValueError("arm eq requires option_slots=0")
        if self.id == "mix" and (self.option_slots <= 0 or self.equity_slots <= 0):
            raise ValueError("arm mix requires both option_slots>0 and equity_slots>0")
        if self.id == "mix" and self.alpha_claim:
            # Frozen dataclass: coerce rather than require every call site to pass False.
            object.__setattr__(self, "alpha_claim", False)
        if self.n_slots <= 0:
            raise ValueError("arm must have at least one slot")

    @property
    def n_slots(self) -> int:
        return int(self.option_slots) + int(self.equity_slots)

    def option_index(self) -> np.ndarray:
        return np.arange(self.option_slots, dtype=np.int64)

    def equity_index(self) -> np.ndarray:
        return np.arange(
            self.option_slots,
            self.option_slots + self.equity_slots,
            dtype=np.int64,
        )

    def delta_vector(self, option_deltas: np.ndarray | None = None) -> np.ndarray:
        """Build the delta vector used by the overlay projection.

        Option slots take published option deltas; equity slots take 1.0.
        """
        out = np.ones(self.n_slots, dtype=np.float64)
        if self.option_slots:
            if option_deltas is None:
                raise ValueError("option_deltas required when option_slots>0")
            d = np.asarray(option_deltas, dtype=np.float64).reshape(-1)
            if d.size != self.option_slots:
                raise ValueError(
                    f"option_deltas size {d.size} != option_slots {self.option_slots}"
                )
            out[self.option_index()] = d
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "option_slots": self.option_slots,
            "equity_slots": self.equity_slots,
            "delta_mode": self.delta_mode,
            "option_label_stem": self.option_label_stem,
            "equity_label_stem": self.equity_label_stem,
            "n_slots": self.n_slots,
            "train_distribution": self.train_distribution,
            "residual_model": self.residual_model,
            "friction_spec_id": self.friction_spec_id,
            "synthetic_pretrain": self.synthetic_pretrain,
            "feature_allowlist_hash": self.feature_allowlist_hash,
            "alpha_claim": bool(self.alpha_claim),
        }


def fingerprint_arm_spec(spec: ArmSpec) -> str:
    """Stable sha256 of canonical ArmSpec fields (for manifests / DSR)."""
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_explicit_arm(cfg: Mapping[str, Any] | None) -> ArmSpec:
    """Require an explicit ``arm:`` block (v2 acceptance path)."""
    cfg = dict(cfg or {})
    if not cfg.get("arm"):
        raise ValueError("explicit arm required: config missing arm: block")
    return arm_spec_from_cfg(cfg)


def default_arm_spec(n_assets: int = 50) -> ArmSpec:
    """Status-quo options-only arm (matches pre-spectrum overnight.yaml)."""
    return ArmSpec(
        id="opt",
        option_slots=int(n_assets),
        equity_slots=0,
        delta_mode="soft",
    )


def arm_spec_from_cfg(cfg: Mapping[str, Any] | None) -> ArmSpec:
    """Resolve ``arm:`` block from YAML; absent block uses ``portfolio_arm``.

    B-ARM: ``portfolio_arm: eq|mix`` must not silently fall through to the
    options-only status-quo default. Prefer an explicit ``arm:`` block when
    present; otherwise resolve via ``portfolio_arm`` (default ``opt``).
    """
    cfg = dict(cfg or {})
    arm = cfg.get("arm")
    n_assets = int(cfg.get("n_assets") or 50)
    if not arm:
        from src.arms.training import resolve_portfolio_arm

        return resolve_portfolio_arm(cfg)
    arm = dict(arm)
    # Explicit fail-closed only when requested. Spectrum mix stays loadable;
    # residual-alpha promotion still refuses mix via alpha_claim=False.
    if bool(arm.get("fail_on_load")):
        raise ValueError(
            f"arm {arm.get('id')!r} disabled (fail_on_load=true); "
            "refuse load"
        )
    if arm.get("enabled") is False and bool(cfg.get("residual_equity_protocol") or cfg.get("alpha_v2")):
        raise ValueError(
            f"arm {arm.get('id')!r} enabled=false under residual_equity_protocol; refuse load"
        )
    arm_id = str(arm.get("id") or "opt")
    if arm_id == "opt":
        option_slots = int(arm.get("option_slots", n_assets))
        equity_slots = int(arm.get("equity_slots", 0))
        delta_mode = str(arm.get("delta_mode", "soft"))
        residual_default = "ipca3" if arm.get("residual_model") else "none"
        alpha_claim_default = True
    elif arm_id == "eq":
        option_slots = int(arm.get("option_slots", 0))
        equity_slots = int(arm.get("equity_slots", n_assets))
        delta_mode = str(arm.get("delta_mode", "off"))
        residual_default = "ff4" if arm.get("residual_model") else "none"
        alpha_claim_default = True
    elif arm_id == "mix":
        option_slots = int(arm.get("option_slots", n_assets))
        equity_slots = int(arm.get("equity_slots", n_assets))
        delta_mode = str(arm.get("delta_mode", "joint"))
        residual_default = "none"
        alpha_claim_default = False
    else:
        raise ValueError(f"unknown arm id={arm_id!r}")
    alpha_claim = arm.get("alpha_claim", alpha_claim_default)
    if isinstance(alpha_claim, str):
        alpha_claim = alpha_claim.strip().lower() in ("1", "true", "yes", "on")
    return ArmSpec(
        id=arm_id,
        option_slots=option_slots,
        equity_slots=equity_slots,
        delta_mode=delta_mode,
        option_label_stem=str(arm.get("option_label_stem", OPTION_LABEL_STEM)),
        equity_label_stem=str(arm.get("equity_label_stem", EQUITY_LABEL_STEM)),
        train_distribution=str(arm.get("train_distribution", cfg.get("train_distribution", "rbergomi"))),
        residual_model=str(arm.get("residual_model", residual_default)),
        friction_spec_id=str(arm.get("friction_spec_id", cfg.get("friction_spec_id", "legacy"))),
        synthetic_pretrain=str(
            arm.get("synthetic_pretrain", cfg.get("synthetic_pretrain", "off"))
        ),
        feature_allowlist_hash=str(
            arm.get("feature_allowlist_hash", cfg.get("feature_allowlist_hash", ""))
        ),
        alpha_claim=bool(alpha_claim),
    )
