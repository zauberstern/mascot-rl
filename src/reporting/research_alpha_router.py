"""Resolve research_alpha primary train path (hist env + friction parity)."""
from __future__ import annotations

from typing import Any, Mapping, overload

from src.eval.friction import FrictionSpec, friction_spec_from_cfg

RESEARCH_PRIMARY_HIST = "historical_arm_env"
# Spectrum-relaxed allowlist (hist remains the research default).
RESEARCH_PRIMARY_ALLOWED = frozenset(
    {
        "historical_arm_env",
        "hybrid_pretrain_finetune",
        "historical",
    }
)


@overload
def resolve_research_primary_train(
    cfg: Mapping[str, Any],
    *,
    with_meta: bool = False,
) -> str: ...


@overload
def resolve_research_primary_train(
    cfg: Mapping[str, Any],
    *,
    with_meta: bool,
) -> str | dict[str, Any]: ...


def resolve_research_primary_train(
    cfg: Mapping[str, Any],
    *,
    with_meta: bool = False,
) -> str | dict[str, Any]:
    """Return primary train id; refuse unknown / missing values."""
    raw = cfg.get("primary_train")
    if raw is None or str(raw).strip() == "":
        raise ValueError("primary_train is required for research alpha trial")
    primary = str(raw).strip()
    # Normalize aliases onto historical_arm_env when hist-shaped.
    if primary in ("historical", "optionmetrics"):
        primary = RESEARCH_PRIMARY_HIST
    if primary not in RESEARCH_PRIMARY_ALLOWED:
        raise ValueError(
            f"unsupported primary_train={primary!r}; "
            f"allowed={sorted(RESEARCH_PRIMARY_ALLOWED)}"
        )
    warm = int(cfg.get("synth_warmstart_episodes", 0) or 0)
    if with_meta:
        return {
            "primary_train": primary,
            "synth_warmstart_episodes": warm,
            "synth_primary": False,
        }
    return primary


def research_train_friction_pair(
    cfg: Mapping[str, Any],
) -> tuple[FrictionSpec, FrictionSpec]:
    """Train and OOS FrictionSpec from the same cfg (matched costs)."""
    spec = friction_spec_from_cfg(cfg)
    # Identical objects by value; caller must assert_friction_parity.
    return spec, friction_spec_from_cfg(cfg)


def assert_train_claim_label_align(cfg: Mapping[str, Any]) -> None:
    """Fail closed when ``train_label_stem`` disagrees with ``claim_label_stem``."""
    train = cfg.get("train_label_stem")
    claim = cfg.get("claim_label_stem")
    if train is None or claim is None:
        return
    if str(train).strip() != str(claim).strip():
        raise AssertionError(
            f"train_label_stem={train!r} != claim_label_stem={claim!r}"
        )


def resolve_arm_id(cfg: Mapping[str, Any], arm: Any | None = None) -> str | None:
    if arm is not None:
        aid = getattr(arm, "id", None)
        if aid is not None:
            return str(aid)
    arm_cfg = cfg.get("arm") or {}
    if isinstance(arm_cfg, Mapping) and arm_cfg.get("id") is not None:
        return str(arm_cfg.get("id"))
    if cfg.get("portfolio_arm") is not None:
        return str(cfg.get("portfolio_arm"))
    return None


def should_route_historical_arm_env(
    cfg: Mapping[str, Any],
    arm: Any | None = None,
) -> bool:
    """True when eq arm must train on HistoricalArmEnv (not silent CMDPEnv)."""
    raw = cfg.get("primary_train")
    primary = str(raw).strip() if raw is not None else ""
    if primary in ("historical", "optionmetrics"):
        primary = RESEARCH_PRIMARY_HIST
    arm_id = resolve_arm_id(cfg, arm)
    if arm_id != "eq":
        return False
    if primary == RESEARCH_PRIMARY_HIST:
        return True
    tw = str(cfg.get("train_world") or cfg.get("train_distribution") or "").strip()
    return tw == "historical"
