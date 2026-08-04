"""Human label registry for figures.

Stops raw snake_case identifiers reaching legends, ticks, and axis labels.
Strict mode (``MASCOTRL_STRICT_LABELS=1``) raises on unmapped keys.
"""
from __future__ import annotations

import os
import re

STRATEGY_LABELS: dict[str, str] = {
    "policy": "RL policy",
    "happo": "HAPPO policy",
    "no_trade": "No trade",
    "equal_weight": "Equal weight",
    "equal_weight_tau_matched": "Equal weight (tau-matched)",
    "cap_weight_bah": "Cap-weighted buy and hold",
    "buy_and_hold": "Buy and hold",
    "long": "Long-only baseline",
    "random": "Random baseline",
    "sign_lag": "Sign-lag baseline",
    "inverse_vol": "Inverse volatility",
    "parity_erc": "Equal risk contribution",
    "risk_parity_erc": "Equal risk contribution",
    "min_variance": "Minimum variance",
    "min_variance_lw": "Minimum variance (Ledoit-Wolf)",
    "mean_variance": "Mean variance",
    "mv_shrinkage": "Mean variance (shrinkage)",
    "max_diversification": "Maximum diversification",
    "hrp": "Hierarchical risk parity",
    "vol_managed_long": "Volatility-managed long",
    "vol_managed": "Volatility-managed",
    "low_vol_long": "Low-volatility long",
    "mom_12_1": "Momentum 12 minus 1",
    "xs_momentum_12_1": "Momentum 12 minus 1",
    "short_reversal": "Short-term reversal",
    "short_term_reversal": "Short-term reversal",
    "ridge": "Ridge signal portfolio",
    "olps:pamr": "PAMR (online)",
    "olps:ons": "ONS (online)",
    "olps:eg": "Exponentiated gradient (online)",
    "ceiling:kelly_cnn": "Kelly CNN ceiling",
    "ceiling:ridge_composite": "Ridge composite ceiling",
}

METRIC_LABELS: dict[str, str] = {
    "total_net": "Total net return",
    "residual": "Residual (FF4-adjusted) return",
    "gross": "Gross return",
    "cost": "Transaction cost",
    "turnover": "Turnover",
    "sharpe": "Annualised Sharpe ratio (dimensionless)",
    "dh_ret_lagdelta": "Delta-hedged option return (lagged delta)",
    "stk_ret": "Equity total return",
}

AXIS_LABELS: dict[str, str] = {
    "date": "Date",
    "turnover": "Turnover (fraction of NAV per day)",
    "weight": "Portfolio weight (fraction of NAV)",
    "cum_return": "Cumulative net return (fraction of initial NAV)",
    "sharpe": "Annualised Sharpe ratio (dimensionless)",
    "cvar": "Conditional Value-at-Risk (fraction of NAV)",
    "entropic": "Entropic risk measure (dimensionless)",
    "tilt": "Active weight versus equal weight (fraction of NAV)",
    "participation": "ADV participation rate (fraction)",
    "vix_z": "VIX, 252-day rolling z-score (dimensionless)",
    "entropy": "Action entropy (nats)",
    "hhi": "Herfindahl concentration (dimensionless)",
    "n_names": "Number of names (count)",
    "stage": "Selection stage",
    "gate": "Gate",
    "fold": "Fold index",
    "source": "Data source",
    "control": "Negative control",
    "cell": "Spectrum cell",
    "sleeve": "Sleeve",
    "arm": "Arm",
    "spread_mult": "Spread multiplier (dimensionless)",
    "effect": "Effect size versus reference (Sharpe units)",
    "coeff": "Regression coefficient (tilt units per predictor unit)",
    "rotation": "Rotation rate (fraction of NAV per day)",
    "concentration": "Concentration (Herfindahl, dimensionless)",
    "update": "Training update (index)",
    "mean_reward": "Mean episode reward (return units)",
    "policy_loss": "Policy loss (nats)",
    "value_loss": "Value loss (squared return units)",
    "approx_kl": "Approximate KL divergence (nats)",
    "channel_group": "Channel group",
    "l1_delta": "L1 weight-path delta (fraction of NAV)",
    "r2_oos": "Out-of-sample distillation R-squared (dimensionless)",
}

SLEEVE_LABELS: dict[str, str] = {
    "trend": "Trend",
    "reversal": "Reversal",
    "carry": "Carry",
    "defensive": "Defensive",
    "lottery": "Lottery",
    "illiquid": "Illiquid",
    "core": "Core",
}

ARCHETYPE_LABELS: dict[str, str] = {
    "trend_follower": "Trend follower",
    "contrarian": "Contrarian",
    "risk_manager": "Risk manager",
    "speculator": "Speculator",
    "tactical_rotator": "Tactical rotator",
    "mixed": "Balanced",
}

_KIND_MAP: dict[str, dict[str, str]] = {
    "strategy": STRATEGY_LABELS,
    "metric": METRIC_LABELS,
    "axis": AXIS_LABELS,
    "sleeve": SLEEVE_LABELS,
    "archetype": ARCHETYPE_LABELS,
}

_SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")


def strict_labels_enabled() -> bool:
    return os.environ.get("MASCOTRL_STRICT_LABELS", "").strip() in ("1", "true", "TRUE", "yes")


def is_snake_case(text: str) -> bool:
    return bool(_SNAKE.match(str(text).strip()))


def human(key: str, *, kind: str = "strategy") -> str:
    """Return the human label, or raise in strict mode on an unmapped id."""
    table = _KIND_MAP.get(kind, STRATEGY_LABELS)
    k = str(key)
    if k in table:
        return table[k]
    # Allow already-human strings through when not snake_case.
    if not is_snake_case(k) and " " in k:
        return k
    if kind == "axis" and k in AXIS_LABELS:
        return AXIS_LABELS[k]
    if strict_labels_enabled():
        raise KeyError(f"unmapped {kind} label id: {k!r} (MASCOTRL_STRICT_LABELS=1)")
    # Soft fallback: title-case the last path segment without claiming a registry hit.
    return k.replace("_", " ").replace(":", " ").strip().title() or k


# ---------------------------------------------------------------------------
# Spectrum stem short labels (B1 / B2 / B4 heatmaps, titles, legends)
# ---------------------------------------------------------------------------

_STEM_PREFIXES: tuple[str, ...] = (
    "eq_K100_single_",
    "eq_K200_single_",
    "eq_K100_multi_",
    "eq_K200_multi_",
)

_STEM_TOKEN_MAP: tuple[tuple[str, str], ...] = (
    # Longer / more specific tokens first.
    ("pm-archetype_carry", "carry"),
    ("pm-archetype_inflation", "inflation"),
    ("differential_sharpe", "DiffSharpe"),
    ("mean_std_cao", "cost-aware"),
    ("meanvar_kolm", "mean-var"),
    ("sparse_tilt", "sparse"),
    ("tanh_l1", "tanh-L1"),
    ("entmax_15", "entmax"),
    ("mtm_pnl", "profit"),
    ("uni-crucible", ""),
    ("tw-rbergomi", ""),
    ("tw-historical", ""),
    ("hardtau", "hard-tau"),
    ("softmax", "softmax"),
    ("featnet", ""),
    ("weekly_rebal", "weekly"),
    ("relaxed_turnover", "relax-TO"),
    ("mikkila_asym", "mikkila"),
    ("entropic_oce", "entropic"),
    ("cvar_ru", "CVaR"),
    ("smse", "SMSE"),
)

_STEM_ALGO_MAP: dict[str, str] = {
    "ppo": "PPO",
    "td3": "TD3",
    "sac": "SAC",
    "ddpg": "DDPG",
    "dqn": "DQN",
    "cppo": "CPPO",
    "mcpg": "MCPG",
    "rrl": "RRL",
    "happo": "HAPPO",
}

# Locked desk roster (stem -> animal). Prefer this over algorithmic short labels.
LOCKED_ROSTER_STEMS: dict[str, str] = {
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_tw-rbergomi": "Fox",
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_hardtau": "Tortoise",
    "eq_K100_single_ppo_mlp_sparse_tilt_smse": "Cheetah",
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_uni-crucible": "Magpie",
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_pm-archetype_carry": "Hummingbird",
    "eq_K100_single_td3_mlp_softmax_mtm_pnl": "Owl",
}

_MASCOT_DISPLAY: dict[str, str] = {
    "cheetah": "Cheetah",
    "fox": "Fox",
    "tortoise": "Tortoise",
    "magpie": "Magpie",
    "hummingbird": "Hummingbird",
    "owl": "Owl",
}

_CHILD_RE = re.compile(r"^_?child[_\s-]?(\d+)$", re.IGNORECASE)
_EVAL_ORIGIN = "2014-01-03"


def _normalize_stem_key(stem: str) -> str:
    """Collapse title-cased / spaced stems back toward snake_case keys."""
    s = str(stem).strip()
    if not s:
        return s
    if " " in s and "_" not in s:
        s = s.replace(" ", "_").lower()
    # Strip common panel suffixes.
    for suf in ("_policy_behavior", "_interpretability", ".json"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def stem_short_label(stem: str, *, max_len: int = 25) -> str:
    """Map ``eq_K100_single_ppo_mlp_sparse_tilt_smse`` -> ``PPO sparse SMSE``.

    Keeps labels under ``max_len`` characters for heatmap / radar / legend use.
    Use :func:`roster_codename` when an animal seat name is preferred.
    """
    raw = _normalize_stem_key(stem)
    if not raw:
        return raw
    low = raw.lower()
    if low in _MASCOT_DISPLAY:
        return _MASCOT_DISPLAY[low]
    # Already a short human label.
    if " " in str(stem) and "_" not in str(stem) and len(str(stem)) <= max_len:
        return str(stem)

    body = raw
    for pref in _STEM_PREFIXES:
        if body.startswith(pref):
            body = body[len(pref) :]
            break

    # Drop architecture token when present.
    parts = body.split("_")
    if parts and parts[0].lower() in _STEM_ALGO_MAP:
        algo = _STEM_ALGO_MAP[parts[0].lower()]
        rest = "_".join(parts[1:])
    else:
        algo = ""
        rest = body
    if rest.startswith("mlp_"):
        rest = rest[4:]
    elif rest.startswith("mlp-"):
        rest = rest[4:]

    tokens: list[str] = []
    if algo:
        tokens.append(algo)
    # Greedy replace of known multi-token phrases.
    cursor = rest
    while cursor:
        matched = False
        for src, dst in _STEM_TOKEN_MAP:
            if cursor == src or cursor.startswith(src + "_") or cursor.startswith(src + "-"):
                if dst:
                    tokens.append(dst)
                cursor = cursor[len(src) :]
                if cursor.startswith("_") or cursor.startswith("-"):
                    cursor = cursor[1:]
                matched = True
                break
        if matched:
            continue
        # Single leftover token.
        bit, _, cursor = cursor.partition("_")
        if not bit:
            bit, _, cursor = cursor.partition("-")
        if bit:
            if bit.lower() in _STEM_ALGO_MAP:
                tokens.append(_STEM_ALGO_MAP[bit.lower()])
            elif len(bit) <= 8:
                tokens.append(bit)
            else:
                tokens.append(bit[:6])
    label = " ".join(t for t in tokens if t).strip() or raw
    if len(label) > max_len:
        label = label[:max_len].rstrip()
    return label


def figure_cell_label(stem: str, *, prefer_roster: bool = True) -> str:
    """Prefer locked-roster animal codename, else :func:`stem_short_label`."""
    if prefer_roster:
        animal = roster_codename(stem)
        if animal:
            return animal
    return stem_short_label(stem)


def roster_codename(stem: str) -> str | None:
    """Return locked-roster animal for a stem, if known."""
    return LOCKED_ROSTER_STEMS.get(_normalize_stem_key(stem))


def expert_display_name(
    key: str,
    *,
    expert_names: list[str] | None = None,
    index: int | None = None,
) -> str:
    """Map desk expert keys (``fox``, ``child4``, stems) to human legend labels."""
    k = str(key).strip()
    if not k:
        return "Expert"
    m = _CHILD_RE.match(k)
    if m is not None:
        idx = int(m.group(1))
        if expert_names and 0 <= idx < len(expert_names):
            return expert_display_name(expert_names[idx], expert_names=expert_names)
        return f"Expert {idx}"
    if index is not None and expert_names and 0 <= index < len(expert_names):
        return expert_display_name(expert_names[index], expert_names=expert_names)
    low = k.lower()
    if low in _MASCOT_DISPLAY:
        return _MASCOT_DISPLAY[low]
    if low in {"ew", "equal_weight", "equal-weight"}:
        return "EW benchmark"
    if low in {"hold_leader", "hold_leader_annual", "hold-leader", "primary"}:
        return "Hold-leader"
    if low in {"oracle", "oracle_best_k_shift", "look-ahead oracle"}:
        return "Oracle"
    if low.startswith("eq_"):
        return stem_short_label(k)
    if k[:1].isupper() and " " not in k and "_" not in k:
        return k
    return stem_short_label(k) if "_" in k else k.replace("_", " ").strip().title()


def axis_label_short(key: str, *, max_len: int = 15) -> str:
    """Short axis label for cramped 3D / multi-panel plots."""
    full = human(key, kind="axis")
    shorts = {
        "turnover": "Turnover",
        "concentration": "Concentration",
        "rotation": "Rotation",
        "tilt": "Tilt",
        "cvar": "CVaR",
        "entropic": "Entropic",
        "sharpe": "Sharpe",
    }
    lab = shorts.get(str(key), full)
    if len(lab) > max_len:
        lab = lab[:max_len].rstrip()
    return lab


def parse_eval_dates(dates: list | tuple | None, n: int | None = None):
    """Parse figure eval dates; integer offsets map onto 2014-01-03.

    Returns a pandas DatetimeIndex (or list of Timestamps).
    """
    import pandas as pd

    origin = pd.Timestamp(_EVAL_ORIGIN)

    def _offset(days: int):
        return origin + pd.to_timedelta(int(days), unit="D")

    if dates is None or (hasattr(dates, "__len__") and len(dates) == 0):
        length = int(n or 0)
        return pd.DatetimeIndex([_offset(i) for i in range(length)])

    seq = list(dates)
    if n is not None:
        seq = seq[: int(n)]

    # Integer / float day offsets (including range() materialised as ints).
    if seq and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in seq):
        # Treat small integers / zero-based indices as offsets from eval origin,
        # not as matplotlib/unix epochs (which yield 1970-1977 artifacts).
        vals = [float(x) for x in seq]
        if min(vals) >= 0 and max(vals) < 1e5:
            return pd.DatetimeIndex([_offset(round(v)) for v in vals])

    parsed = pd.to_datetime(seq, errors="coerce", utc=False)
    if getattr(parsed, "isna", lambda: False)().any():  # type: ignore[operator]
        # Mixed / failed parse: fall back to origin offsets by position.
        out = []
        for i, x in enumerate(seq):
            ts = pd.to_datetime(x, errors="coerce")
            if pd.isna(ts):
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    ts = _offset(round(float(x)))
                else:
                    ts = _offset(i)
            out.append(ts)
        return pd.DatetimeIndex(out)
    return pd.DatetimeIndex(parsed)
