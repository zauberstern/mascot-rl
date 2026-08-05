"""Feature group registry and YAML exclude resolution (default all-on)."""
from __future__ import annotations

from typing import Iterable, Sequence

from src.features.blocks.fundamentals_pit import FUNDAMENTALS_PIT_CHANNELS
from src.features.blocks.iv_surface import DEFAULT_SURFACE_CHANNELS
from src.features.blocks.returns_momentum import LOG_RETURN_WINDOWS

CORE_MOMENTUM: tuple[str, ...] = tuple(
    name
    for w in LOG_RETURN_WINDOWS
    for name in (f"log_ret_{w}", f"price_rel_{w}")
) + ("mom_12_1", "resid_mom_12_1")

CORE_VOLATILITY: tuple[str, ...] = ("hv_21", "hv_63", "hv_252", "vol_of_vol", "vrp")

CORE_LIQUIDITY: tuple[str, ...] = ("amihud", "adv_dollar_volume")

RANGE_VOLATILITY: tuple[str, ...] = (
    "parkinson_21",
    "garman_klass_21",
    "rogers_satchell_21",
    "yang_zhang_21",
)

MICROSTRUCTURE: tuple[str, ...] = (
    "eff_spread_21",
    "vwap_dev_5",
    "block_share_21",
    "turnover_21",
)

SENTIMENT: tuple[str, ...] = (
    "si_pct",
    "si_pct_chg_21",
    "rec_mean_inv",
    "rec_chg_63",
    "pt_gap",
)

OPTION_FLOW: tuple[str, ...] = (
    "pc_vol",
    "pc_oi",
    "opt_stock_vol",
    "oi_chg_21",
    "div_yield_ttm",
)

JKP_LOTTERY: tuple[str, ...] = (
    "max_ret_21",
    "min_ret_21",
    "ret_skew_63",
    "ret_kurt_63",
    "idio_vol_ff4_21",
    "beta_asym_63",
    "log_me",
    "ivol_capm_21d",
    "ret_1_0",
)

EXPERIMENTAL: tuple[str, ...] = (
    "x_range_cc_ratio",
    "x_overnight_share_21",
    "x_vrp_term",
    "x_spread_x_amihud",
    "x_si_x_mom",
    "x_gics_rel_mom_21",
    "x_oi_skew_flow",
    "x_tail_dep_asym_63",
)

BORROW: tuple[str, ...] = ("borrow_rate", "borrow_rate_fee")

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "core_momentum": CORE_MOMENTUM,
    "core_volatility": CORE_VOLATILITY,
    "core_liquidity": CORE_LIQUIDITY,
    "surface_signals": tuple(DEFAULT_SURFACE_CHANNELS),
    "borrow": BORROW,
    "kelly_images": (),  # dynamic kelly_px_* handled by prefix
    "macro": (),  # dynamic macro_* / rates handled by prefix / runtime names
    "range_volatility": RANGE_VOLATILITY,
    "microstructure": MICROSTRUCTURE,
    "fundamentals_pit": FUNDAMENTALS_PIT_CHANNELS,
    "sentiment": SENTIMENT,
    "option_flow": OPTION_FLOW,
    "jkp_lottery": JKP_LOTTERY,
    "experimental": EXPERIMENTAL,
}

_KELLY_PREFIX = "kelly_px_"
_MACRO_PREFIXES = ("macro_", "zrate_", "term_", "d_term_")


def all_registered_channel_names() -> frozenset[str]:
    names: set[str] = set()
    for group, chans in FEATURE_GROUPS.items():
        names.update(chans)
    return frozenset(names)


def resolve_excludes(
    groups_exclude: Sequence[str] | None,
    channels_exclude: Sequence[str] | None,
    all_names: Iterable[str],
) -> set[str]:
    """Return channel names to drop. Unknown group/channel fails closed."""
    names = list(all_names)
    name_set = set(names)
    drop: set[str] = set()
    for g in groups_exclude or ():
        g = str(g)
        if g not in FEATURE_GROUPS:
            raise ValueError(
                f"unknown feature_groups_exclude entry {g!r}; "
                f"known={sorted(FEATURE_GROUPS)}"
            )
        if g == "kelly_images":
            drop.update(n for n in names if n.startswith(_KELLY_PREFIX))
        elif g == "macro":
            drop.update(
                n
                for n in names
                if n.startswith(_MACRO_PREFIXES)
                or n.startswith("macro_")
                or n in {"term_slope", "term_curv", "d_term_slope_21"}
                or n.startswith("zrate_")
            )
        else:
            for ch in FEATURE_GROUPS[g]:
                if ch in name_set:
                    drop.add(ch)
    for ch in channels_exclude or ():
        ch = str(ch)
        if ch not in name_set and not ch.startswith(_KELLY_PREFIX):
            # Allow excluding a registered name even if absent from this cube.
            if ch not in all_registered_channel_names() and not any(
                ch.startswith(p) for p in ("kelly_px_", "macro_", "fund_", "surf_", "zrate_")
            ):
                raise ValueError(
                    f"unknown feature_channels_exclude entry {ch!r}"
                )
        if ch in name_set:
            drop.add(ch)
        elif ch.startswith(_KELLY_PREFIX):
            drop.update(n for n in names if n == ch)
    return drop
