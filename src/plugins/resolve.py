"""Resolve ``plugins:`` subtree to an explicit status-quo-safe dict."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


# Defaults match canonical happo_cmdp / train_happo hard-coded construction.
# Equity allocation spine (eq_alloc_* YAML) opts into:
#   plugins.actor_backend: shared   (+ cfg share_temporal_encoder: true)
# STATUS_QUO keeps ModuleList actors / per-asset temporal blocks for paper parity.
STATUS_QUO_PLUGINS: dict[str, Any] = {
    "projection_backend": "cvxpy",
    "dhgnn_mode": "undirected",  # with use_dhgnn=true elsewhere
    "tau_mode": "fixed",
    "actor_backend": "modulelist",
    "critic_backend": "flatten",
    "execution_drag_mode": "fixed",
    "funding": {
        "enabled": False,
        "mode": "sofr_gc",
        "sofr_key": "sofr",
        "sofr_level": None,
        "gc_borrow_bps": 25.0,
        "margin_funding": False,
        "margin_rate_spread_bps": 0.0,
        "notional_proxy": "abs_weight",
        "dt_years": 1.0 / 252.0,
        "name_borrow_path": None,
    },
    "om_touch": {
        "enabled": False,
        "fee_bps": 0.0,
        "spread_multiplier": 1.0,
        "hedge_leg_spread_bps": 5.0,
    },
    # OOS / CPCV measurement only — defaults off (paper spine MTM train stays clean).
    "hedge_impact": {
        "enabled": False,
        "coef": 1.0,
        "adv": None,
        "sigma": None,
    },
    "pretrain": {
        "enabled": False,
        "method": "infonce",
        "checkpoint": None,
        "freeze_steps": 0,
        "finetune_lr_mult": 0.1,
        "temperature": 0.1,
    },
    "tau": {
        "tau0": None,  # filled from turnover_limit
        "tau_min": 0.05,
        "tau_max": 0.40,
        "vix_z_ref": 0.0,
        "vix_z_scale": 0.25,
        "vix_macro_index": 0,
    },
    "admm": {
        "max_iters": 50,
        "rho": 1.0,
        "abs_tol": 1.0e-5,
        "rel_tol": 1.0e-4,
        "fallback_to_cvxpy": True,
    },
    "multibook": {
        "n_books": 5,
        "book_size": 50,
        "partition": "fixed_shards",
    },
    "overlay": {
        "delta_mode": "soft",
        "option_slots": None,
    },
    "dhgnn": {
        "tail_threshold": 0.90,
        "lower_tail_threshold": 0.90,
        "edge_threshold": 0.35,
        "top_m": 2,
        "laplace_alpha": 1.0,
    },
    "execution_drag": {
        "vol_ref": 0.20,
        "vol_floor": 0.05,
        "vol_cap": 1.0,
    },
    "hypernet": {
        "embed_dim": 16,
        "hidden": 64,
        "condition_on": ["atm_iv", "tail_centrality"],
    },
    "market": {
        "train_universe": "us_optionmetrics",
        "eval_universe": None,
        "zero_shot": False,
    },
}

_ALLOWED_PROJECTION = {"cvxpy", "admm", "multibook_cvxpy", "overlay_cvxpy"}
_ALLOWED_DHGNN = {"undirected", "directed", "off"}
_ALLOWED_TAU = {"fixed", "macro_schedule"}
_ALLOWED_ACTOR = {"modulelist", "hypernet", "shared", "shared_mappo"}
_ALLOWED_CRITIC = {"flatten", "deepsets"}
_ALLOWED_DRAG = {"fixed", "vol_scaled"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in out:
            raise KeyError(f"unknown plugins key: {k!r}")
        if isinstance(v, dict) and isinstance(out[k], dict):
            for sk, sv in v.items():
                if sk not in out[k]:
                    raise KeyError(f"unknown plugins.{k} key: {sk!r}")
                out[k][sk] = sv
        else:
            out[k] = v
    return out


# @lat: [[plugins#Registry]]
def resolve_plugins(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge optional ``cfg['plugins']`` onto status-quo defaults.

    Omitting ``plugins:`` entirely yields canonical happo_cmdp behavior.
    ``use_dhgnn: false`` forces ``dhgnn_mode: off`` for backward compat.
    """
    cfg = cfg or {}
    raw = cfg.get("plugins")
    if raw is None:
        plugins = copy.deepcopy(STATUS_QUO_PLUGINS)
    elif not isinstance(raw, dict):
        raise TypeError("cfg['plugins'] must be a mapping")
    else:
        plugins = _deep_merge(STATUS_QUO_PLUGINS, raw)

    # Align tau0 with existing turnover_limit when not set under plugins.tau.
    if plugins["tau"]["tau0"] is None:
        plugins["tau"]["tau0"] = float(cfg.get("turnover_limit", 0.15))

    # Backward compat: use_dhgnn=false ⇒ dhgnn off.
    if "use_dhgnn" in cfg and not bool(cfg.get("use_dhgnn", True)):
        plugins["dhgnn_mode"] = "off"

    # Validate enums.
    if plugins["projection_backend"] not in _ALLOWED_PROJECTION:
        raise ValueError(
            f"plugins.projection_backend={plugins['projection_backend']!r} "
            f"not in {_ALLOWED_PROJECTION}"
        )
    if plugins["dhgnn_mode"] not in _ALLOWED_DHGNN:
        raise ValueError(f"plugins.dhgnn_mode={plugins['dhgnn_mode']!r}")
    if plugins["tau_mode"] not in _ALLOWED_TAU:
        raise ValueError(f"plugins.tau_mode={plugins['tau_mode']!r}")
    if plugins["actor_backend"] not in _ALLOWED_ACTOR:
        raise ValueError(f"plugins.actor_backend={plugins['actor_backend']!r}")
    if plugins["critic_backend"] not in _ALLOWED_CRITIC:
        raise ValueError(f"plugins.critic_backend={plugins['critic_backend']!r}")
    if plugins["execution_drag_mode"] not in _ALLOWED_DRAG:
        raise ValueError(
            f"plugins.execution_drag_mode={plugins['execution_drag_mode']!r}"
        )
    return plugins


def dump_resolved_plugins(plugins: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plugins, indent=2, default=str) + "\n")
    return path


def is_status_quo(plugins: dict[str, Any]) -> bool:
    """True when resolved plugins match status-quo algorithm + market choices."""
    market = plugins.get("market") or {}
    return (
        plugins.get("projection_backend") == "cvxpy"
        and plugins.get("dhgnn_mode") in ("undirected", "off")
        and plugins.get("tau_mode") == "fixed"
        and plugins.get("actor_backend") == "modulelist"
        and plugins.get("critic_backend") == "flatten"
        and plugins.get("execution_drag_mode") == "fixed"
        and not bool((plugins.get("funding") or {}).get("enabled", False))
        and not bool((plugins.get("pretrain") or {}).get("enabled", False))
        and not bool((plugins.get("om_touch") or {}).get("enabled", False))
        and not bool(market.get("zero_shot", False))
    )
