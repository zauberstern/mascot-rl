"""Factory entry points — status-quo defaults instantiate today's classes."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from mascotrl.features.dhgnn import SpatialDHGNN
from mascotrl.features.extractor import AlphaFeatureExtractor
from mascotrl.plugins.admm_projection import ADMMProjectionLayer
from mascotrl.plugins.directed_dhgnn import DirectedSpatialDHGNN
from mascotrl.plugins.funding_drag import FundingDrag
from mascotrl.plugins.multibook_projection import MultibookProjectionLayer
from mascotrl.plugins.resolve import resolve_plugins
from mascotrl.plugins.tau_schedule import FixedTau, MacroScheduleTau
from mascotrl.policy.convex_projection import ConvexProjectionLayer
from mascotrl.policy.overlay_projection import OverlayProjectionLayer
from mascotrl.policy.happo import HAPPOEngine


def build_tau_schedule(cfg: dict[str, Any] | None = None, plugins: dict | None = None):
    plugins = plugins or resolve_plugins(cfg or {})
    tau_cfg = plugins.get("tau") or {}
    # Honor explicit tau0=0.0 (do not treat as missing via `or`).
    if tau_cfg.get("tau0") is not None:
        tau0 = float(tau_cfg["tau0"])
    else:
        tau0 = float((cfg or {}).get("turnover_limit", 0.15))
    if plugins.get("tau_mode", "fixed") == "macro_schedule":
        return MacroScheduleTau(
            tau0=tau0,
            tau_min=float(tau_cfg.get("tau_min", 0.05)),
            tau_max=float(tau_cfg.get("tau_max", 0.40)),
            vix_z_ref=float(tau_cfg.get("vix_z_ref", 0.0)),
            vix_z_scale=float(tau_cfg.get("vix_z_scale", 0.25)),
            vix_macro_index=int(tau_cfg.get("vix_macro_index", 0)),
        )
    return FixedTau(tau0=tau0)


def build_projection(
    num_assets: int,
    cfg: dict[str, Any] | None = None,
    plugins: dict | None = None,
) -> nn.Module:
    plugins = plugins or resolve_plugins(cfg or {})
    cfg = cfg or {}
    tau0 = float(cfg.get("turnover_limit", 0.15))
    max_name = float(cfg.get("risk_max_name_abs_weight", 5.0))
    backend = plugins.get("projection_backend", "cvxpy")
    if backend == "cvxpy":
        return ConvexProjectionLayer(
            num_assets, turnover_limit=tau0, max_name_abs_weight=max_name
        )
    if backend == "overlay_cvxpy":
        from mascotrl.arms import arm_spec_from_cfg

        ov = plugins.get("overlay") or {}
        arm = arm_spec_from_cfg(cfg)
        delta_mode = str(ov.get("delta_mode") or arm.delta_mode)
        option_slots = ov.get("option_slots")
        if option_slots is None:
            option_slots = arm.option_slots if arm.option_slots > 0 else num_assets
        return OverlayProjectionLayer(
            num_assets,
            delta_mode=delta_mode,
            option_slots=int(option_slots),
            turnover_limit=tau0,
            max_name_abs_weight=max_name,
        )
    if backend == "admm":
        admm = plugins.get("admm") or {}
        return ADMMProjectionLayer(
            num_assets,
            turnover_limit=tau0,
            max_name_abs_weight=max_name,
            max_iters=int(admm.get("max_iters", 50)),
            rho=float(admm.get("rho", 1.0)),
            abs_tol=float(admm.get("abs_tol", 1e-5)),
            rel_tol=float(admm.get("rel_tol", 1e-4)),
            fallback_to_cvxpy=bool(admm.get("fallback_to_cvxpy", True)),
            use_ste=bool(admm.get("use_ste", True)),
        )
    if backend == "multibook_cvxpy":
        mb = plugins.get("multibook") or {}
        partition = str(mb.get("partition", "fixed_shards"))
        if partition == "sector_gics" and mb.get("sector_ids") is None:
            raise ValueError(
                "multibook.partition=sector_gics requires multibook.sector_ids "
                "(integer id per asset); lake has no GICS today — use "
                "fixed_shards or cluster_copula"
            )
        scores = mb.get("sector_ids")
        if scores is not None and not hasattr(scores, "shape"):
            import torch

            scores = torch.as_tensor(scores)
        return MultibookProjectionLayer(
            num_assets,
            turnover_limit=tau0,
            max_name_abs_weight=max_name,
            n_books=int(mb.get("n_books", 5)),
            book_size=int(mb.get("book_size", 50)),
            partition=partition,
            partition_scores=scores,
        )
    raise ValueError(f"unknown projection_backend={backend!r}")


def build_spatial_dhgnn(
    d_model: int,
    num_assets: int,
    cfg: dict[str, Any] | None = None,
    plugins: dict | None = None,
) -> nn.Module | None:
    plugins = plugins or resolve_plugins(cfg or {})
    mode = plugins.get("dhgnn_mode", "undirected")
    if mode == "off":
        return None
    dcfg = plugins.get("dhgnn") or {}
    if mode == "directed":
        return DirectedSpatialDHGNN(
            d_model=d_model,
            num_assets=num_assets,
            tail_threshold=float(dcfg.get("tail_threshold", 0.90)),
            lower_tail_threshold=float(dcfg.get("lower_tail_threshold", 0.90)),
            edge_threshold=float(dcfg.get("edge_threshold", 0.35)),
            top_m=int(dcfg.get("top_m", 2)),
            laplace_alpha=float(dcfg.get("laplace_alpha", 1.0)),
        )
    # undirected status quo; spectrum spatial_mode selects pearson vs copula.
    spatial_mode = str(
        (cfg or {}).get("spatial_mode")
        or dcfg.get("spatial_mode")
        or "dhgnn_copula"
    )
    edge_thr = float(
        dcfg.get(
            "edge_threshold",
            (cfg or {}).get("universe_iv_corr_threshold", 0.35),
        )
    )
    return SpatialDHGNN(
        d_model=d_model,
        num_assets=num_assets,
        tail_threshold=float(dcfg.get("tail_threshold", 0.90)),
        edge_threshold=edge_thr,
        spatial_mode=spatial_mode,
        update_incidence_at_eval=bool(
            dcfg.get(
                "update_incidence_at_eval",
                (cfg or {}).get("update_incidence_at_eval", False),
            )
        ),
        allow_pearson_incidence=bool(
            dcfg.get(
                "allow_pearson_incidence",
                (cfg or {}).get("allow_pearson_incidence", spatial_mode == "dhgnn_pearson"),
            )
        ),
    )


def build_feature_extractor(
    num_assets: int,
    d_model: int,
    d_state: int = 16,
    cfg: dict[str, Any] | None = None,
    plugins: dict | None = None,
) -> AlphaFeatureExtractor:
    plugins = plugins or resolve_plugins(cfg or {})
    cfg = cfg or {}
    spatial_mode = str(cfg.get("spatial_mode", "dhgnn_copula")).lower().strip()
    use_dhgnn = plugins.get("dhgnn_mode", "undirected") != "off"
    if spatial_mode in ("none", "off", "") and not bool(cfg.get("use_dhgnn", False)):
        use_dhgnn = False
    elif "use_dhgnn" in cfg:
        use_dhgnn = bool(cfg["use_dhgnn"]) and use_dhgnn
    fe = AlphaFeatureExtractor(
        num_assets,
        d_model,
        d_state=d_state,
        temporal_backend=str(cfg.get("temporal_backend", cfg.get("architecture", "mamba"))),
        use_dhgnn=use_dhgnn,
        spatial_mode=str(cfg.get("spatial_mode", "dhgnn_copula")),
        # eq_alloc YAML sets share_temporal_encoder: true; STATUS_QUO stays False.
        share_temporal_encoder=bool(cfg.get("share_temporal_encoder", False)),
        update_incidence_at_eval=bool(cfg.get("update_incidence_at_eval", False)),
    )
    # Always apply plugins.dhgnn knobs (undirected thresholds OR directed sibling).
    if use_dhgnn:
        spatial = build_spatial_dhgnn(
            d_model, num_assets, cfg=cfg, plugins=plugins
        )
        if spatial is not None:
            fe.spatial_dhgnn = spatial
    return fe


def build_happo_engine(
    num_assets: int,
    enriched_dim: int,
    macro_dim: int,
    cfg: dict[str, Any] | None = None,
    plugins: dict | None = None,
) -> HAPPOEngine:
    from mascotrl.reporting.capital_gates import PROJECTION_K_CEILING

    plugins = plugins or resolve_plugins(cfg or {})
    cfg = cfg or {}
    tau0 = float(cfg.get("turnover_limit", 0.15))
    aps = cfg.get("actor_portfolio_state", plugins.get("actor_portfolio_state", False))
    if isinstance(aps, dict):
        aps = bool(aps.get("enabled", False))

    backend = str(plugins.get("projection_backend") or "cvxpy")
    # Exact cvxpylayers SCS is the K<=50 ceiling; K>50 defaults must use ADMM
    # and must not construct a CvxpyLayer at all (DPP compile stalls narrative).
    forced_admm = backend == "cvxpy" and int(num_assets) > int(PROJECTION_K_CEILING)
    if forced_admm:
        plugins = dict(plugins)
        plugins["projection_backend"] = "admm"
        admm = dict(plugins.get("admm") or {})
        admm["fallback_to_cvxpy"] = False
        plugins["admm"] = admm
        backend = "admm"

    want_proj = bool(cfg.get("use_projection", True))
    skip_cvx_ctor = want_proj and backend != "cvxpy"
    engine = HAPPOEngine(
        num_assets,
        enriched_dim,
        macro_dim,
        turnover_limit=tau0,
        use_projection=want_proj and not skip_cvx_ctor,
        max_name_abs_weight=float(cfg.get("risk_max_name_abs_weight", 5.0)),
        actor_backend=str(plugins.get("actor_backend", "modulelist")),
        critic_backend=str(plugins.get("critic_backend", "flatten")),
        hypernet_cfg=plugins.get("hypernet") or {},
        initial_log_std=float(cfg.get("initial_log_std", -2.0)),
        enable_cost_critic=bool((cfg.get("cmdp") or {}).get("enabled", False)),
        actor_portfolio_state=bool(aps),
    )
    if skip_cvx_ctor:
        engine.use_projection = True
        engine.convex_projection = build_projection(
            num_assets, cfg=cfg, plugins=plugins
        )
        engine._projection_backend = backend
    elif not want_proj:
        engine._projection_backend = "none"
    else:
        engine._projection_backend = "cvxpy"
    engine.tau_schedule = build_tau_schedule(cfg=cfg, plugins=plugins)
    return engine


def build_funding(
    cfg: dict[str, Any] | None = None, plugins: dict | None = None
) -> FundingDrag:
    plugins = plugins or resolve_plugins(cfg or {})
    f = plugins.get("funding") or {}
    sofr_level = f.get("sofr_level", None)
    name_path = f.get("name_borrow_path") or cfg.get("name_borrow_path")
    fd = FundingDrag(
        enabled=bool(f.get("enabled", False)),
        mode=str(f.get("mode", "sofr_gc")),
        gc_borrow_bps=float(f.get("gc_borrow_bps", 25.0)),
        margin_funding=bool(f.get("margin_funding", False)),
        margin_rate_spread_bps=float(f.get("margin_rate_spread_bps", 0.0)),
        notional_proxy=str(f.get("notional_proxy", "abs_weight")),
        dt_years=float(f.get("dt_years", 1.0 / 252.0)),
        sofr_key=str(f.get("sofr_key", "sofr")),
        sofr_level=float(sofr_level) if sofr_level is not None else None,
        name_borrow_path=str(name_path) if name_path else None,
    )
    if fd.name_borrow_path:
        tickers = (cfg or {}).get("tickers") or (cfg or {}).get("universe_tickers")
        if tickers is not None:
            tickers = [str(t) for t in list(tickers)]
        fd.ensure_name_schedule(
            tickers=tickers,
            n_assets=int((cfg or {}).get("n_assets") or 0) or None,
        )
    return fd


def env_drag_kwargs(
    cfg: dict[str, Any] | None = None, plugins: dict | None = None
) -> dict[str, Any]:
    """Extra CMDPEnv kwargs for execution drag + funding plugins."""
    plugins = plugins or resolve_plugins(cfg or {})
    cfg = cfg or {}
    drag = plugins.get("execution_drag") or {}
    return {
        "execution_spread_bps": float(cfg.get("execution_spread_bps", 0.0)),
        "execution_impact_coef": float(cfg.get("execution_impact_coef", 0.0)),
        "execution_drag_mode": str(plugins.get("execution_drag_mode", "fixed")),
        "execution_vol_ref": float(drag.get("vol_ref", 0.20)),
        "execution_vol_floor": float(drag.get("vol_floor", 0.05)),
        "execution_vol_cap": float(drag.get("vol_cap", 1.0)),
        "funding": build_funding(cfg=cfg, plugins=plugins),
    }


def oos_friction_kwargs(
    cfg: dict[str, Any] | None = None, plugins: dict | None = None
) -> dict[str, Any]:
    """Hist-OOS / shadow friction kwargs (includes OM-touch measurement)."""
    plugins = plugins or resolve_plugins(cfg or {})
    touch = plugins.get("om_touch") or {}
    out = env_drag_kwargs(cfg=cfg, plugins=plugins)
    out["om_touch_enabled"] = bool(touch.get("enabled", False))
    out["om_touch_fee_bps"] = float(touch.get("fee_bps", 0.0))
    out["om_touch_spread_multiplier"] = float(touch.get("spread_multiplier", 1.0))
    out["hedge_leg_spread_bps"] = float(
        (cfg or {}).get("hedge_leg_spread_bps", touch.get("hedge_leg_spread_bps", 0.0))
    )
    out["hedge_frequency"] = str((cfg or {}).get("hedge_frequency", "daily"))
    hi = plugins.get("hedge_impact") or {}
    cfg_d = cfg or {}
    out["hedge_impact_enabled"] = bool(
        cfg_d.get("hedge_impact_enabled", hi.get("enabled", False))
    )
    out["hedge_impact_coef"] = float(
        cfg_d.get("hedge_impact_coef", hi.get("coef", 1.0)) or 1.0
    )
    adv = cfg_d.get("hedge_adv", hi.get("adv"))
    out["hedge_adv"] = float(adv) if adv is not None else None
    sigma = cfg_d.get("hedge_sigma", hi.get("sigma"))
    out["hedge_sigma"] = float(sigma) if sigma is not None else None
    ca = cfg_d.get("corporate_actions") or {}
    path = ca.get("table_path") or cfg_d.get("corporate_actions_path")
    if path:
        out["corporate_actions_path"] = str(path)
    tickers = cfg_d.get("tickers") or cfg_d.get("universe_tickers")
    if tickers is not None:
        out["corporate_actions_tickers"] = [str(t) for t in list(tickers)]
    return out
