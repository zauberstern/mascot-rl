"""
Characteristic sorts and factor-adjusted alpha for delta-hedged option returns.

Why this exists. The sharpest current threat to any claimed option-return edge
is that a suitable factor model absorbs it: Goyal and Saretto (2024, RFS 38(6)),
*Can equity option returns be explained by a factor model? IPCA says yes*, and
the companion Dallas Fed study *Are equity option returns abnormal? IPCA says
no* (WP 2214) both reach that conclusion. A raw Sharpe with no factor-adjusted
alpha will not survive review.

Reference factor models in this literature:

  * Buechner and Kelly (2022, JFE 143(3)) — IPCA level, slope, skew factors.
  * Horenstein, Vasquez and Xiao (2025, RFS) — four characteristic factors: the
    equally weighted option portfolio, a historical-minus-implied volatility
    factor, a corporate cash-holdings factor, and a volatility-of-volatility
    factor.
  * Bali, Cao, Song and Zhan (2022) — five delta-hedged call factors sorted on
    option spread, option price, model-free implied kurtosis, and RV minus IV.

What is implemented here is an **HVX-style proxy** restricted to what the
available data supports: the equally weighted option portfolio, HV minus IV, and
volatility of volatility. The cash-holdings and IPCA factors require Compustat
and analyst data that this lake does not carry; that gap is reported explicitly
rather than papered over, because omitting factors biases alpha upward.

Portfolio construction follows the convention of the delta-hedged option
literature: equal-weighted quintile sorts, long the top and short the bottom.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mascotrl.eval.stats_inference import hac_mean_tstat
from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.eval.factor_alpha")

ANNUALIZATION = 252.0

# Characteristics computable from the OptionMetrics + CRSP lake in this repo.
FEASIBLE_CHARACTERISTICS = (
    "vol_deviation",
    "ivol",
    "option_spread",
    "option_price",
    "vol_of_vol",
)

# Documented as unavailable so the paper states the omission rather than
# implying the factor set is complete.
UNAVAILABLE_CHARACTERISTICS = {
    "cash_holdings": "requires Compustat balance-sheet data",
    "analyst_dispersion": "requires I/B/E/S analyst forecasts",
    "profit_margin": "requires Compustat fundamentals",
    "model_free_implied_kurtosis": "requires full strike grid per name/date",
    "ipca_latent_factors": "requires IPCA estimation on a full contract panel",
}


def _xs_rank_weights(x: np.ndarray, n_quantiles: int = 5) -> np.ndarray:
    """
    Equal-weighted long-short quintile weights from a cross-section.

    Long the top quantile, short the bottom, each leg summing to 1 in absolute
    weight so the portfolio is dollar-neutral within the delta-hedged space.
    Names with a missing characteristic get zero weight.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    w = np.zeros(x.size, dtype=np.float64)
    n = int(finite.sum())
    if n < 2 * n_quantiles:
        return w
    idx = np.where(finite)[0]
    order = idx[np.argsort(x[idx], kind="stable")]
    m = max(1, n // n_quantiles)
    short_leg, long_leg = order[:m], order[-m:]
    w[long_leg] = 1.0 / len(long_leg)
    w[short_leg] = -1.0 / len(short_leg)
    return w


def long_short_sort(
    characteristic: np.ndarray,
    labels: np.ndarray,
    *,
    n_quantiles: int = 5,
) -> dict[str, Any]:
    """
    Daily equal-weighted long-short return series from a characteristic panel.

    ``characteristic`` and ``labels`` are (T, K); the characteristic at t-1 is
    sorted and held into the label realized over (t-1, t], so the sort never
    uses contemporaneous information.

    The returned ``pnl`` series has length ``T`` and is aligned to the label
    calendar: day 0 holds the first lagged-sort P&L (char[0] × label[0]); the
    final day is NaN when no further lag pair exists. Callers must not compare
    a raw ``T-1`` factor vector to a length-``T`` strategy series.
    """
    C = np.asarray(characteristic, dtype=np.float64)
    R = np.asarray(labels, dtype=np.float64)
    if C.shape != R.shape or C.ndim != 2:
        raise ValueError("characteristic and labels must be equal-shaped (T, K)")
    T = C.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    n_active = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        w = _xs_rank_weights(C[t - 1], n_quantiles=n_quantiles)
        r = R[t - 1]
        mask = np.isfinite(r)
        if not mask.any() or not np.any(w != 0):
            continue
        contrib = np.where(mask, np.nan_to_num(r, nan=0.0), 0.0)
        out[t - 1] = float(np.dot(w, contrib))
        n_active[t - 1] = int(np.sum(w != 0))
    fin = out[np.isfinite(out)]
    return {
        "pnl": out.tolist(),
        "n_days": int(fin.size),
        "mean": float(fin.mean()) if fin.size else float("nan"),
        "sharpe": (
            float(fin.mean() / fin.std(ddof=0) * np.sqrt(ANNUALIZATION))
            if fin.size > 1 and fin.std(ddof=0) > 1e-15
            else float("nan")
        ),
        "mean_names_per_leg": float(np.mean(n_active[n_active > 0])) if np.any(n_active) else 0.0,
        "n_quantiles": int(n_quantiles),
        "construction": "equal_weighted_quintile_long_short_lagged_sort",
    }


def _lagged_ew_market(labels: np.ndarray) -> list[float]:
    """Length-T EW market factor aligned to the label calendar (last day NaN)."""
    R = np.asarray(labels, dtype=np.float64)
    T = R.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(1, T):
        r = R[t - 1]
        m = np.isfinite(r)
        if m.any():
            out[t - 1] = float(np.nanmean(r[m]))
    return out.tolist()


def build_characteristics(
    *,
    atm_iv: np.ndarray,
    bid_ask_spread: np.ndarray,
    mid: np.ndarray,
    realized_vol: np.ndarray | None = None,
    idio_vol: np.ndarray | None = None,
    vol_of_vol_window: int = 60,
) -> dict[str, np.ndarray]:
    """
    Build the feasible characteristic panels, all (T, K) and all lag-safe.

    ``vol_deviation`` is log(HV/IV), the Goyal and Saretto (2009) volatility
    deviation. ``ivol`` should be market-model residual volatility (Cao and Han
    2013); when only total volatility is supplied the caller must label it a
    proxy. ``vol_of_vol`` is the trailing standard deviation of ATM implied
    volatility (Horenstein, Vasquez and Xiao).
    """
    iv = np.asarray(atm_iv, dtype=np.float64)
    out: dict[str, np.ndarray] = {}

    out["option_spread"] = np.divide(
        np.asarray(bid_ask_spread, dtype=np.float64),
        np.where(np.abs(mid) > 1e-9, np.abs(mid), np.nan),
    )
    out["option_price"] = np.asarray(mid, dtype=np.float64)

    if realized_vol is not None:
        hv = np.asarray(realized_vol, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vol_deviation"] = np.log(
                np.where(hv > 1e-9, hv, np.nan) / np.where(iv > 1e-9, iv, np.nan)
            )
    if idio_vol is not None:
        out["ivol"] = np.asarray(idio_vol, dtype=np.float64)

    # Trailing volatility of implied volatility (causal window).
    T, K = iv.shape
    vov = np.full((T, K), np.nan, dtype=np.float64)
    w = max(5, int(vol_of_vol_window))
    for t in range(w, T):
        win = iv[t - w : t]
        with np.errstate(invalid="ignore"):
            vov[t] = np.nanstd(win, axis=0)
    out["vol_of_vol"] = vov
    return out


def build_option_factors(
    labels: np.ndarray,
    characteristics: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    HVX-style proxy factor set.

    Factor 1 is the equally weighted delta-hedged option portfolio (the option
    market factor). Factors 2 and 3 are long-short sorts on HV minus IV and on
    volatility of volatility. The cash-holdings leg of the published four-factor
    model is omitted for lack of Compustat data and is reported as such.
    """
    R = np.asarray(labels, dtype=np.float64)
    T = R.shape[0]
    factors: dict[str, list[float]] = {"option_market_ew": _lagged_ew_market(R)}
    built: dict[str, Any] = {}

    for fname, cname in (
        ("hv_minus_iv", "vol_deviation"),
        ("vol_of_vol", "vol_of_vol"),
    ):
        C = characteristics.get(cname)
        if C is None:
            continue
        sort = long_short_sort(C, R)
        factors[fname] = sort["pnl"]
        built[fname] = {
            "from_characteristic": cname,
            "sharpe": sort["sharpe"],
            "n_days": sort["n_days"],
        }

    return {
        "factors": factors,
        "diagnostics": built,
        "model": "HVX_proxy",
        "reference": "Horenstein, Vasquez and Xiao (2025, RFS)",
        "omitted_factors": dict(UNAVAILABLE_CHARACTERISTICS),
        "caveat": (
            "Proxy factor set restricted to data available in this lake. "
            "Omitted factors bias alpha upward, so a surviving alpha here is "
            "necessary but not sufficient evidence against Goyal and Saretto "
            "(2024), who find IPCA prices the cross-section of option returns."
        ),
    }


def factor_alpha(
    strategy_pnl: Sequence[float],
    factors: dict[str, Sequence[float]],
    *,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """
    OLS of strategy returns on factors, reporting alpha with a HAC t-statistic.

    Alpha is the intercept: the average return unexplained by the factor set.
    Standard errors use Newey-West because daily strategy returns are serially
    correlated (see :mod:`src.eval.stats_inference`).
    """
    y = np.asarray(list(strategy_pnl), dtype=np.float64)
    names: list[str] = []
    skipped: list[str] = []
    skipped_degenerate: list[str] = []
    for k in factors:
        v = np.asarray(list(factors[k]), dtype=np.float64)
        if v.size != y.size:
            skipped.append(k)
            continue
        if int(np.isfinite(v).sum()) < 30:
            skipped_degenerate.append(k)
            continue
        names.append(k)
    if y.size < 30:
        return {
            "ok": False,
            "reason": "need >= 30 aligned observations",
            "n": int(y.size),
        }
    X_cols = [np.asarray(list(factors[k]), dtype=np.float64) for k in names]
    if X_cols:
        X = np.column_stack([np.ones(y.size)] + X_cols)
    else:
        X = np.ones((y.size, 1))
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(good.sum()) < 30:
        return {
            "ok": False,
            "reason": "fewer than 30 jointly finite observations",
            "n": int(good.sum()),
            "factors_skipped_misaligned": skipped,
            "factors_skipped_degenerate": skipped_degenerate,
        }
    Xg, yg = X[good], y[good]
    beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
    resid = yg - Xg @ beta
    # HAC inference on the intercept via the residual-augmented series: adding
    # the residuals back to the intercept gives the alpha series whose mean is
    # the OLS intercept, so the Newey-West machinery applies directly.
    alpha_series = resid + beta[0]
    hac = hac_mean_tstat(alpha_series, lags=hac_lags)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(yg - yg.mean(), yg - yg.mean()))
    return {
        "ok": True,
        "alpha_daily": float(beta[0]),
        "alpha_annualized": float(beta[0] * ANNUALIZATION),
        "alpha_t_hac": hac.get("t_hac"),
        "alpha_se_hac": hac.get("se_hac"),
        "alpha_t_iid": hac.get("t_iid"),
        "hac_lags": hac.get("lags"),
        "betas": {k: float(b) for k, b in zip(names, beta[1:])},
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(good.sum()),
        "factors_used": names,
        "factors_skipped_misaligned": skipped,
        "factors_skipped_degenerate": skipped_degenerate,
        "alpha_significant_05": bool(
            hac.get("t_hac") is not None
            and np.isfinite(hac.get("t_hac", float("nan")))
            and abs(float(hac["t_hac"])) > 1.96
            and float(beta[0]) > 0
        ),
        "citation": "Newey and West (1987); Horenstein, Vasquez and Xiao (2025)",
    }


def attach_factor_alpha(
    report: dict[str, Any],
    *,
    strategy_pnl: Sequence[float],
    labels: np.ndarray,
    characteristics: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build the proxy factors, regress, and write the result into ``report``."""
    fac = build_option_factors(labels, characteristics)
    alpha = factor_alpha(strategy_pnl, fac["factors"])
    sorts = {}
    for cname, C in characteristics.items():
        try:
            sorts[cname] = {
                k: v for k, v in long_short_sort(C, labels).items() if k != "pnl"
            }
        except Exception as exc:
            sorts[cname] = {"error": str(exc)}
    out = {
        "factor_model": fac["model"],
        "reference": fac["reference"],
        "alpha": alpha,
        "factor_diagnostics": fac["diagnostics"],
        "characteristic_sorts": sorts,
        "omitted_factors": fac["omitted_factors"],
        "caveat": fac["caveat"],
    }
    report["factor_alpha"] = out
    return out


def hlz_hurdles(tstat: float | None) -> dict[str, Any]:
    """Harvey–Liu–Zhu (2016) multiple-testing hurdles for a new factor t-stat."""
    t = float("nan") if tstat is None else float(tstat)
    abs_t = abs(t) if np.isfinite(t) else float("nan")
    return {
        "t_stat": t,
        "clears_t_3_0": bool(np.isfinite(abs_t) and abs_t > 3.0),
        "clears_t_3_9": bool(np.isfinite(abs_t) and abs_t > 3.9),
        "clears_conventional_1_96": bool(np.isfinite(abs_t) and abs_t > 1.96),
        "citation": "Harvey, Liu and Zhu (2016, RFS 29(1))",
    }


def bh_fdr(
    pvalues: Sequence[float],
    *,
    q: float = 0.05,
) -> dict[str, Any]:
    """Benjamini–Hochberg FDR control across a family of tests."""
    p = np.asarray(list(pvalues), dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return {"ok": False, "reason": "empty pvalue list", "reject": []}
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if not np.any(below):
        cutoff = -1
    else:
        cutoff = int(np.max(np.where(below)[0]))
    reject = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        reject[order[: cutoff + 1]] = True
    return {
        "ok": True,
        "q": float(q),
        "m": m,
        "reject": reject.tolist(),
        "pvalues": p.tolist(),
        "bh_critical": thresh.tolist(),
    }


def build_hvx_factors(
    labels: np.ndarray,
    characteristics: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Alias to the implemented HVX-style proxy factor set."""
    return build_option_factors(labels, characteristics)


def build_bcsz_factors(
    labels: np.ndarray,
    characteristics: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    BCSZ-style quintile factors from feasible characteristics.

    Uses spread, price, and (when present) model-free kurtosis / RV−IV.
    Missing legs are recorded under omitted_factors rather than invented.
    """
    R = np.asarray(labels, dtype=np.float64)
    T = R.shape[0]
    factors: dict[str, list[float]] = {
        "option_market_oi_proxy_ew": _lagged_ew_market(R)
    }
    diagnostics: dict[str, Any] = {}
    omitted: dict[str, str] = {}
    mapping = (
        ("bcsz_spread", "option_spread"),
        ("bcsz_price", "option_price"),
        ("bcsz_mfik", "model_free_implied_kurtosis"),
        ("bcsz_rv_minus_iv", "vol_deviation"),
    )
    for fname, cname in mapping:
        C = characteristics.get(cname)
        if C is None:
            omitted[fname] = f"characteristic '{cname}' unavailable"
            continue
        sort = long_short_sort(C, R)
        factors[fname] = sort["pnl"]
        diagnostics[fname] = {
            "from_characteristic": cname,
            "sharpe": sort["sharpe"],
            "n_days": sort["n_days"],
        }
    return {
        "factors": factors,
        "diagnostics": diagnostics,
        "model": "BCSZ_feasible",
        "reference": "Bali, Cao, Song and Zhan (2022)",
        "omitted_factors": omitted,
    }


def build_ff_plus_factors(
    ff_panel: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """
    Wrap externally downloaded FF5 + Mom (+ optional BAB/straddle) factor series.

    Construction of BAB / Coval–Shumway straddles lives in the factor-data
    pipeline; this helper only packages aligned series for ``factor_alpha``.
    """
    if not ff_panel:
        return {
            "factors": {},
            "model": "FF_plus",
            "reference": "Büchner and Kelly (2022, JFE); Ken French library",
            "ok": False,
            "reason": "ff_panel empty — run scripts/download_factor_data.py",
        }
    factors = {k: list(v) for k, v in ff_panel.items()}
    omitted = {}
    for missing in ("BAB", "CS_straddle", "zero_beta_straddle"):
        if missing not in factors and not any(
            missing.lower() in str(k).lower() for k in factors
        ):
            omitted[missing] = (
                "not constructed in this lake pass — disclose as omitted FF+ "
                "leg (Büchner–Kelly BAB / Coval–Shumway straddle)"
            )
    return {
        "factors": factors,
        "model": "FF_plus",
        "reference": "Büchner and Kelly (2022, JFE); Ken French library",
        "ok": True,
        "omitted_factors": omitted,
    }


def run_ipca_panel(
    characteristics_panel: np.ndarray,
    dh_returns: np.ndarray,
    *,
    n_factors: int = 4,
    indices: np.ndarray | None = None,
    wald_ndraws: int = 200,
) -> dict[str, Any]:
    """
    Fit InstrumentedPCA on a broad DH return panel (Test CS).

    ``characteristics_panel`` is ``(N_obs, L)`` with matching ``dh_returns``
    ``(N_obs,)``. When ``indices`` is omitted, a dense rectangular panel index
    is assumed only if ``X`` reshape is unambiguous; prefer passing
    ``indices`` with columns ``(entity_id, time_id)``.

    After fit, runs ``BS_Walpha`` (bootstrap H0: Γα = 0) when the package
    supports it; ``wald_ndraws`` controls bootstrap size.
    """
    try:
        from ipca import InstrumentedPCA
    except ImportError as exc:
        return {"ok": False, "reason": f"ipca unavailable: {exc}"}
    X = np.asarray(characteristics_panel, dtype=np.float64)
    y = np.asarray(dh_returns, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        return {"ok": False, "reason": f"X must be 2-D (N_obs, L), got {X.shape}"}
    if y.shape[0] != X.shape[0]:
        return {
            "ok": False,
            "reason": f"y length {y.shape[0]} != X rows {X.shape[0]}",
        }
    idx = indices
    if idx is None:
        # Fallback: treat rows as a single long panel with unique (i,t) ids.
        idx = np.column_stack(
            [
                np.arange(X.shape[0], dtype=np.int64),
                np.zeros(X.shape[0], dtype=np.int64),
            ]
        )
    else:
        idx = np.asarray(indices)
        if idx.ndim != 2 or idx.shape[1] != 2 or idx.shape[0] != X.shape[0]:
            return {
                "ok": False,
                "reason": (
                    f"indices must be (N_obs, 2), got {getattr(idx, 'shape', None)}"
                ),
            }
    try:
        regr = InstrumentedPCA(n_factors=int(n_factors), intercept=True)
        regr = regr.fit(X=X, y=y, indices=idx)
        try:
            Gamma, Factors = regr.get_factors(label_ind=True)
        except Exception:
            Gamma = getattr(regr, "Gamma", None)
            Factors = getattr(regr, "Factors", None)
        if Gamma is None or Factors is None:
            return {
                "ok": False,
                "reason": "fit completed but Gamma/Factors unavailable",
                "model": "IPCA",
            }
        wald: dict[str, Any] = {
            "h0_statement": "Gamma_alpha = 0",
            "method": "InstrumentedPCA.BS_Walpha",
            "citation": "Kelly, Pruitt and Su (2019, JFE)",
        }
        try:
            # Bootstrap H0: Γα = 0. Keep draws modest for production runs;
            # callers can re-fit with larger ndraws for the appendix table.
            pval = float(regr.BS_Walpha(ndraws=int(wald_ndraws), n_jobs=1))
            wald.update(
                {
                    "ok": True,
                    "pvalue": pval,
                    "ndraws": int(wald_ndraws),
                    "reject_05": bool(np.isfinite(pval) and pval < 0.05),
                }
            )
        except Exception as exc:
            wald.update({"ok": False, "reason": str(exc)})
        return {
            "ok": True,
            "n_factors": int(n_factors),
            "Gamma_shape": list(np.asarray(Gamma).shape),
            "Factors_shape": list(np.asarray(Factors).shape),
            "n_obs": int(X.shape[0]),
            "model": "IPCA",
            "reference": "Kelly, Pruitt and Su (2019, JFE); Goyal and Saretto (2024)",
            "gamma_alpha_wald": wald,
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "model": "IPCA"}


def attach_factor_alpha_suite(
    report: dict[str, Any],
    *,
    strategy_pnl: Sequence[float],
    labels: np.ndarray,
    characteristics: dict[str, np.ndarray],
    factor_bundles: dict[str, dict[str, Sequence[float]]] | None = None,
    ipca_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Time-series alphas vs HVX / BCSZ / optional FF+ bundles + HLZ / FDR.

    ``factor_bundles`` maps model name → {factor_name: series}. IPCA cross-section
    results are attached separately when provided (Test CS).
    """
    bundles = dict(factor_bundles or {})
    if "HVX" not in bundles:
        bundles["HVX"] = build_hvx_factors(labels, characteristics)["factors"]
    if "BCSZ" not in bundles:
        bundles["BCSZ"] = build_bcsz_factors(labels, characteristics)["factors"]

    models: dict[str, Any] = {}
    pvals: list[float] = []
    for name, facs in bundles.items():
        alpha = factor_alpha(strategy_pnl, facs)
        t_hac = alpha.get("alpha_t_hac") if alpha.get("ok") else None
        models[name] = {
            "alpha": alpha,
            "hlz": hlz_hurdles(t_hac if isinstance(t_hac, (int, float)) else None),
        }
        if alpha.get("ok") and isinstance(t_hac, (int, float)) and np.isfinite(t_hac):
            # Two-sided normal p from HAC t (for FDR family size).
            from math import erfc, sqrt

            pvals.append(float(erfc(abs(float(t_hac)) / sqrt(2.0))))

    suite = {
        "models": models,
        "fdr": bh_fdr(pvals) if pvals else {"ok": False, "reason": "no pvalues"},
        "ipca_cs": ipca_result or {"ok": False, "reason": "not_run"},
        "note": (
            "Test TS = strategy PnL on factor portfolios; Test CS = IPCA on the "
            "broad DH panel. Condition 6 requires both."
        ),
    }
    report["factor_alpha_suite"] = suite
    return suite
