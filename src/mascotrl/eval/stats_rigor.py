"""Publication statistical rigor: PSR, DSR (Bailey & López de Prado 2014), regimes."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


EULER_MASCHERONI = 0.5772156649015329


def annualized_sharpe(returns: np.ndarray, *, periods: float | int = 252) -> float:
    """Annualize mean/std by ``sqrt(periods)``.

    ``periods`` may be a float when derived from a punctured date grid via
    ``periods_per_year_from_dates`` (do not silently force 252).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=0)) + 1e-12
    return float(mu / sd * np.sqrt(float(periods)))


def pack_return_summary(
    xs: list[float] | np.ndarray,
    *,
    periods: float | int = 252,
) -> dict[str, float]:
    """Shared mean / std / Sharpe / hit-rate packing (eps zero-vol policy)."""
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_pnl": float("nan"),
            "std_pnl": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "n_days": 0,
            "pnl_sum": 0.0,
        }
    finite = np.isfinite(arr)
    use = arr[finite] if finite.any() else arr
    if use.size == 0:
        return {
            "mean_pnl": float("nan"),
            "std_pnl": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "n_days": int(arr.size),
            "pnl_sum": float("nan"),
        }
    mu = float(use.mean())
    sd = float(use.std(ddof=0))
    return {
        "mean_pnl": mu,
        "std_pnl": sd,
        "sharpe": annualized_sharpe(use, periods=periods),
        "hit_rate": float((use > 0).mean()),
        "n_days": int(arr.size),
        "pnl_sum": float(np.nansum(arr)),
    }


def non_annualized_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=0)) + 1e-12
    return float(mu / sd)


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def expected_max_sharpe(
    n_trials: int,
    sr_variance: float,
    *,
    sr_mean: float = 0.0,
) -> float:
    """
    E[max SR] under null / selection (Bailey & López de Prado 2014).

    SR_0 = E[SR] + √V · ((1-γ) Z^{-1}(1-1/N) + γ Z^{-1}(1-1/(N e)))
    """
    n = max(int(n_trials), 1)
    v = max(float(sr_variance), 1e-12)
    if n == 1:
        return float(sr_mean)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sr_mean + np.sqrt(v) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """
    PSR(SR*) = Φ((SR* − SR_b) √(T−1) / √(1 − γ₃ SR* + ((γ₄−1)/4) SR*²)).

    ``kurtosis`` is Pearson kurtosis (normal = 3), not excess.
    ``observed_sr`` must match the sampling frequency of ``n_obs`` (non-annualized).
    """
    t = int(n_obs)
    if t < 3 or not np.isfinite(observed_sr):
        return float("nan")
    denom = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr**2)
    denom = max(float(denom), 1e-12)
    z = (observed_sr - benchmark_sr) * np.sqrt(t - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray | list[float],
    *,
    n_trials: int,
    trial_sharpes: list[float] | np.ndarray | None = None,
    periods_per_year: float | int = 252,
    n_trials_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, JPM 2014).

    Uses non-annualized SR inside PSR/DSR (correct for the sampling formula),
    and reports annualized Sharpe for human-readable tables.
    Skew/kurtosis are computed on the **raw** daily PnL series (not smoothed).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    t = int(r.size)
    if t < 3:
        return {
            "n_obs": t,
            "sharpe_ann": float("nan"),
            "psr": float("nan"),
            "dsr": float("nan"),
            "sr0_nonann": float("nan"),
            "n_trials": int(n_trials),
            "n_trials_breakdown": n_trials_breakdown,
            "significant_05": False,
        }
    sr = non_annualized_sharpe(r)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # Pearson
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        ts = np.asarray(trial_sharpes, dtype=np.float64)
        ts = ts[np.isfinite(ts)]
        # trial_sharpes are assumed annualized → convert to non-ann for SR0 scale
        ts_na = ts / np.sqrt(periods_per_year)
        sr_var = float(np.var(ts_na, ddof=0)) if ts_na.size else 1.0 / periods_per_year
        n_eff = max(int(n_trials), int(ts_na.size))
    else:
        # Conservative null: unit variance of non-ann SR estimates ≈ 1/T
        sr_var = 1.0 / max(t, 1)
        n_eff = max(int(n_trials), 1)
    sr0 = expected_max_sharpe(n_eff, sr_var, sr_mean=0.0)
    psr = probabilistic_sharpe_ratio(sr, 0.0, t, skew, kurt)
    dsr = probabilistic_sharpe_ratio(sr, sr0, t, skew, kurt)
    return {
        "n_obs": t,
        "sharpe_nonann": sr,
        "sharpe_ann": float(sr * np.sqrt(periods_per_year)),
        "skew": skew,
        "kurtosis_pearson": kurt,
        "moments_source": "raw_daily_pnl",
        "n_trials": int(n_eff),
        "n_trials_breakdown": n_trials_breakdown,
        "sr_variance_nonann": sr_var,
        "sr0_nonann": sr0,
        "sr0_ann": float(sr0 * np.sqrt(periods_per_year)),
        "psr": psr,
        "dsr": dsr,
        "significant_05": bool(np.isfinite(dsr) and dsr >= 0.95),
        "citation": "Bailey & López de Prado (2014), JPM — Deflated Sharpe Ratio",
    }


# Stress windows used for publication regime tables (inclusive calendar dates).
# Extended for Phase F robustness: pre-GFC, GFC, 2010s, COVID, 2022 hike.
DEFAULT_REGIMES: list[dict[str, str]] = [
    {
        "id": "pre_gfc",
        "label": "Pre-GFC",
        "start": "2005-01-01",
        "end": "2007-06-30",
    },
    {"id": "gfc_2008", "label": "2008 GFC", "start": "2008-09-01", "end": "2009-03-31"},
    {
        "id": "decade_2010s",
        "label": "2010s",
        "start": "2010-01-01",
        "end": "2019-12-31",
    },
    {
        "id": "covid_2020",
        "label": "2020 COVID",
        "start": "2020-02-15",
        "end": "2020-04-30",
    },
    {
        "id": "hike_2022",
        "label": "2022 rate-hike vol",
        "start": "2022-01-01",
        "end": "2022-12-31",
    },
]


def regime_performance_table(
    dates: list[str],
    pnls: list[float],
    turnovers: list[float] | None = None,
    *,
    regimes: list[dict[str, str]] | None = None,
    expect_stress_degradation: bool = True,
) -> dict[str, Any]:
    """Slice daily PnL by historical stress windows; report Sharpe / MDD / turnover.

    Missing windows are explicit ``status="unavailable"`` rows (never zero-filled
    Sharpes that could be misread as flat GFC performance).
    """
    regimes = regimes or DEFAULT_REGIMES
    if not dates or not pnls:
        empty_rows = [
            {
                "id": reg["id"],
                "label": reg["label"],
                "start": reg["start"],
                "end": reg["end"],
                "n_days": 0,
                "available": False,
                "status": "unavailable",
                "note": "N/A — no evaluation PnL series",
                "mean_pnl": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "mean_turnover": float("nan"),
                "hit_rate": float("nan"),
                "pnl_sum": float("nan"),
            }
            for reg in regimes
        ]
        return {
            "regimes": empty_rows,
            "n_dates": 0,
            "sanity": {
                "warnings": ["no PnL series for regime table"],
                "full_sample_sharpe": float("nan"),
                "expect_stress_degradation": bool(expect_stress_degradation),
            },
        }
    if len(dates) != len(pnls):
        return {
            "regimes": [
                {
                    "id": reg["id"],
                    "label": reg["label"],
                    "start": reg["start"],
                    "end": reg["end"],
                    "n_days": 0,
                    "available": False,
                    "status": "unavailable",
                    "note": (
                        f"N/A — date/PnL length mismatch "
                        f"({len(dates)} dates vs {len(pnls)} pnls)"
                    ),
                    "mean_pnl": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "mean_turnover": float("nan"),
                    "hit_rate": float("nan"),
                    "pnl_sum": float("nan"),
                }
                for reg in regimes
            ],
            "n_dates": len(dates),
            "sanity": {
                "warnings": [
                    f"date/PnL length mismatch: {len(dates)} vs {len(pnls)}"
                ],
                "full_sample_sharpe": float("nan"),
                "expect_stress_degradation": bool(expect_stress_degradation),
            },
        }
    # Normalize to comparable YYYY-MM-DD strings for window masks.
    idx = np.asarray([str(d)[:10] for d in dates])
    r = np.asarray(pnls, dtype=np.float64)
    turns = (
        np.asarray(turnovers, dtype=np.float64)
        if turnovers is not None and len(turnovers) == len(pnls)
        else np.full(len(pnls), np.nan)
    )
    panel_start = str(min(idx.tolist())) if idx.size else None
    panel_end = str(max(idx.tolist())) if idx.size else None
    rows = []
    for reg in regimes:
        mask = (idx >= reg["start"]) & (idx <= reg["end"])
        n_days = int(mask.sum())
        sub = r[mask]
        tu = turns[mask]
        if n_days == 0:
            note = (
                f"N/A — data unavailable (panel covers {panel_start}→{panel_end}; "
                f"window {reg['start']}→{reg['end']} has no overlap)"
            )
            rows.append(
                {
                    "id": reg["id"],
                    "label": reg["label"],
                    "start": reg["start"],
                    "end": reg["end"],
                    "n_days": 0,
                    "available": False,
                    "status": "unavailable",
                    "note": note,
                    "mean_pnl": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "mean_turnover": float("nan"),
                    "hit_rate": float("nan"),
                    "pnl_sum": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "id": reg["id"],
                "label": reg["label"],
                "start": reg["start"],
                "end": reg["end"],
                "n_days": n_days,
                "available": True,
                "status": "ok",
                "note": None,
                "mean_pnl": float(sub.mean()),
                "sharpe": annualized_sharpe(sub) if sub.size >= 2 else float("nan"),
                "max_drawdown": max_drawdown(sub),
                "mean_turnover": float(np.nanmean(tu)) if tu.size else float("nan"),
                "hit_rate": float((sub > 0).mean()),
                "pnl_sum": float(sub.sum()),
            }
        )
    # Full-sample Sharpe for degradation sanity (short-vol flavored strategies).
    full_sh = annualized_sharpe(r) if r.size >= 2 else float("nan")
    sanity = {
        "full_sample_sharpe": full_sh,
        "expect_stress_degradation": bool(expect_stress_degradation),
        "panel_start": panel_start,
        "panel_end": panel_end,
        "warnings": [],
    }
    for row in rows:
        if not row.get("available", True):
            sanity["warnings"].append(f"{row['id']}: {row.get('note')}")
    if expect_stress_degradation and np.isfinite(full_sh):
        for row in rows:
            if not row.get("available") or row["n_days"] < 5 or not np.isfinite(row["sharpe"]):
                continue
            # Crash windows looking *better* than full sample → leakage red flag.
            if row["id"] in ("gfc_2008", "covid_2020") and row["sharpe"] > full_sh + 0.5:
                sanity["warnings"].append(
                    f"{row['id']}: stress Sharpe {row['sharpe']:.3f} > full "
                    f"{full_sh:.3f} + 0.5 — inspect date alignment / leakage"
                )
    return {"regimes": rows, "n_dates": len(dates), "sanity": sanity}


def stationary_bootstrap_indices(
    n: int,
    *,
    block_mean: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index path of length n."""
    rng = rng or np.random.default_rng()
    p = 1.0 / max(int(block_mean), 1)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = int(rng.integers(0, n))
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def block_bootstrap_metric_ci(
    returns: np.ndarray | list[float],
    *,
    metric: str = "sharpe",
    n_boot: int = 499,
    block_mean: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
    periods: int = 252,
    backend: str = "custom",
) -> dict[str, Any]:
    """
    Block-bootstrap CI for Sharpe / mean / max_drawdown / mean_turnover proxy.

    Uses stationary bootstrap (not iid) — appropriate for fat-tailed, autocorrelated
    vol P&L. ``metric='turnover'`` expects ``returns`` to actually be turnovers.
    ``backend='arch'`` dispatches to ``arch_bootstrap.block_bootstrap_metric_ci_arch``.
    """
    backend_key = str(backend or "custom").lower().strip()
    if backend_key == "arch":
        from src.eval.arch_bootstrap import block_bootstrap_metric_ci_arch

        out = block_bootstrap_metric_ci_arch(
            returns,
            metric=metric,
            n_boot=n_boot,
            block_mean=block_mean,
            alpha=alpha,
            seed=seed,
            periods=periods,
        )
        out["backend"] = "arch"
        return out

    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return {
            "metric": metric,
            "n_obs": int(r.size),
            "point": float("nan"),
            "ci": None,
            "backend": "custom",
        }
    rng = np.random.default_rng(int(seed))

    def _compute(x: np.ndarray) -> float:
        if metric == "sharpe":
            return annualized_sharpe(x, periods=periods)
        if metric == "mean":
            return float(x.mean())
        if metric == "max_drawdown":
            return max_drawdown(x)
        if metric == "mean_abs":
            return float(np.mean(np.abs(x)))
        raise KeyError(metric)

    point = _compute(r)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = stationary_bootstrap_indices(r.size, block_mean=block_mean, rng=rng)
        boots[b] = _compute(r[idx])
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {
        "metric": metric,
        "n_obs": int(r.size),
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "alpha": float(alpha),
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "boot_mean": float(boots.mean()),
        "boot_std": float(boots.std(ddof=0)),
        "backend": "custom",
    }


def wilcoxon_paired_delta(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Wilcoxon signed-rank on paired fold/seed Sharpes (a − b)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "n_pairs": int(x.size),
            "mean_delta": float("nan"),
            "pvalue": float("nan"),
            "statistic": float("nan"),
        }
    d = x - y
    # zero_method='wilcox'; scipy ≥1. something — use prune zeros
    d = d[d != 0]
    if d.size < 3:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "n_pairs": int(x.size),
            "mean_delta": float((x - y).mean()),
            "pvalue": float("nan"),
            "statistic": float("nan"),
            "note": "too many zero deltas",
        }
    stat, p = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": int(x.size),
        "mean_delta": float((x - y).mean()),
        "median_delta": float(np.median(x - y)),
        "statistic": float(stat),
        "pvalue": float(p),
        "significant_05": bool(p < 0.05),
    }


def hansen_spa_test(
    benchmark_pnls: np.ndarray | list[float],
    rival_pnls: dict[str, list[float] | np.ndarray],
    *,
    n_boot: int = 499,
    block_mean: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Hansen (2005) SPA / White Reality Check style test on negative PnL as loss.

    Null: no rival has lower expected loss than the benchmark (HAPPO).
    Loss_t = −pnl_t. Differential d_{k,t} = loss_bench − loss_k = pnl_k − pnl_bench.
    Studentized max statistic with stationary bootstrap under recentered null.
    Returns consistent / lower / upper p-value bounds (Hansen SPA manual).
    """
    bench = np.asarray(benchmark_pnls, dtype=np.float64)
    names = sorted(rival_pnls.keys())
    if not names or bench.size < 20:
        return {"ok": False, "reason": "insufficient data", "n_obs": int(bench.size)}
    mats = []
    used = []
    for name in names:
        r = np.asarray(rival_pnls[name], dtype=np.float64)
        n = min(bench.size, r.size)
        if n < 20:
            continue
        mats.append(r[:n] - bench[:n])  # pnl_rival − pnl_bench = loss_b − loss_r
        used.append(name)
    if not mats:
        return {"ok": False, "reason": "no rival series", "n_obs": int(bench.size)}
    D = np.column_stack(mats)  # (T, K)
    t, k = D.shape
    d_bar = D.mean(axis=0)
    # Variance via stationary bootstrap of demeaned series
    rng = np.random.default_rng(int(seed))
    boot_means = np.empty((int(n_boot), k), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = stationary_bootstrap_indices(t, block_mean=block_mean, rng=rng)
        boot_means[b] = D[idx].mean(axis=0)
    # σ̂_k from bootstrap
    sig = boot_means.std(axis=0, ddof=0) + 1e-12
    t_stat = np.max(np.sqrt(t) * d_bar / sig)

    # Recentered nulls (Hansen): lower / consistent / upper
    def _p_from_center(center: np.ndarray) -> float:
        # bootstrap max of studentized (boot_mean − center)
        centered = boot_means - center
        boot_t = np.max(np.sqrt(t) * centered / sig, axis=1)
        return float(np.mean(boot_t > t_stat))

    # Lower: center at max(d_bar, 0) for each (liberal)
    center_l = np.minimum(d_bar, 0.0)
    # Upper: center at 0 (conservative — all rivals as good as bench under null)
    center_u = np.zeros(k)
    # Consistent: discard clearly inferior rivals (d_bar < -sqrt(var log T / T))
    thresh = -np.sqrt(sig**2 * np.log(max(t, 2)) / t)
    center_c = np.where(d_bar >= thresh, 0.0, d_bar)

    return {
        "ok": True,
        "n_obs": int(t),
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "rivals": used,
        "mean_pnl_diff_vs_bench": {n: float(d_bar[i]) for i, n in enumerate(used)},
        "t_spa": float(t_stat),
        "pvalue_lower": _p_from_center(center_l),
        "pvalue_consistent": _p_from_center(center_c),
        "pvalue_upper": _p_from_center(center_u),
        "citation": "Hansen (2005) SPA; White (2000) Reality Check (stationary bootstrap)",
        "note": (
            "Loss = −PnL. Low pvalue_consistent rejects null that benchmark is not "
            "inferior to the best rival (i.e. some rival beats HAPPO)."
        ),
    }
