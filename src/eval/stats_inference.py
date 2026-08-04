"""
Publication-grade inference: HAC standard errors, Romano-Wolf stepdown, and
CSCV on reconstructed return paths.

Three gaps this closes relative to the previous stack:

  * **HAC**: daily strategy P&L is serially correlated (overlapping hedges,
    persistent positions), so an iid t-statistic overstates significance.
    Newey and West (1987) with the Newey-West (1994) automatic lag rule is the
    standard correction, and finance referees ask for it by name.
  * **Romano-Wolf**: Hansen's SPA answers only whether the *single best* rival
    beats the benchmark. Romano and Wolf (2005, Econometrica 73(4)) stepdown
    controls the familywise error rate across the whole family while exploiting
    the dependence structure, so it identifies *which* comparisons survive.
  * **CSCV on paths**: the previous PBO ran on a vector of trial Sharpes, which
    is a proxy. With CPCV reconstructed paths available, the combinatorially
    symmetric cross-validation of Bailey, Borwein, Lopez de Prado and Zhu can
    be run on the return paths it was defined for.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np

from src.logging_utils import get_logger

log = get_logger("mascotrl.eval.stats_inference")

ANNUALIZATION = 252.0


# ------------------------------------------------------------------------- HAC

def newey_west_lag(n: int) -> int:
    """
    Automatic bandwidth: floor(4 * (n/100)^(2/9)).

    The Newey-West (1994) plug-in rule, which is the convention in the empirical
    asset-pricing tables this work will be compared against.
    """
    if n <= 1:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean_tstat(
    x: Sequence[float],
    *,
    lags: int | None = None,
) -> dict[str, Any]:
    """
    Mean, Newey-West HAC standard error, and t-statistic for a return series.

    Uses the Bartlett kernel with weights ``1 - j/(L+1)``. Reports the iid
    standard error alongside so the inflation from serial correlation is visible
    rather than hidden.
    """
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return {
            "n": n,
            "mean": float("nan"),
            "se_iid": float("nan"),
            "se_hac": float("nan"),
            "t_iid": float("nan"),
            "t_hac": float("nan"),
            "lags": 0,
            "reason": "need >= 3 observations",
        }
    L = int(newey_west_lag(n) if lags is None else max(0, int(lags)))
    mu = float(arr.mean())
    e = arr - mu
    # Long-run variance: gamma_0 + 2 * sum_j w_j * gamma_j
    gamma0 = float(np.dot(e, e) / n)
    lrv = gamma0
    for j in range(1, L + 1):
        if j >= n:
            break
        gamma_j = float(np.dot(e[j:], e[:-j]) / n)
        w = 1.0 - j / (L + 1.0)
        lrv += 2.0 * w * gamma_j
    # A Bartlett kernel keeps this non-negative in theory; guard numerically.
    lrv = max(lrv, 1e-24)
    se_hac = float(np.sqrt(lrv / n))
    sd = float(arr.std(ddof=1))
    se_iid = float(sd / np.sqrt(n)) if sd > 0 else float("nan")
    return {
        "n": n,
        "mean": mu,
        "se_iid": se_iid,
        "se_hac": se_hac,
        "t_iid": float(mu / se_iid) if se_iid and np.isfinite(se_iid) else float("nan"),
        "t_hac": float(mu / se_hac) if se_hac > 0 else float("nan"),
        "lags": L,
        "lag_rule": "Newey-West (1994) floor(4*(n/100)^(2/9))",
        "kernel": "bartlett",
        "annualized_mean": mu * ANNUALIZATION,
        "citation": "Newey and West (1987, Econometrica 55(3))",
    }


def hac_sharpe_se(x: Sequence[float], *, lags: int | None = None) -> dict[str, Any]:
    """
    Annualized Sharpe with a HAC standard error.

    Uses the Lo (2002) style adjustment: the Sharpe standard error is scaled by
    the ratio of the HAC to the iid standard error of the mean, which captures
    the first-order effect of autocorrelation on Sharpe inference.
    """
    m = hac_mean_tstat(x, lags=lags)
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    if not np.isfinite(sd) or sd <= 0 or n < 3:
        return {**m, "sharpe_annual": float("nan"), "se_sharpe_hac": float("nan")}
    sr = float(arr.mean() / sd)
    sr_ann = sr * np.sqrt(ANNUALIZATION)
    inflation = (
        m["se_hac"] / m["se_iid"]
        if m.get("se_iid") and np.isfinite(m["se_iid"]) and m["se_iid"] > 0
        else 1.0
    )
    se_sr = float(np.sqrt((1.0 + 0.5 * sr**2) / n) * inflation * np.sqrt(ANNUALIZATION))
    return {
        **m,
        "sharpe_annual": sr_ann,
        "se_sharpe_hac": se_sr,
        "t_sharpe_hac": float(sr_ann / se_sr) if se_sr > 0 else float("nan"),
        "hac_inflation_factor": float(inflation),
        "citation_sharpe": "Lo (2002, FAJ 58(4))",
    }


# ---------------------------------------------------------- White Reality Check

def white_reality_check(
    benchmark_pnls: Sequence[float],
    rival_pnls: dict[str, Sequence[float]],
    *,
    n_boot: int = 499,
    block_mean: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """White (2000) Bootstrap Reality Check on negative PnL as loss.

    Null: no rival has lower expected loss than the benchmark. Loss_t = -pnl_t.
    Differential d_{k,t} = pnl_k - pnl_bench (positive means rival beats bench).
    Studentized max statistic with stationary bootstrap under the recentered null
    (each rival demeaned to zero). Less conservative than Hansen SPA; reports a
    single p-value for the best rival vs benchmark.
    """
    from src.eval.stats_rigor import stationary_bootstrap_indices

    bench = np.asarray(list(benchmark_pnls), dtype=np.float64)
    names = sorted(rival_pnls.keys())
    if not names or bench.size < 20:
        return {"ok": False, "reason": "insufficient data", "n_obs": int(bench.size)}
    mats = []
    used = []
    for name in names:
        r = np.asarray(list(rival_pnls[name]), dtype=np.float64)
        n = min(bench.size, r.size)
        if n < 20:
            continue
        mats.append(r[:n] - bench[:n])
        used.append(name)
    if not mats:
        return {"ok": False, "reason": "no rival series", "n_obs": int(bench.size)}
    D = np.column_stack(mats)
    t, k = D.shape
    d_bar = D.mean(axis=0)
    rng = np.random.default_rng(int(seed))
    boot_means = np.empty((int(n_boot), k), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = stationary_bootstrap_indices(t, block_mean=block_mean, rng=rng)
        boot_means[b] = D[idx].mean(axis=0)
    sig = boot_means.std(axis=0, ddof=0) + 1e-12
    t_stat = float(np.max(np.sqrt(t) * d_bar / sig))

    def _p_from_recentered(center: np.ndarray) -> float:
        centered = boot_means - center
        boot_t = np.max(np.sqrt(t) * centered / sig, axis=1)
        return float(np.mean(boot_t >= t_stat))

    pvalue = _p_from_recentered(np.zeros(k))
    best_idx = int(np.argmax(d_bar))
    return {
        "ok": True,
        "n_obs": int(t),
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "rivals": used,
        "best_rival": used[best_idx] if used else "",
        "mean_pnl_diff_vs_bench": {n: float(d_bar[i]) for i, n in enumerate(used)},
        "t_rc": t_stat,
        "pvalue": pvalue,
        "citation": "White (2000) Bootstrap Reality Check",
        "note": (
            "Loss = -PnL. Low pvalue rejects null that benchmark is not inferior "
            "to the best rival."
        ),
    }


# ---------------------------------------------------------------- Romano-Wolf

def _studentized_diff(
    bench: np.ndarray, rival: np.ndarray
) -> tuple[float, float, float]:
    d = rival - bench
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return float("nan"), float("nan"), 0
    mu = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se <= 0:
        return float("nan"), float("nan"), n
    return mu, float(mu / se), n


def romano_wolf_stepdown(
    benchmark_pnl: Sequence[float],
    rivals: dict[str, Sequence[float]],
    *,
    n_boot: int = 999,
    block_mean: int = 5,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Stepwise multiple testing controlling the familywise error rate.

    Tests, for each rival, H_k: rival does not outperform the benchmark. Follows
    Romano and Wolf (2005): studentize the mean differences, bootstrap the joint
    distribution of the maximum statistic with a stationary block bootstrap to
    respect dependence, reject the largest statistic exceeding the critical
    value, then re-run on the survivors.

    More powerful than a single-step reality check because later steps use a
    critical value computed only over the remaining hypotheses.
    """
    bench = np.asarray(list(benchmark_pnl), dtype=np.float64)
    names = [k for k, v in rivals.items() if len(list(v)) == bench.size]
    skipped = [k for k in rivals if k not in names]
    if bench.size < 20 or not names:
        return {
            "protocol": "romano_wolf_stepdown",
            "citation": "Romano and Wolf (2005, Econometrica 73(4))",
            "reason": "need >=20 aligned observations and >=1 rival",
            "rejected": [],
            "results": [],
            "skipped_misaligned": skipped,
        }

    R = {k: np.asarray(list(rivals[k]), dtype=np.float64) for k in names}
    obs: dict[str, float] = {}
    for k in names:
        _, t, _ = _studentized_diff(bench, R[k])
        obs[k] = t if np.isfinite(t) else -np.inf

    rng = np.random.default_rng(int(seed))
    n = bench.size
    p_block = 1.0 / max(1, int(block_mean))

    def _boot_indices() -> np.ndarray:
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(0, n))
        for j in range(n):
            idx[j] = i
            if rng.random() < p_block:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
        return idx

    # Centered bootstrap distribution of each studentized difference.
    boot: dict[str, np.ndarray] = {k: np.empty(n_boot) for k in names}
    for b in range(int(n_boot)):
        idx = _boot_indices()
        bb = bench[idx]
        for k in names:
            rb = R[k][idx]
            d = rb - bb
            mu = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(n))
            centre = float((R[k] - bench).mean())
            boot[k][b] = (mu - centre) / se if se > 0 else 0.0

    remaining = sorted(names, key=lambda k: obs[k], reverse=True)
    rejected: list[str] = []
    results: list[dict[str, Any]] = []
    step = 0
    while remaining:
        step += 1
        mat = np.vstack([boot[k] for k in remaining])
        max_stat = mat.max(axis=0)
        crit = float(np.quantile(max_stat, 1.0 - float(alpha)))
        top = remaining[0]
        p_adj = float(np.mean(max_stat >= obs[top])) if np.isfinite(obs[top]) else 1.0
        if np.isfinite(obs[top]) and obs[top] > crit:
            mean_d, t, nn = _studentized_diff(bench, R[top])
            rejected.append(top)
            results.append(
                {
                    "rival": top,
                    "step": step,
                    "t_stat": obs[top],
                    "critical_value": crit,
                    "p_adjusted": p_adj,
                    "mean_diff": mean_d,
                    "n": nn,
                    "rejected": True,
                }
            )
            remaining = remaining[1:]
            continue
        # First failure to reject stops the stepdown; the rest inherit it.
        for k in remaining:
            mean_d, t, nn = _studentized_diff(bench, R[k])
            results.append(
                {
                    "rival": k,
                    "step": step,
                    "t_stat": obs[k],
                    "critical_value": crit,
                    "p_adjusted": float(np.mean(max_stat >= obs[k]))
                    if np.isfinite(obs[k])
                    else 1.0,
                    "mean_diff": mean_d,
                    "n": nn,
                    "rejected": False,
                }
            )
        break

    return {
        "protocol": "romano_wolf_stepdown",
        "citation": "Romano and Wolf (2005, Econometrica 73(4))",
        "h0_statement": "H_k: rival_k does not outperform the benchmark",
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "bootstrap": "stationary (Politis and Romano 1994)",
        "n_rivals": len(names),
        "rejected": rejected,
        "n_rejected": len(rejected),
        "results": results,
        "skipped_misaligned": skipped,
        "note": (
            "Controls familywise error across the rival family, unlike SPA which "
            "only tests whether the single best rival beats the benchmark."
        ),
    }


def romano_wolf_over_panel(
    happo_pnl: Sequence[float],
    panel: dict[str, Sequence[float]],
    *,
    n_boot: int = 999,
    block_mean: int = 5,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Romano–Wolf on path-level diffs ``HAPPO − bench_j`` for each panel member.

    Builds aligned difference series and calls :func:`romano_wolf_stepdown` with a
    zero benchmark so studentized means equal mean(HAPPO − bench_j). Rejecting
    member ``j`` means HAPPO significantly outperforms that benchmark under FWER.
    """
    happo = np.asarray(list(happo_pnl), dtype=np.float64)
    diffs: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for name, series in panel.items():
        if str(name) in ("happo", "happo_gross"):
            continue
        r = np.asarray(list(series), dtype=np.float64)
        n = min(happo.size, r.size)
        if n < 20:
            skipped.append(str(name))
            continue
        diffs[str(name)] = happo[:n] - r[:n]
    if not diffs:
        return {
            "protocol": "romano_wolf_over_panel",
            "ok": False,
            "reason": "no aligned panel diffs",
            "skipped": skipped,
            "diff_definition": "HAPPO - bench_j",
        }
    n = int(min(v.size for v in diffs.values()))
    zero = np.zeros(n, dtype=np.float64)
    rivals = {k: v[:n] for k, v in diffs.items()}
    out = romano_wolf_stepdown(
        zero,
        rivals,
        n_boot=int(n_boot),
        block_mean=int(block_mean),
        seed=int(seed),
        alpha=float(alpha),
    )
    out["protocol"] = "romano_wolf_over_panel"
    out["ok"] = True
    out["diff_definition"] = "HAPPO - bench_j"
    out["h0_statement"] = "H_k: HAPPO does not outperform panel member k"
    out["claimant"] = "happo"
    out["skipped_short"] = skipped
    return out


# ------------------------------------------------------- CSCV on return paths

def cscv_pbo_from_paths(
    paths: Sequence[Sequence[float]],
    *,
    n_partitions: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Probability of backtest overfitting via CSCV on reconstructed return paths.

    This is the procedure as defined: split the *time* axis into S even blocks,
    take every balanced combination of blocks as in-sample, rank the candidate
    paths in-sample, and record the out-of-sample rank of the in-sample winner.
    PBO is the frequency with which that winner lands below the OOS median.

    ``paths`` is a candidate-by-time matrix — for CPCV, the reconstructed
    backtest paths.
    """
    mat = [np.asarray(list(p), dtype=np.float64) for p in paths]
    if not mat:
        return {"pbo": float("nan"), "reason": "no paths", "n_paths": 0}
    T = min(int(m.size) for m in mat)
    P = len(mat)
    if P < 2 or T < 8:
        return {
            "pbo": float("nan"),
            "reason": "need >=2 paths and >=8 aligned observations",
            "n_paths": P,
            "n_obs": T,
        }
    X = np.vstack([m[:T] for m in mat])  # (P, T)

    S = 8 if T >= 32 else 4
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]
    half = S // 2
    combos = list(combinations(range(S), half))
    rng = np.random.default_rng(int(seed))
    if len(combos) > int(n_partitions):
        sel = rng.choice(len(combos), size=int(n_partitions), replace=False)
        combos = [combos[i] for i in sel]

    def _sr(v: np.ndarray) -> float:
        v = v[np.isfinite(v)]
        if v.size < 2:
            return float("nan")
        sd = float(v.std(ddof=0))
        return float(v.mean() / sd) if sd > 1e-15 else float("nan")

    logits: list[float] = []
    fails = 0
    used = 0
    for combo in combos:
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate(
            [blocks[i] for i in range(S) if i not in combo]
        )
        is_sr = np.array([_sr(X[p, is_idx]) for p in range(P)])
        oos_sr = np.array([_sr(X[p, oos_idx]) for p in range(P)])
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue
        best = int(np.nanargmax(is_sr))
        finite = np.isfinite(oos_sr)
        if finite.sum() < 2:
            continue
        # Relative rank of the IS winner in the OOS distribution.
        ranks = oos_sr[finite].argsort().argsort()
        order = np.where(finite)[0]
        pos = int(np.where(order == best)[0][0]) if best in order else None
        if pos is None:
            continue
        w = float(ranks[pos]) / float(finite.sum() - 1) if finite.sum() > 1 else 0.5
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))
        if w < 0.5:
            fails += 1
        used += 1

    if used == 0:
        return {"pbo": float("nan"), "reason": "no usable partitions", "n_paths": P}
    return {
        "protocol": "cscv_on_return_paths",
        "citation": (
            "Bailey, Borwein, Lopez de Prado and Zhu — The Probability of "
            "Backtest Overfitting; CSCV as defined on return paths"
        ),
        "pbo": float(fails / used),
        "median_logit": float(np.median(logits)) if logits else float("nan"),
        "n_paths": P,
        "n_obs": T,
        "n_blocks": S,
        "n_partitions_used": used,
        "is_proxy": False,
        "interpretation": (
            "PBO is the frequency with which the in-sample best path ranks below "
            "the out-of-sample median. Low PBO is necessary, not sufficient."
        ),
    }
