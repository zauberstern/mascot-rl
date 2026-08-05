"""Regime detection performance scorecard.

REGIME_SCORECARD_SPEC (frozen)
------------------------------
Detectors: label_regimes (3-state), turbulence_index+classify_regime (binary),
walk_forward_hmm_filter on turbulence (KPT secondary Jaccard). Metrics: prefix
hygiene, Jaccard turb↔filtered-HMM, duration/switch stats, DEFAULT_REGIMES
event alignment, cross-taxonomy Jaccard(macro crisis, turbulence). Embeds
wiring + macro leverage. No OOS tuning of thresholds; N=2 pre-registered.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.regime_labels import REGIME_IDS, label_regimes
from mascotrl._root import REPO_ROOT
from mascotrl.eval.stats_rigor import DEFAULT_REGIMES
from mascotrl.eval.turbulence import (
    chi2_turbulence_threshold,
    classify_regime,
    classify_regime_chi2,
    turbulence_index,
)
from mascotrl.eval.walk_forward_hmm import (
    jaccard_turbulent,
    walk_forward_hmm_filter,
    walk_forward_markov_filter,
)


def operational_markov_mask_from_returns(
    asset_returns: np.ndarray,
    *,
    turbulence_window: int = 252,
    hmm_window: int = 756,
    hmm_step: int = 21,
    macro_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean operational mask: filtered Markov hard==1 on daily turbulence."""
    r = np.asarray(asset_returns, dtype=np.float64)
    turb = turbulence_index(r, window=int(turbulence_window), macro_cols=macro_cols)
    filt = walk_forward_markov_filter(
        turb,
        window=int(hmm_window),
        step=int(hmm_step),
        k_regimes=2,
        growing=False,
    )
    hard = np.asarray(filt["hard"], dtype=np.int32)
    return hard == 1


def jaccard_sic_vs_gics_operational(
    sic_returns: np.ndarray,
    gics_returns: np.ndarray,
    *,
    turbulence_window: int = 252,
    hmm_window: int = 756,
    hmm_step: int = 21,
    macro_cols: np.ndarray | None = None,
) -> float:
    """Report-only Jaccard of operational Markov masks on SIC vs GICS panels."""
    a = operational_markov_mask_from_returns(
        sic_returns,
        turbulence_window=turbulence_window,
        hmm_window=hmm_window,
        hmm_step=hmm_step,
        macro_cols=macro_cols,
    )
    b = operational_markov_mask_from_returns(
        gics_returns,
        turbulence_window=turbulence_window,
        hmm_window=hmm_window,
        hmm_step=hmm_step,
        macro_cols=macro_cols,
    )
    return float(jaccard_turbulent(a, b))


def hygiene_prefix_stability(
    macro: pd.DataFrame,
    *,
    min_history_days: int = 756,
    persistence_days: int = 10,
    cut_frac: float = 0.55,
) -> dict[str, Any]:
    """Truncate mid-sample; require prefix labels identical (no look-ahead)."""
    labels_full, meta_full = label_regimes(
        macro,
        min_history_days=min_history_days,
        persistence_days=persistence_days,
    )
    n = len(macro)
    cut = max(int(min_history_days) + 10, int(n * float(cut_frac)))
    cut = min(cut, n - 1)
    labels_pref, _ = label_regimes(
        macro.iloc[:cut],
        min_history_days=min_history_days,
        persistence_days=persistence_days,
    )
    stable = labels_full.iloc[:cut].tolist() == labels_pref.tolist()
    return {
        "status": "pass" if stable else "fail",
        "macro_prefix_stable": bool(stable),
        "cut_index": int(cut),
        "n_warmup": int(meta_full["warmup"].sum()) if "warmup" in meta_full else None,
    }


def occupancy_stats(
    labels: pd.Series,
    meta: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Post-warmup occupancy fractions and switch dynamics."""
    if meta is not None and "warmup" in meta.columns:
        active = labels.loc[~meta["warmup"].to_numpy()]
    else:
        active = labels
    n = int(len(active))
    fractions: dict[str, float] = {}
    counts: dict[str, int] = {}
    for rid in REGIME_IDS:
        c = int((active.astype(str) == rid).sum())
        counts[rid] = c
        fractions[rid] = float(c / n) if n else float("nan")
    days_in = (
        meta.loc[~meta["warmup"].to_numpy(), "days_in_regime"]
        if meta is not None and "days_in_regime" in meta.columns and "warmup" in meta.columns
        else None
    )
    switches = (
        int(meta.loc[~meta["warmup"].to_numpy(), "switch_flag"].sum())
        if meta is not None and "switch_flag" in meta.columns and "warmup" in meta.columns
        else int((active.astype(str).to_numpy()[1:] != active.astype(str).to_numpy()[:-1]).sum())
        if n > 1
        else 0
    )
    return {
        "status": "ok",
        "n_active": n,
        "fractions": fractions,
        "n_calm": counts["calm"],
        "n_inflationary": counts["inflationary"],
        "n_crisis": counts["crisis"],
        "mean_days_in_regime": float(days_in.mean()) if days_in is not None and len(days_in) else float("nan"),
        "n_switches": switches,
        "switch_rate": float(switches / max(n, 1)),
    }


def run_duration_stats(mask: np.ndarray | Sequence[bool]) -> dict[str, Any]:
    """Mean duration of True runs, switch rate, occupancy."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(m.size)
    if n == 0:
        return {
            "status": "unavailable",
            "mean_turbulent_run_days": float("nan"),
            "n_turbulent_runs": 0,
            "turbulent_day_frac": float("nan"),
            "switch_rate": float("nan"),
        }
    switches = int((m[1:] != m[:-1]).sum()) if n > 1 else 0
    runs: list[int] = []
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        runs.append(j - i)
        i = j
    return {
        "status": "ok",
        "mean_turbulent_run_days": float(np.mean(runs)) if runs else 0.0,
        "n_turbulent_runs": int(len(runs)),
        "turbulent_day_frac": float(m.mean()),
        "switch_rate": float(switches / max(n, 1)),
    }


def event_alignment(
    dates: Sequence | pd.DatetimeIndex,
    labels: pd.Series | Sequence[str],
    turbulent_mask: np.ndarray | Sequence[bool],
    *,
    regimes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Hit rates of crisis / turbulent flags inside calendar stress windows."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    labs = pd.Series(list(labels), index=idx).astype(str)
    turb = np.asarray(turbulent_mask, dtype=bool).reshape(-1)
    if turb.size != len(idx):
        raise ValueError("turbulent_mask length must match dates")
    windows_out: list[dict[str, Any]] = []
    for reg in regimes or DEFAULT_REGIMES:
        start = pd.Timestamp(reg["start"])
        end = pd.Timestamp(reg["end"])
        mask = (idx >= start) & (idx <= end)
        n = int(mask.sum())
        if n == 0:
            windows_out.append(
                {
                    "id": reg["id"],
                    "label": reg.get("label", reg["id"]),
                    "status": "unavailable",
                    "n_days": 0,
                }
            )
            continue
        crisis_hit = float((labs.loc[mask].to_numpy() == "crisis").mean())
        turb_hit = float(turb[np.asarray(mask, dtype=bool)].mean())
        windows_out.append(
            {
                "id": reg["id"],
                "label": reg.get("label", reg["id"]),
                "status": "ok",
                "n_days": n,
                "crisis_frac": crisis_hit,
                "turbulent_frac": turb_hit,
            }
        )
    return {"status": "ok", "windows": windows_out}


def _macro_cols_from_frame(
    macro: pd.DataFrame | None,
    n_rows: int,
) -> np.ndarray | None:
    """PIT VIX / HY OAS / term-spread columns when present (pre-registered set)."""
    if macro is None or len(macro) != n_rows:
        return None
    cols: list[np.ndarray] = []
    for level_key, chg_key in (
        ("vix_level", "vix_chg_21"),
        ("hy_oas_level", "hy_oas_chg_21"),
        ("term_spread_level", "term_spread_chg_21"),
    ):
        if level_key in macro.columns:
            cols.append(macro[level_key].to_numpy(dtype=np.float64))
        elif chg_key in macro.columns:
            cols.append(macro[chg_key].to_numpy(dtype=np.float64))
    if len(cols) < 1:
        return None
    return np.column_stack(cols)


def _agreement_block(
    asset_returns: np.ndarray,
    *,
    turbulence_window: int,
    hmm_window: int,
    hmm_step: int = 21,
    hmm_n_iter: int = 200,
    macro_cols: np.ndarray | None = None,
    return_series: bool = False,
    return_models: bool = False,
    inflation_series: np.ndarray | None = None,
    dates: pd.DatetimeIndex | None = None,
    include_markov_robustness: bool = False,
) -> dict[str, Any]:
    r = np.asarray(asset_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("asset_returns must be (T, n)")
    t_len = r.shape[0]
    try:
        turb = turbulence_index(
            r, window=int(turbulence_window), macro_cols=macro_cols
        )
        q75_mask = classify_regime(turb, quantile=0.75)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"turbulence failed: {exc}",
            "jaccard_turbulence_hmm": float("nan"),
            "turbulent_day_frac": float("nan"),
            "operational_label": "markov_filtered_p05",
        }

    n_y = r.shape[1] + (
        0 if macro_cols is None else int(np.asarray(macro_cols).shape[1])
    )
    chi2_thr = chi2_turbulence_threshold(n_y, quantile=0.75)
    chi2_mask = classify_regime_chi2(turb, n_cols=n_y, quantile=0.75)
    duration_q75 = run_duration_stats(q75_mask)

    # Prefer statsmodels Hamilton filter; hmmlearn fallback if missing.
    p_high = np.full(t_len, np.nan, dtype=np.float64)
    hard = np.full(t_len, -1, dtype=np.int32)
    hard_piger = np.full(t_len, -1, dtype=np.int32)
    hard_growing = np.full(t_len, -1, dtype=np.int32)
    operational = np.zeros(t_len, dtype=bool)
    valid = np.zeros(t_len, dtype=bool)
    models: dict[int, Any] | None = None
    filt: dict[str, Any] | None = None
    backend = "unavailable"
    p11 = float("nan")
    exp_dur = float("nan")
    jacc = float("nan")
    jacc_piger = float("nan")
    status = "unavailable"
    reason = ""
    fit_hygiene: dict[str, Any] = {}

    try:
        filt = walk_forward_markov_filter(
            turb,
            window=int(hmm_window),
            step=int(hmm_step),
            k_regimes=2,
            growing=False,
            return_models=return_models,
        )
        backend = "statsmodels_markov"
        hard = np.asarray(filt["hard"], dtype=np.int32)
        hard_piger = np.asarray(filt["hard_piger"], dtype=np.int32)
        p_high = np.asarray(filt["p_highvol"], dtype=np.float64)
        operational = hard == 1
        valid = hard >= 0
        p11 = float(filt.get("p11_highvol", float("nan")))
        exp_dur = float(filt.get("expected_duration_highvol", float("nan")))
        fit_hygiene = dict(filt.get("fit_hygiene") or {})
        if return_models:
            models = filt.get("models")
    except ImportError:
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401

            turb_feat = np.asarray(turb, dtype=np.float64).reshape(-1, 1)
            filt = walk_forward_hmm_filter(
                turb_feat,
                window=int(hmm_window),
                step=int(hmm_step),
                n_components=2,
                random_state=42,
                n_iter=int(hmm_n_iter),
                return_models=return_models,
            )
            backend = "hmmlearn_fallback"
            hard = np.asarray(filt["hard"], dtype=np.int32)
            p_high = np.asarray(filt["p_highvol"], dtype=np.float64)
            operational = hard == 1
            valid = hard >= 0
            if return_models:
                models = filt.get("models")
        except ImportError:
            status = "unavailable"
            reason = "statsmodels and hmmlearn unavailable"
            filt = None

    if filt is not None:
        labeled_frac = float(fit_hygiene.get("labeled_frac", float("nan")))
        if not np.isfinite(labeled_frac):
            n_post = max(0, t_len - int(hmm_window))
            labeled_frac = (
                float(int(valid.sum()) / n_post) if n_post > 0 else float("nan")
            )
            fit_hygiene["labeled_frac"] = labeled_frac
        if int(valid.sum()) < 10:
            jacc = float("nan")
            status = "partial"
            reason = "insufficient HMM labels after warmup"
        else:
            jacc = float(jaccard_turbulent(q75_mask[valid], operational[valid]))
            status = "ok"
            reason = ""
        if np.isfinite(labeled_frac) and labeled_frac < 0.80:
            status = "partial"
            reason = "Markov labeled_frac below 0.80"
        piger_valid = hard_piger >= 0
        if int(piger_valid.sum()) >= 10:
            jacc_piger = float(
                jaccard_turbulent(
                    q75_mask[piger_valid], hard_piger[piger_valid] == 1
                )
            )

    duration_op = run_duration_stats(operational)

    # Robustness: growing-window filtered Markov (not headline; expensive).
    jacc_growing = float("nan")
    jacc_op_growing = float("nan")
    growing_block: dict[str, Any] = {"status": "skipped", "reason": "robustness off"}
    if include_markov_robustness and filt is not None and backend == "statsmodels_markov":
        try:
            grow = walk_forward_markov_filter(
                turb,
                window=int(hmm_window),
                step=int(hmm_step),
                k_regimes=2,
                growing=True,
                return_models=False,
            )
            g_hard = np.asarray(grow["hard"], dtype=np.int32)
            hard_growing = g_hard
            g_valid = g_hard >= 0
            if int(g_valid.sum()) >= 10:
                jacc_growing = float(
                    jaccard_turbulent(q75_mask[g_valid], g_hard[g_valid] == 1)
                )
                jacc_op_growing = float(
                    jaccard_turbulent(operational[g_valid], g_hard[g_valid] == 1)
                )
                growing_block = {
                    "status": "ok",
                    "n_labeled": int(g_valid.sum()),
                    "jaccard_q75_vs_growing": jacc_growing,
                    "jaccard_operational_vs_growing": jacc_op_growing,
                }
            else:
                growing_block = {"status": "partial", "reason": "insufficient labels"}
        except Exception as exc:
            growing_block = {"status": "unavailable", "reason": str(exc)}

    # KPT monthly average robustness (resample ME mean of daily turb).
    kpt_monthly: dict[str, Any] = {
        "status": "skipped" if not include_markov_robustness else "unavailable",
        "reason": "robustness off" if not include_markov_robustness else "need dates",
    }
    if include_markov_robustness and dates is not None and len(dates) == t_len:
        try:
            turb_s = pd.Series(turb, index=pd.DatetimeIndex(dates))
            flag_s = pd.Series(q75_mask.astype(float), index=turb_s.index)
            monthly_turb = turb_s.resample("ME").mean()
            monthly_flag = flag_s.resample("ME").mean() > 0.5
            if len(monthly_turb) >= 36:
                m_filt = walk_forward_markov_filter(
                    monthly_turb.to_numpy(dtype=np.float64),
                    window=min(36, max(24, len(monthly_turb) // 3)),
                    step=1,
                    k_regimes=2,
                    growing=True,
                    return_models=False,
                )
                m_hard = np.asarray(m_filt["hard"], dtype=np.int32)
                m_p = np.asarray(m_filt["p_highvol"], dtype=np.float64)
                m_valid = m_hard >= 0
                m_flags = monthly_flag.to_numpy(dtype=bool)
                if int(m_valid.sum()) >= 10:
                    m_piger = np.zeros(len(m_p), dtype=bool)
                    for i in range(2, len(m_p)):
                        window_p = m_p[i - 2 : i + 1]
                        if np.isfinite(window_p).all() and bool(
                            np.all(window_p >= 0.8)
                        ):
                            m_piger[i] = True
                    kpt_monthly = {
                        "status": "ok",
                        "n_months": int(len(monthly_turb)),
                        "jaccard_vs_monthly_q75_mean": float(
                            jaccard_turbulent(
                                m_flags[m_valid], m_hard[m_valid] == 1
                            )
                        ),
                        "kpt_monthly_piger_frac": float(m_piger.mean()),
                        "piger_note": (
                            "3 consecutive months filtered P>=0.8 "
                            "(Chauvet-Piger style); daily 5xP>0.8 is separate"
                        ),
                    }
                else:
                    kpt_monthly = {
                        "status": "partial",
                        "n_months": int(len(monthly_turb)),
                        "reason": "insufficient monthly Markov labels",
                    }
            else:
                kpt_monthly = {
                    "status": "unavailable",
                    "reason": f"T_month={len(monthly_turb)} < 36",
                }
        except Exception as exc:
            kpt_monthly = {"status": "unavailable", "reason": str(exc)}

    # Separate inflation 2-state (KPT); never mixed into turbulence HMM.
    infl_block: dict[str, Any] = {"status": "unavailable", "reason": "no inflation series"}
    infl_p_high = None
    if inflation_series is not None:
        try:
            infl = walk_forward_markov_filter(
                np.asarray(inflation_series, dtype=np.float64),
                window=int(hmm_window),
                step=int(hmm_step),
                k_regimes=2,
                return_models=False,
            )
            infl_p_high = infl["p_highvol"]
            infl_block = {
                "status": "ok",
                "p11_highvol": infl.get("p11_highvol"),
                "n_labeled": int(np.sum(np.asarray(infl["hard"]) >= 0)),
            }
        except Exception as exc:
            infl_block = {"status": "unavailable", "reason": str(exc)}

    out: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "operational_label": "markov_filtered_p05",
        "hmm_backend": backend,
        "jaccard_turbulence_hmm": jacc,
        "jaccard_turbulence_hmm_piger": jacc_piger,
        "jaccard_turbulence_hmm_growing": jacc_growing,
        "jaccard_q75_vs_growing": jacc_growing,
        "jaccard_operational_vs_growing": jacc_op_growing,
        "jaccard_note": (
            "Jaccard(expanding-q75 Skulls comparator, operational filtered P>0.5)"
        ),
        "piger_daily_note": (
            "5x daily filtered P>0.8; not Chauvet-Piger monthly dating"
        ),
        "taxonomy_disclaimer": (
            "binary operational turb vs 3-state macro; not a detector failure"
        ),
        "turbulent_day_frac": float(duration_op["turbulent_day_frac"]),
        "mean_turbulent_run_days": duration_op["mean_turbulent_run_days"],
        "n_turbulent_runs": duration_op["n_turbulent_runs"],
        "turbulence_switch_rate": duration_op["switch_rate"],
        "turbulent_day_frac_q75": float(duration_q75["turbulent_day_frac"]),
        "mean_turbulent_run_days_q75": duration_q75["mean_turbulent_run_days"],
        "turbulence_switch_rate_q75": duration_q75["switch_rate"],
        "hmm_mean_turbulent_run_days": duration_op["mean_turbulent_run_days"],
        "hmm_switch_rate": duration_op["switch_rate"],
        "hmm_p11_highvol": p11,
        "hmm_expected_duration_highvol": exp_dur,
        "chi2_threshold": chi2_thr,
        "chi2_turbulent_day_frac": (
            float(chi2_mask.mean()) if chi2_mask.size else float("nan")
        ),
        "jaccard_empirical_vs_chi2": float(jaccard_turbulent(q75_mask, chi2_mask)),
        "macro_cols_used": int(
            0 if macro_cols is None else np.asarray(macro_cols).shape[1]
        ),
        "n_hmm_labeled": int(np.asarray(valid).sum()),
        "hmm_step": int(hmm_step),
        "hmm_n_iter": int(hmm_n_iter),
        "fit_hygiene": fit_hygiene,
        "growing_window": growing_block,
        "kpt_monthly": kpt_monthly,
        "turbulent_mask": operational,
        "turbulent_q75_mask": q75_mask,
        "hmm_mask": operational,
        "chi2_mask": chi2_mask,
        "inflation_markov": infl_block,
    }
    if return_series:
        out["turbulence"] = turb
        out["hmm_p_highvol"] = p_high
        out["hmm_hard"] = hard
        out["hmm_hard_piger"] = hard_piger
        out["hmm_hard_growing"] = hard_growing
        out["turbulent_chi2"] = chi2_mask
        out["turbulent_q75"] = q75_mask
        out["train_ends"] = list(filt.get("train_ends", [])) if filt is not None else []
        if infl_p_high is not None:
            out["inflation_p_high"] = infl_p_high
    elif infl_p_high is not None:
        out["inflation_p_high"] = infl_p_high
    if return_models and models is not None:
        out["models"] = models
    return out


def build_regime_scorecard(
    *,
    macro: pd.DataFrame | None = None,
    asset_returns: np.ndarray | None = None,
    repo_root: Path | str | None = None,
    usb_lake_root: Path | str | None = None,
    min_history_days: int = 756,
    persistence_days: int = 10,
    turbulence_window: int = 252,
    hmm_window: int = 252 * 3,
    hmm_step: int = 21,
    hmm_n_iter: int = 200,
    include_wiring: bool = True,
    include_leverage: bool = True,
    behavior_path: Path | str | None = None,
    use_macro_yt: bool = True,
    return_series: bool = False,
    return_models: bool = False,
    include_markov_robustness: bool = False,
) -> dict[str, Any]:
    """Assemble detector scorecard (+ optional wiring / leverage embeds)."""
    root = (
        Path(repo_root)
        if repo_root is not None
        else REPO_ROOT
    )
    limitations: list[str] = []
    hygiene: dict[str, Any] = {"status": "unavailable", "reason": "macro not provided"}
    occupancy: dict[str, Any] = {
        "status": "unavailable",
        "reason": "macro not provided",
    }
    labels: pd.Series | None = None
    meta: pd.DataFrame | None = None
    dates: pd.DatetimeIndex | None = None

    if macro is not None and len(macro) > 0:
        hygiene = hygiene_prefix_stability(
            macro,
            min_history_days=min_history_days,
            persistence_days=persistence_days,
        )
        labels, meta = label_regimes(
            macro,
            min_history_days=min_history_days,
            persistence_days=persistence_days,
        )
        occupancy = occupancy_stats(labels, meta)
        dates = pd.DatetimeIndex(macro.index)

    agreement: dict[str, Any] = {
        "status": "unavailable",
        "reason": "asset_returns not provided",
        "jaccard_turbulence_hmm": float("nan"),
        "turbulent_day_frac": float("nan"),
    }
    turb_mask = None
    chi2_mask = None
    series_payload: dict[str, Any] = {}
    models_payload: dict[int, Any] | None = None
    macro_sources: dict[str, str] = {}
    if asset_returns is not None:
        if use_macro_yt and dates is not None:
            from mascotrl.eval.regime_dual_source import resolve_macro_yt_cols

            macro_cols, macro_sources = resolve_macro_yt_cols(
                macro, dates, usb_root=usb_lake_root
            )
        else:
            macro_cols = (
                _macro_cols_from_frame(macro, int(np.asarray(asset_returns).shape[0]))
                if use_macro_yt
                else None
            )
        if use_macro_yt and macro is not None and macro_cols is None:
            limitations.append(
                "macro_cols unavailable for y_t "
                "(returns-only turbulence; not silent zeros)"
            )
        infl_arr = None
        if macro is not None and "inflation_yoy_level" in macro.columns:
            infl_arr = macro["inflation_yoy_level"].to_numpy(dtype=np.float64)
        agreement = _agreement_block(
            asset_returns,
            turbulence_window=turbulence_window,
            hmm_window=hmm_window,
            hmm_step=hmm_step,
            hmm_n_iter=hmm_n_iter,
            macro_cols=macro_cols,
            return_series=return_series,
            return_models=return_models,
            inflation_series=infl_arr,
            dates=dates,
            include_markov_robustness=include_markov_robustness,
        )
        if macro_sources:
            agreement["macro_source"] = macro_sources
        turb_mask = agreement.pop("turbulent_mask", None)
        q75_mask = agreement.pop("turbulent_q75_mask", None)
        agreement.pop("hmm_mask", None)
        chi2_mask = agreement.pop("chi2_mask", None)
        # Inflation rule vs Markov Jaccard (cross-taxonomy).
        infl_meta = agreement.get("inflation_markov") or {}
        if (
            labels is not None
            and infl_meta.get("status") == "ok"
            and return_series
            and "inflation_p_high" in agreement
        ):
            # Will compute after series extracted below
            pass
        if return_series:
            for k in (
                "turbulence",
                "hmm_p_highvol",
                "hmm_hard",
                "hmm_hard_piger",
                "hmm_hard_growing",
                "turbulent_chi2",
                "turbulent_q75",
                "train_ends",
                "inflation_p_high",
            ):
                if k in agreement:
                    series_payload[k] = agreement.pop(k)
        if return_models and "models" in agreement:
            models_payload = agreement.pop("models")

        # Overlay 3-state for per_regime_sharpe (calm→crisis only on operational).
        from mascotrl.reporting.behavior_metrics import turbulence_regimes_from_returns
        from mascotrl.spectrum.policy_mode import per_regime_sharpe

        existing = labels.to_numpy() if labels is not None else None
        overlay = turbulence_regimes_from_returns(
            asset_returns,
            existing=existing,
            macro_cols=macro_cols,
            crisis_mask=turb_mask,
        )
        if overlay is not None:
            ew = np.nanmean(np.asarray(asset_returns, dtype=np.float64), axis=1)
            agreement["causal_per_regime_sharpe"] = per_regime_sharpe(ew, overlay)
            if return_series:
                series_payload["labels"] = (
                    pd.Series(overlay, index=dates) if dates is not None else overlay
                )
        if q75_mask is not None and return_series and "turbulent_q75" not in series_payload:
            series_payload["turbulent_q75"] = q75_mask

    events: dict[str, Any] = {
        "status": "unavailable",
        "reason": "need macro dates + turb mask",
    }
    if labels is not None and dates is not None and turb_mask is not None:
        if len(turb_mask) == len(dates):
            events = event_alignment(dates, labels, turb_mask)
        else:
            events = {
                "status": "unavailable",
                "reason": f"length mismatch labels={len(dates)} turb={len(turb_mask)}",
            }
            limitations.append(events["reason"])

    if labels is not None and turb_mask is not None and len(turb_mask) == len(labels):
        crisis = labels.astype(str).to_numpy() == "crisis"
        agreement["jaccard_macro_crisis_turbulence"] = float(
            jaccard_turbulent(crisis, turb_mask)
        )
        agreement["jaccard_macro_crisis_note"] = (
            "cross-taxonomy (3-state macro vs binary turbulence); "
            "not a detector failure"
        )
        agreement["taxonomy_disclaimer"] = (
            "binary operational turb vs 3-state macro; not a detector failure"
        )
        # Inflation rule vs separate KPT inflation Markov.
        infl_p = series_payload.get("inflation_p_high")
        if infl_p is None:
            infl_p = agreement.pop("inflation_p_high", None)
        elif "inflation_p_high" in agreement:
            agreement.pop("inflation_p_high", None)
        if infl_p is not None:
            infl_hard = np.asarray(infl_p, dtype=np.float64) > 0.5
            rule_infl = labels.astype(str).to_numpy() == "inflationary"
            valid_i = np.isfinite(np.asarray(infl_p, dtype=np.float64))
            if int(valid_i.sum()) >= 10:
                agreement["jaccard_inflation_rule_vs_markov"] = float(
                    jaccard_turbulent(rule_infl[valid_i], infl_hard[valid_i])
                )
                agreement["jaccard_inflation_note"] = (
                    "cross-taxonomy / separate KPT inflation series"
                )

    # Calendar stress windows stay separate from causal_per_regime_sharpe.
    agreement["calendar_stress_windows"] = events

    downstream: dict[str, Any] = {"status": "skipped", "reason": "no behavior_path"}
    if behavior_path is not None:
        bp = Path(behavior_path)
        if bp.is_file():
            import json

            try:
                blob = json.loads(bp.read_text(encoding="utf-8"))
                deltas = (
                    blob.get("regime_behaviour_deltas")
                    or blob.get("behaviour_by_regime")
                    or {}
                )
                downstream = {
                    "status": "ok",
                    "has_behaviour_by_regime": bool(blob.get("behaviour_by_regime")),
                    "delta_keys": sorted(
                        k
                        for k in (deltas.keys() if isinstance(deltas, Mapping) else [])
                        if str(k).startswith("delta_")
                    ),
                }
            except Exception as exc:
                downstream = {"status": "unavailable", "reason": str(exc)}
        else:
            downstream = {"status": "unavailable", "reason": f"missing {bp}"}

    wiring = None
    if include_wiring:
        from mascotrl.eval.regime_wiring_audit import audit_regime_wiring

        wiring = audit_regime_wiring(root)

    leverage = None
    if include_leverage:
        from mascotrl.eval.macro_lake_leverage import build_macro_lake_leverage

        leverage = build_macro_lake_leverage(
            repo_root=root,
            usb_lake_root=usb_lake_root,
        )

    if hygiene.get("status") == "fail":
        limitations.append("macro prefix hygiene failed")
    if agreement.get("status") == "unavailable":
        limitations.append(str(agreement.get("reason") or "agreement unavailable"))

    overall = "ok"
    if hygiene.get("status") == "fail":
        overall = "fail"
    elif agreement.get("status") in ("unavailable", "partial") or occupancy.get(
        "status"
    ) == "unavailable":
        overall = "partial"

    j = agreement.get("jaccard_turbulence_hmm")
    if isinstance(j, (int, float)) and np.isfinite(j):
        if j >= 0.4:
            agreement["jaccard_grade"] = "agree"
        else:
            agreement["jaccard_grade"] = "limitation"
            limitations.append(
                f"Jaccard turb↔HMM={j:.3f} < 0.4 (report as limitation)"
            )

    out: dict[str, Any] = {
        "status": overall,
        "hygiene": hygiene,
        "agreement": agreement,
        "occupancy": occupancy,
        "event_alignment": events,
        "downstream": downstream,
        "wiring": wiring,
        "leverage": {
            "status": leverage.get("status") if leverage else None,
            "metrics": leverage.get("metrics") if leverage else None,
            "productive_gaps": leverage.get("productive_gaps") if leverage else None,
            "n_assets": len(leverage.get("assets", [])) if leverage else 0,
        }
        if leverage
        else None,
        "limitations": limitations,
    }
    if return_series:
        series_out = {
            **series_payload,
            "turbulent_mask": turb_mask,
            "dates": dates,
        }
        # Prefer overlay labels (calm→crisis only) when present.
        if "labels" not in series_out:
            series_out["labels"] = labels
        out["_series"] = series_out
    if return_models:
        out["_models"] = models_payload
    return out
