"""Kritzman-Chow market turbulence index (causal rolling Mahalanobis).

Primary Ch.10 regime detector. Threshold classification uses an expanding
quantile so future turbulent episodes cannot recalibrate past labels.

Optional ``macro_cols`` (pre-registered Ch.10 co-movement): PIT-aligned
VIX level or 21d change, HY OAS level/change, term-spread level — already
lagged so they are known at t. Macro columns are **window-z-scored** before
concat (returns stay raw) so heterogeneous units do not dominate Mahalanobis.
Missing macro stays returns-only (callers must not invent zeros).

``classify_regime`` expanding quantile at t is inclusive of d_t. That is causal
because μ/Σ for d_t used only the past. Macro ``label_regimes`` ranks vs t-1;
do not force these rules to match by excluding d_t or by leaking future Σ.
"""
from __future__ import annotations

import numpy as np

# Pre-registered names for documentation / scorecard manifests.
TURBULENCE_MACRO_COL_IDS: tuple[str, ...] = (
    "vix_level_or_chg21",
    "hy_oas_level_or_chg21",
    "term_spread_level",
)


def _complete_case_cov(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean and cov from rows that are fully finite. Returns None if too few."""
    ok = np.isfinite(hist).all(axis=1)
    clean = hist[ok]
    n_rows, n_cols = clean.shape
    if n_rows < n_cols + 2:
        return None
    mu = clean.mean(axis=0)
    xc = clean - mu
    sigma = (xc.T @ xc) / max(n_rows - 1, 1)
    return mu, sigma


def turbulence_index(
    returns: np.ndarray,
    *,
    window: int = 252,
    macro_cols: np.ndarray | None = None,
    scale_macro: bool = True,
    min_names: int = 5,
) -> np.ndarray:
    """Squared Mahalanobis distance of each cross-section vs past window.

    At date ``t`` (0-based), ``mu`` and ``Sigma`` are estimated from
    ``y[t-window:t]`` only (strictly past). Scores ``y[t]``.
    Rows before ``window`` are NaN.

    When ``macro_cols`` is provided and ``scale_macro=True`` (default), each
    macro column is z-scored using that same past window only, then concatenated
    onto raw returns. Columns with NaN at t or too few finite history rows are
    dropped for that day. If fewer than ``min_names`` usable columns remain,
    ``out[t]`` is NaN (never invent zeros).
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns must be (T, n)")
    t_len, n = r.shape
    m_arr: np.ndarray | None = None
    if macro_cols is not None:
        m_arr = np.asarray(macro_cols, dtype=np.float64)
        if m_arr.ndim != 2:
            raise ValueError("macro_cols must be (T, m)")
        if m_arr.shape[0] != t_len:
            raise ValueError(
                f"macro_cols length {m_arr.shape[0]} != returns length {t_len}"
            )
    # Upper bound for window check (all columns if present).
    n_y_max = n + (0 if m_arr is None else m_arr.shape[1])
    if window < max(n_y_max, int(min_names)) + 1:
        raise ValueError(f"window={window} too small for n={n_y_max} columns")

    out = np.full(t_len, np.nan, dtype=np.float64)
    min_hist = max(window // 2, int(min_names) + 1)

    for t in range(window, t_len):
        hist_r = r[t - window : t]
        yt_r = r[t]

        usable_r: list[int] = []
        for j in range(n):
            if not np.isfinite(yt_r[j]):
                continue
            if int(np.isfinite(hist_r[:, j]).sum()) < min_hist:
                continue
            usable_r.append(j)
        parts_hist: list[np.ndarray] = []
        parts_yt: list[np.ndarray] = []
        if usable_r:
            parts_hist.append(hist_r[:, usable_r])
            parts_yt.append(yt_r[usable_r])

        if m_arr is not None:
            hist_m = m_arr[t - window : t]
            yt_m = m_arr[t]
            usable_m: list[int] = []
            z_hist_cols: list[np.ndarray] = []
            z_yt_vals: list[float] = []
            for j in range(m_arr.shape[1]):
                if not np.isfinite(yt_m[j]):
                    continue
                col = hist_m[:, j]
                finite = col[np.isfinite(col)]
                if finite.size < min_hist:
                    continue
                if scale_macro:
                    mu_j = float(np.mean(finite))
                    sd_j = float(np.std(finite, ddof=1))
                    if not np.isfinite(sd_j) or sd_j < 1e-12:
                        continue
                    z_h = (col - mu_j) / sd_j
                    z_h = np.where(np.isfinite(col), z_h, np.nan)
                    z_t = (float(yt_m[j]) - mu_j) / sd_j
                else:
                    z_h = col.astype(np.float64, copy=True)
                    z_t = float(yt_m[j])
                usable_m.append(j)
                z_hist_cols.append(z_h)
                z_yt_vals.append(z_t)
            if z_hist_cols:
                parts_hist.append(np.column_stack(z_hist_cols))
                parts_yt.append(np.asarray(z_yt_vals, dtype=np.float64))

        if not parts_hist:
            continue
        hist = np.concatenate(parts_hist, axis=1)
        yt = np.concatenate(parts_yt, axis=0)
        if hist.shape[1] < int(min_names):
            continue

        cov = _complete_case_cov(hist)
        if cov is None:
            # Pairwise fallback: nanmean + pairwise cov via masked arrays.
            mu = np.nanmean(hist, axis=0)
            if not np.isfinite(mu).all() or not np.isfinite(yt).all():
                continue
            # Drop columns that are all-NaN in hist after selection (should not happen).
            xc = hist - mu
            # Replace NaN with 0 for outer product only after centering known values;
            # use nan-aware Gram: for each pair, mean of product over finite pairs.
            n_cols = hist.shape[1]
            sigma = np.full((n_cols, n_cols), np.nan, dtype=np.float64)
            for a in range(n_cols):
                for b in range(a, n_cols):
                    mask = np.isfinite(xc[:, a]) & np.isfinite(xc[:, b])
                    if int(mask.sum()) < 2:
                        continue
                    val = float(np.dot(xc[mask, a], xc[mask, b]) / max(int(mask.sum()) - 1, 1))
                    sigma[a, b] = val
                    sigma[b, a] = val
            if not np.isfinite(sigma).all():
                continue
        else:
            mu, sigma = cov
            if not np.isfinite(yt).all():
                # yt may still have been filtered to finite; double-check.
                continue

        inv = np.linalg.pinv(sigma)
        d = yt - mu
        out[t] = float(d @ inv @ d)
    return out


def classify_regime(
    turbulence: np.ndarray,
    *,
    quantile: float = 0.75,
) -> np.ndarray:
    """Expanding-quantile turbulent/calm labels (True = turbulent).

    At index ``t``, the threshold is the ``quantile`` of ``turbulence[:t+1]``
    over finite values only (inclusive of d_t). NaN turbulence -> False.
    """
    if not (0.0 < float(quantile) < 1.0):
        raise ValueError(f"quantile must be in (0,1); got {quantile}")
    turb = np.asarray(turbulence, dtype=np.float64).reshape(-1)
    labels = np.zeros(turb.shape[0], dtype=bool)
    hist: list[float] = []
    for t, val in enumerate(turb):
        if not np.isfinite(val):
            continue
        hist.append(float(val))
        thr = float(np.quantile(np.asarray(hist, dtype=np.float64), quantile))
        # Strictly above the expanding quantile (ties at the threshold stay calm).
        labels[t] = val > thr
    return labels


def chi2_turbulence_threshold(n_cols: int, *, quantile: float = 0.75) -> float:
    """Theoretical chi-square threshold for n_cols Mahalanobis degrees of freedom.

    Robustness cross-check vs expanding empirical quantile (Chow et al.).
    Report both; do not pick whichever matches a known crisis window.
    """
    from scipy.stats import chi2

    if n_cols < 1:
        raise ValueError("n_cols must be >= 1")
    if not (0.0 < float(quantile) < 1.0):
        raise ValueError(f"quantile must be in (0,1); got {quantile}")
    return float(chi2.ppf(float(quantile), df=int(n_cols)))


def classify_regime_chi2(
    turbulence: np.ndarray,
    *,
    n_cols: int,
    quantile: float = 0.75,
) -> np.ndarray:
    """Hard labels using chi-square theoretical threshold (not expanding empirical)."""
    thr = chi2_turbulence_threshold(n_cols, quantile=quantile)
    turb = np.asarray(turbulence, dtype=np.float64).reshape(-1)
    labels = np.zeros(turb.shape[0], dtype=bool)
    finite = np.isfinite(turb)
    labels[finite] = turb[finite] > thr
    return labels
