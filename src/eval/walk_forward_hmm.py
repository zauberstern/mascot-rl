"""Walk-forward Gaussian HMM regime labels (leakage-safe cross-check).

Secondary to Kritzman turbulence. Fit only on a strictly-past rolling window
and refit on a schedule; never fit once on the full sample and predict back.

Kritzman–Page–Turkington (2012) models a 2-state Markov chain on the
**turbulence** series. Use :func:`walk_forward_hmm_filter` (forward-algorithm
filtered probabilities) for confirmatory Jaccard — not Viterbi ``predict`` on
a future step block (intra-block look-ahead).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class MarkovCheckpoint(NamedTuple):
    """Sealed statsmodels MarkovRegression params for one train_end."""

    params: np.ndarray
    high_idx: int
    k_regimes: int
    train_end: int
    train_window: int


def _remap_by_emission_vol(model: Any) -> dict[int, int]:
    """Map raw HMM state ids so remapped 0 = low vol, 1 = high vol."""
    n_components = int(model.n_components)
    cov = np.asarray(model.covars_, dtype=np.float64)
    vol = cov.reshape(n_components, -1).mean(axis=1)
    order = np.argsort(vol)
    return {int(old): int(new_idx) for new_idx, old in enumerate(order.tolist())}


def _forward_filter_posteriors(model: Any, x: np.ndarray) -> np.ndarray:
    """Causal P(s_t | x_0..x_t). Shape (T, n_components).

    Uses only the forward recursion (Hamilton filter), not forward-backward
    smoothing, so day t does not depend on observations after t.
    """
    from scipy.special import logsumexp

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be (T, d)")
    # hmmlearn private but stable across versions we pin; fallback would reimplement emissions.
    log_emit = np.asarray(model._compute_log_likelihood(x), dtype=np.float64)
    n_comp = int(model.n_components)
    t_len = x.shape[0]
    log_start = np.log(np.asarray(model.startprob_, dtype=np.float64) + 1e-300)
    log_trans = np.log(np.asarray(model.transmat_, dtype=np.float64) + 1e-300)

    log_alpha = np.empty((t_len, n_comp), dtype=np.float64)
    log_alpha[0] = log_start + log_emit[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, t_len):
        # log_alpha[t, j] = log_emit[t,j] + logsumexp_i(log_alpha[t-1,i] + log_trans[i,j])
        for j in range(n_comp):
            log_alpha[t, j] = log_emit[t, j] + logsumexp(log_alpha[t - 1] + log_trans[:, j])
        log_alpha[t] -= logsumexp(log_alpha[t])
    return np.exp(log_alpha)


def walk_forward_hmm_regimes(
    features: np.ndarray,
    *,
    window: int = 252 * 3,
    step: int = 21,
    n_components: int = 2,
    random_state: int = 42,
    n_iter: int = 200,
) -> np.ndarray:
    """Return integer regime ids; warmup / gaps are -1.

    After each fit on ``features[end-window:end]``, predict the next ``step``
    rows only via Viterbi. **Legacy / diagnostics only** — prefer
    :func:`walk_forward_hmm_filter` for KPT confirmatory labels (no intra-block
    look-ahead). Labels are sorted so state 0 = lower-vol (by train emission std).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "walk_forward_hmm_regimes requires hmmlearn; pip install hmmlearn"
        ) from exc

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be (T, d)")
    t_len, _d = x.shape
    if window < 10:
        raise ValueError("window too small")
    if step < 1:
        raise ValueError("step must be >= 1")

    labels = np.full(t_len, -1, dtype=np.int32)
    for end in range(window, t_len, step):
        train = x[end - window : end]
        # Drop non-finite rows in the train window.
        ok = np.isfinite(train).all(axis=1)
        train = train[ok]
        if train.shape[0] < max(20, n_components * 5):
            continue
        model = GaussianHMM(
            n_components=int(n_components),
            covariance_type="diag",
            n_iter=int(n_iter),
            random_state=int(random_state),
        )
        model.fit(train)
        remap = _remap_by_emission_vol(model)

        test_end = min(end + step, t_len)
        test = x[end:test_end]
        finite = np.isfinite(test).all(axis=1)
        if not finite.any():
            continue
        pred = np.full(test.shape[0], -1, dtype=np.int32)
        pred[finite] = model.predict(test[finite])
        for i, p in enumerate(pred):
            if p >= 0:
                labels[end + i] = remap[int(p)]
    return labels


def walk_forward_hmm_filter(
    features: np.ndarray,
    *,
    window: int = 252 * 3,
    step: int = 21,
    n_components: int = 2,
    random_state: int = 42,
    n_iter: int = 200,
    hard_threshold: float = 0.5,
    return_models: bool = False,
) -> dict[str, Any]:
    """Walk-forward 2-state HMM with **forward-filter** probabilities (KPT).

    Fit on ``features[end-window:end]`` only. For each OOS day ``t`` in
    ``[end, end+step)``, emit ``P(high-vol | observations through t)`` by
    continuing the Hamilton filter one day at a time (never Viterbi / FB on the
    future block). Hard label: ``P > hard_threshold`` (pre-registered 0.5).

    Returns dict with:
      - ``p_highvol``: float (T,), NaN where unset
      - ``hard``: int (T,), -1 warmup / unset, else 0/1
      - ``train_ends``: list of fit endpoints used
      - ``models`` (optional): ``{train_end: fitted GaussianHMM}`` for seal/replay
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "walk_forward_hmm_filter requires hmmlearn; pip install hmmlearn"
        ) from exc

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be (T, d)")
    t_len, _d = x.shape
    if window < 10:
        raise ValueError("window too small")
    if step < 1:
        raise ValueError("step must be >= 1")
    if int(n_components) != 2:
        raise ValueError(
            "n_components must be 2 (pre-registered; do not BIC-search K)"
        )
    thr = float(hard_threshold)

    p_high = np.full(t_len, np.nan, dtype=np.float64)
    hard = np.full(t_len, -1, dtype=np.int32)
    train_ends: list[int] = []
    models: dict[int, Any] = {}

    for end in range(window, t_len, step):
        train = x[end - window : end]
        ok = np.isfinite(train).all(axis=1)
        train_clean = train[ok]
        if train_clean.shape[0] < max(20, n_components * 5):
            continue
        model = GaussianHMM(
            n_components=int(n_components),
            covariance_type="diag",
            n_iter=int(n_iter),
            random_state=int(random_state),
        )
        model.fit(train_clean)
        remap = _remap_by_emission_vol(model)
        high_raw = next(old for old, new in remap.items() if new == 1)

        # Seed filter on the (finite) train window, then step one OOS day at a time.
        post_train = _forward_filter_posteriors(model, train_clean)
        log_alpha = np.log(post_train[-1] + 1e-300)

        from scipy.special import logsumexp

        log_trans = np.log(np.asarray(model.transmat_, dtype=np.float64) + 1e-300)

        test_end = min(end + step, t_len)
        for t in range(end, test_end):
            row = x[t : t + 1]
            if not np.isfinite(row).all():
                continue
            log_emit = np.asarray(
                model._compute_log_likelihood(row), dtype=np.float64
            ).reshape(-1)
            # One-step ahead filter update from previous filtered state.
            new_log = np.empty(n_components, dtype=np.float64)
            for j in range(n_components):
                new_log[j] = log_emit[j] + logsumexp(log_alpha + log_trans[:, j])
            new_log -= logsumexp(new_log)
            log_alpha = new_log
            post = np.exp(log_alpha)
            p = float(post[high_raw])
            p_high[t] = p
            hard[t] = 1 if p > thr else 0

        train_ends.append(int(end))
        if return_models:
            models[int(end)] = model

    out: dict[str, Any] = {
        "p_highvol": p_high,
        "hard": hard,
        "train_ends": train_ends,
    }
    if return_models:
        out["models"] = models
    return out


def walk_forward_markov_filter(
    series: np.ndarray,
    *,
    window: int = 252 * 3,
    step: int = 21,
    k_regimes: int = 2,
    growing: bool = False,
    hard_threshold: float = 0.5,
    piger_threshold: float = 0.8,
    piger_consecutive: int = 5,
    search_reps: int = 5,
    maxiter: int = 200,
    return_models: bool = False,
) -> dict[str, Any]:
    """Walk-forward 2-state MarkovRegression with Hamilton **filter** (KPT).

    Headline hard label: filtered P(high-vol | data through t) > 0.5.
    Robustness: ``hmm_hard_piger`` = 1 iff last ``piger_consecutive`` days all
    have filtered P > ``piger_threshold`` (filtered, not smoothed).

    Failed / non-finite fits reuse the last successful causal checkpoint so OOS
    blocks are not left unlabeled. Never uses smoothed probs. Never BIC-searches K.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "walk_forward_markov_filter requires statsmodels; pip install statsmodels"
        ) from exc

    if int(k_regimes) != 2:
        raise ValueError(
            "k_regimes must be 2 (pre-registered; Hamilton BIC/LR invalid for N)"
        )
    y = np.asarray(series, dtype=np.float64).reshape(-1)
    t_len = y.shape[0]
    if window < 50:
        raise ValueError("window too small")
    if step < 1:
        raise ValueError("step must be >= 1")
    thr = float(hard_threshold)
    piger_thr = float(piger_threshold)
    piger_n = int(piger_consecutive)

    p_high = np.full(t_len, np.nan, dtype=np.float64)
    hard = np.full(t_len, -1, dtype=np.int32)
    hard_piger = np.full(t_len, -1, dtype=np.int32)
    train_ends: list[int] = []
    models: dict[int, MarkovCheckpoint] = {}
    p11_list: list[float] = []
    dur_list: list[float] = []

    n_attempted = 0
    n_fit_ok = 0
    n_unconverged = 0
    n_reused = 0
    n_failed = 0
    last_ckpt: MarkovCheckpoint | None = None

    def _high_idx_from_params(mod: Any, params: np.ndarray) -> int:
        names = list(getattr(mod, "param_names", []) or [])
        try:
            i0 = names.index("sigma2[0]")
            i1 = names.index("sigma2[1]")
            s0 = float(params[i0])
            s1 = float(params[i1])
        except (ValueError, IndexError, TypeError):
            s0 = float(params[-2])
            s1 = float(params[-1])
        return 1 if s1 >= s0 else 0

    def _filter_block(
        *,
        train_start: int,
        end: int,
        test_end: int,
        params: np.ndarray,
        high_idx: int,
    ) -> bool:
        endog_block = y[train_start:test_end]
        finite_mask = np.isfinite(endog_block)
        if int(finite_mask.sum()) < 10:
            return False
        try:
            mod_block = sm.tsa.MarkovRegression(
                endog_block[finite_mask],
                k_regimes=2,
                trend="c",
                switching_variance=True,
            )
            filt = mod_block.filter(params)
            fmp = np.asarray(filt.filtered_marginal_probabilities, dtype=np.float64)
            if fmp.ndim == 1:
                fmp = fmp.reshape(-1, 1)
            p_fin = np.full(endog_block.shape[0], np.nan, dtype=np.float64)
            p_fin[finite_mask] = fmp[:, high_idx]
            for t in range(end, test_end):
                local = t - train_start
                p = float(p_fin[local])
                if not np.isfinite(p):
                    continue
                p_high[t] = p
                hard[t] = 1 if p > thr else 0
            return True
        except Exception:
            return False

    for end in range(window, t_len, step):
        if growing:
            train_slice = y[:end]
            train_start = 0
        else:
            train_slice = y[end - window : end]
            train_start = end - window
        ok = np.isfinite(train_slice)
        train_clean = train_slice[ok]
        if train_clean.shape[0] < max(50, 20):
            continue
        n_attempted += 1
        test_end = min(end + step, t_len)
        params: np.ndarray | None = None
        high_idx = 0
        reused = False
        fit_ok = False
        unconverged = False

        try:
            mod = sm.tsa.MarkovRegression(
                train_clean,
                k_regimes=2,
                trend="c",
                switching_variance=True,
            )
            res = mod.fit(disp=False, search_reps=int(search_reps), maxiter=int(maxiter))
            cand = np.asarray(res.params, dtype=np.float64)
            if not np.isfinite(cand).all():
                raise RuntimeError("non-finite Markov params")
            mle = getattr(res, "mle_retvals", None) or {}
            if mle.get("converged") is False:
                unconverged = True
            params = cand
            high_idx = _high_idx_from_params(mod, params)
            fit_ok = True
            # Persistence from this train result only.
            try:
                trans = np.asarray(res.regime_transition, dtype=np.float64)
                if trans.ndim == 3:
                    trans = trans[:, :, -1]
                p11_list.append(float(trans[high_idx, high_idx]))
            except Exception:
                pass
            try:
                durs = np.asarray(res.expected_durations, dtype=np.float64).reshape(-1)
                dur_list.append(float(durs[high_idx]))
            except Exception:
                pass
        except Exception:
            if last_ckpt is not None:
                params = np.asarray(last_ckpt.params, dtype=np.float64)
                high_idx = int(last_ckpt.high_idx)
                reused = True
            else:
                n_failed += 1
                continue

        assert params is not None
        labeled = _filter_block(
            train_start=train_start,
            end=end,
            test_end=test_end,
            params=params,
            high_idx=high_idx,
        )
        if not labeled:
            if reused:
                n_failed += 1
            elif fit_ok and last_ckpt is not None:
                # Fresh fit filtered poorly; try last checkpoint once.
                params = np.asarray(last_ckpt.params, dtype=np.float64)
                high_idx = int(last_ckpt.high_idx)
                reused = True
                labeled = _filter_block(
                    train_start=train_start,
                    end=end,
                    test_end=test_end,
                    params=params,
                    high_idx=high_idx,
                )
            if not labeled:
                n_failed += 1
                continue

        # Piger robustness after the block's p_high filled for this window.
        for t in range(end, test_end):
            if not np.isfinite(p_high[t]):
                continue
            if t + 1 < piger_n:
                hard_piger[t] = 0
                continue
            window_p = p_high[t + 1 - piger_n : t + 1]
            if np.isfinite(window_p).all() and bool(np.all(window_p > piger_thr)):
                hard_piger[t] = 1
            else:
                hard_piger[t] = 0

        train_ends.append(int(end))
        if fit_ok:
            n_fit_ok += 1
            if unconverged:
                n_unconverged += 1
            last_ckpt = MarkovCheckpoint(
                params=params.copy(),
                high_idx=int(high_idx),
                k_regimes=2,
                train_end=int(end),
                train_window=int(end - train_start),
            )
        if reused:
            n_reused += 1
        if return_models:
            ckpt = last_ckpt
            if ckpt is None:
                ckpt = MarkovCheckpoint(
                    params=params.copy(),
                    high_idx=int(high_idx),
                    k_regimes=2,
                    train_end=int(end),
                    train_window=int(end - train_start),
                )
            models[int(end)] = MarkovCheckpoint(
                params=np.asarray(ckpt.params, dtype=np.float64).copy(),
                high_idx=int(ckpt.high_idx),
                k_regimes=2,
                train_end=int(end),
                train_window=int(end - train_start),
            )

    n_labeled = int(np.sum(hard >= 0))
    n_post = max(0, t_len - int(window))
    labeled_frac = float(n_labeled / n_post) if n_post > 0 else float("nan")

    out: dict[str, Any] = {
        "p_highvol": p_high,
        "hard": hard,
        "hard_piger": hard_piger,
        "train_ends": train_ends,
        "p11_highvol": float(np.mean(p11_list)) if p11_list else float("nan"),
        "expected_duration_highvol": float(np.mean(dur_list)) if dur_list else float("nan"),
        "n_windows_attempted": int(n_attempted),
        "n_windows_fit_ok": int(n_fit_ok),
        "n_windows_unconverged": int(n_unconverged),
        "n_windows_reused": int(n_reused),
        "n_windows_failed": int(n_failed),
        "labeled_frac": labeled_frac,
        "fit_hygiene": {
            "n_windows_attempted": int(n_attempted),
            "n_windows_fit_ok": int(n_fit_ok),
            "n_windows_unconverged": int(n_unconverged),
            "n_windows_reused": int(n_reused),
            "n_windows_failed": int(n_failed),
            "labeled_frac": labeled_frac,
        },
    }
    if return_models:
        out["models"] = models
    return out


def apply_markov_checkpoint_filter(
    checkpoint: MarkovCheckpoint,
    series: np.ndarray,
    *,
    date_index: int,
    hmm_step: int,
    hard_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Apply sealed MarkovCheckpoint only on validity [train_end, train_end+step)."""
    import statsmodels.api as sm

    train_end = int(checkpoint.train_end)
    t = int(date_index)
    if train_end > t:
        raise ValueError(
            f"refusing backward application: train_end={train_end} > t={t}"
        )
    if t < train_end or t >= train_end + int(hmm_step):
        raise ValueError(
            f"date_index={t} outside checkpoint validity "
            f"[{train_end}, {train_end + int(hmm_step)})"
        )
    y = np.asarray(series, dtype=np.float64).reshape(-1)
    train_start = max(0, train_end - int(checkpoint.train_window))
    endog = y[train_start : t + 1]
    finite = np.isfinite(endog)
    endog_f = endog[finite]
    t_len = y.shape[0]
    p_high = np.full(t_len, np.nan, dtype=np.float64)
    hard = np.full(t_len, -1, dtype=np.int32)
    if endog_f.size < 10 or not finite[-1]:
        return {"p_highvol": p_high, "hard": hard}
    mod = sm.tsa.MarkovRegression(
        endog_f, k_regimes=int(checkpoint.k_regimes), trend="c", switching_variance=True
    )
    filt = mod.filter(np.asarray(checkpoint.params, dtype=np.float64))
    fmp = np.asarray(filt.filtered_marginal_probabilities, dtype=np.float64)
    if fmp.ndim == 1:
        p = float(fmp[-1])
    else:
        p = float(fmp[-1, int(checkpoint.high_idx)])
    p_high[t] = p
    hard[t] = 1 if p > float(hard_threshold) else 0
    return {"p_highvol": p_high, "hard": hard}


def apply_hmm_checkpoint_filter(
    model: Any,
    features: np.ndarray,
    *,
    train_end: int,
    dates_start: int,
    dates_end: int,
    hard_threshold: float = 0.5,
    train_window: int | None = None,
) -> dict[str, np.ndarray]:
    """Apply a sealed window HMM only on ``[dates_start, dates_end)``.

    Hard error if ``train_end > dates_start`` would imply applying a later
    checkpoint backward, or if the requested span is not inside
    ``[train_end, ...)`` for this checkpoint's validity.
    """
    if int(train_end) > int(dates_start):
        raise ValueError(
            f"refusing backward application: train_end={train_end} > dates_start={dates_start}"
        )
    if int(dates_start) < int(train_end):
        raise ValueError(
            f"checkpoint train_end={train_end} cannot label dates before it "
            f"(dates_start={dates_start})"
        )
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be (T, d)")
    remap = _remap_by_emission_vol(model)
    high_raw = next(old for old, new in remap.items() if new == 1)
    n_components = int(model.n_components)

    # Seed from train slice immediately before train_end when available.
    w = int(train_window) if train_window is not None else max(10, train_end)
    seed_start = max(0, train_end - w)
    train = x[seed_start:train_end]
    ok = np.isfinite(train).all(axis=1)
    train_clean = train[ok]
    if train_clean.shape[0] < 5:
        raise ValueError("insufficient finite train rows to seed filter")
    post_train = _forward_filter_posteriors(model, train_clean)
    log_alpha = np.log(post_train[-1] + 1e-300)

    from scipy.special import logsumexp

    log_trans = np.log(np.asarray(model.transmat_, dtype=np.float64) + 1e-300)
    thr = float(hard_threshold)
    t_len = x.shape[0]
    p_high = np.full(t_len, np.nan, dtype=np.float64)
    hard = np.full(t_len, -1, dtype=np.int32)

    for t in range(int(dates_start), min(int(dates_end), t_len)):
        row = x[t : t + 1]
        if not np.isfinite(row).all():
            continue
        log_emit = np.asarray(
            model._compute_log_likelihood(row), dtype=np.float64
        ).reshape(-1)
        new_log = np.empty(n_components, dtype=np.float64)
        for j in range(n_components):
            new_log[j] = log_emit[j] + logsumexp(log_alpha + log_trans[:, j])
        new_log -= logsumexp(new_log)
        log_alpha = new_log
        post = np.exp(log_alpha)
        p = float(post[high_raw])
        p_high[t] = p
        hard[t] = 1 if p > thr else 0

    return {"p_highvol": p_high, "hard": hard}


def jaccard_turbulent(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two boolean turbulent masks."""
    aa = np.asarray(a, dtype=bool).reshape(-1)
    bb = np.asarray(b, dtype=bool).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError("mask shapes must match")
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def hmm_turbulent_mask(labels: np.ndarray, *, turbulent_state: int = 1) -> np.ndarray:
    """Map HMM integer labels to boolean turbulent (default state 1 = high-vol)."""
    lab = np.asarray(labels)
    return lab == int(turbulent_state)
