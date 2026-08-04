"""Non-RL ceiling arms: z-score / ridge composites and Kelly surface CNN.

Weight callables match ``parity_harness.score_strategy``:
``(returns_hist, *, t, w_prev, **kw) -> (K,)``.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WeightFn = Callable[..., np.ndarray]

CEILING_ARM_NAMES: tuple[str, ...] = (
    "zscore_composite",
    "ridge_composite",
    "kelly_cnn",
)

_EPS = 1e-12


def _l1_normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(np.sum(np.abs(w)))
    if denom <= _EPS:
        if w.size == 0:
            return w
        return np.full(w.size, 1.0 / w.size, dtype=np.float64)
    return w / denom


def _scores_to_weights(scores: np.ndarray, *, long_only: bool) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if long_only:
        # Softmax over finite scores.
        m = float(np.max(s)) if s.size else 0.0
        ex = np.exp(s - m)
        return _l1_normalize(ex)
    # Long-short: demean then L1.
    s = s - float(np.mean(s)) if s.size else s
    return _l1_normalize(s)


def _cs_zscore(row: np.ndarray) -> np.ndarray:
    x = np.asarray(row, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x)
    out = np.zeros_like(x)
    if int(mask.sum()) < 2:
        return out
    mu = float(np.mean(x[mask]))
    sd = float(np.std(x[mask]))
    if sd <= _EPS:
        return out
    out[mask] = (x[mask] - mu) / sd
    return out


def zscore_composite_weights(
    signals: Mapping[str, np.ndarray],
    *,
    t: int,
    long_only: bool = True,
) -> np.ndarray:
    """Average cross-sectional z-scores at date ``t`` → portfolio weights."""
    if not signals:
        return np.zeros(0, dtype=np.float64)
    panels = [np.asarray(v, dtype=np.float64) for v in signals.values()]
    k = int(panels[0].shape[1])
    tt = int(t)
    zs = []
    for p in panels:
        if p.ndim != 2 or p.shape[1] != k:
            raise ValueError("all signals must be (T, K) with shared K")
        if tt < 0 or tt >= p.shape[0]:
            raise ValueError(f"t={tt} out of range for signal T={p.shape[0]}")
        zs.append(_cs_zscore(p[tt]))
    score = np.mean(np.stack(zs, axis=0), axis=0)
    return _scores_to_weights(score, long_only=bool(long_only))


def ridge_composite_weights(
    signals: Mapping[str, np.ndarray],
    returns_hist: np.ndarray,
    *,
    t: int,
    l2: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    """Ridge of next-period returns on signals using history strictly before ``t``.

    ``returns_hist`` is ``returns[:t]`` (parity convention). Training pairs use
    signal rows ``0..t-2`` and ``returns_hist[1..t-1]``.
    """
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be (T_hist, K)")
    names = list(signals.keys())
    if not names:
        return _l1_normalize(np.ones(r.shape[1], dtype=np.float64))
    panels = [np.asarray(signals[n], dtype=np.float64) for n in names]
    k = int(r.shape[1])
    tt = int(t)
    for p in panels:
        if p.ndim != 2 or p.shape[1] != k:
            raise ValueError("signals must be (T, K) matching returns K")
        if p.shape[0] < tt:
            raise ValueError("signal panel shorter than decision t")

    # Design: rows = dates 0..t-2; features = n_signals (per asset, pooled CS).
    n_sig = len(names)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    max_tau = min(tt - 1, r.shape[0] - 1)
    for tau in range(max_tau):
        # next return row in hist is tau+1
        y_row = r[tau + 1]
        feat = np.column_stack([p[tau] for p in panels])  # (K, n_sig)
        mask = np.isfinite(y_row) & np.all(np.isfinite(feat), axis=1)
        if not np.any(mask):
            continue
        xs.append(feat[mask])
        ys.append(y_row[mask])
    if not xs:
        return _l1_normalize(np.ones(k, dtype=np.float64))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    # Add intercept column.
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xa = np.hstack([ones, X])
    lam = float(l2)
    xtx = Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float64)
    xty = Xa.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    # Predict at date t-1 (last available signal before decision).
    pred_t = max(0, tt - 1)
    feat_t = np.column_stack([p[pred_t] for p in panels])
    scores = coef[0] + feat_t @ coef[1:]
    return _scores_to_weights(scores, long_only=bool(long_only))


class KellySurfaceCNN(nn.Module):
    """Tiny 2d CNN: (B, 1, 11, 34) surface patch → scalar score."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).flatten(1)
        return self.fc(h).squeeze(-1)

    def fit_expanding(
        self,
        images: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        *,
        epochs: int = 3,
        lr: float = 1e-2,
    ) -> "KellySurfaceCNN":
        """Stub expanding-window fit: a few Adam epochs on provided batch."""
        self.train()
        x = torch.as_tensor(images, dtype=torch.float32)
        y = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[0] != y.shape[0]:
            # Allow score-shaped targets broadcast from predict self-supervise.
            y = y[: x.shape[0]]
            if y.shape[0] != x.shape[0]:
                y = torch.zeros(x.shape[0], dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            pred = self.forward(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
        self.eval()
        return self

    def predict(self, images: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        x = torch.as_tensor(images, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        with torch.no_grad():
            scores = self.forward(x).cpu().numpy()
        return np.asarray(scores, dtype=np.float64).reshape(-1)

    def scores_to_weights(
        self,
        scores: np.ndarray,
        long_only: bool = True,
    ) -> np.ndarray:
        return _scores_to_weights(scores, long_only=bool(long_only))


def _select_kelly_image_batch(images: np.ndarray, *, t: int, k: int) -> np.ndarray | None:
    """Resolve the (K, 1, 11, 34) surface-patch batch at date ``t``.

    Accepted layouts: ``(T, K, 1, 11, 34)``, ``(T, K, 11, 34)``,
    ``(K, 1, 11, 34)``/``(K, 11, 34)`` at a single date. Returns ``None``
    when the layout does not resolve to exactly ``k`` per-asset patches
    (caller decides whether that is fail-closed or a fallback).
    """
    img = np.asarray(images, dtype=np.float64)
    tt = int(t)
    if img.ndim == 5 and img.shape[1] == k:
        row = img[min(tt, img.shape[0] - 1)]
    elif img.ndim == 4 and img.shape[0] == k and img.shape[1] in (1, 11):
        row = img
    elif img.ndim == 4 and img.shape[1] == k:
        row = img[min(tt, img.shape[0] - 1)]
    else:
        return None

    if row.ndim == 3 and row.shape[0] == k:
        batch = row[:, None, :, :] if row.shape[1] != 1 else row
    elif row.ndim == 4 and row.shape[0] == k:
        batch = row
    else:
        return None
    return batch


def _kelly_weights_at_t(
    model: KellySurfaceCNN,
    images: np.ndarray,
    *,
    t: int,
    k: int,
    long_only: bool,
) -> np.ndarray:
    """Resolve per-asset surface images at date t → L1 weights (K,)."""
    batch = _select_kelly_image_batch(images, t=t, k=k)
    if batch is None:
        return _l1_normalize(np.ones(k, dtype=np.float64))
    scores = model.predict(batch)
    if scores.size != k:
        return _l1_normalize(np.ones(k, dtype=np.float64))
    return model.scores_to_weights(scores, long_only=long_only)


class KellyEnsemble:
    """Expanding-window refit ensemble for ``kelly_cnn`` (B3).

    Every ``refit_every`` decisions, retrains ``n_seeds`` independently
    seeded :class:`KellySurfaceCNN` instances on all surface-patch /
    next-period-return pairs observed strictly before the current decision
    date ``t`` (an expanding window, never the future), then predicts by
    averaging the ensemble's scores. This replaces the single-shot
    ``epochs=3`` stub fit with a real walk-forward refit and reduces
    single-seed variance via ensembling.
    """

    def __init__(
        self,
        *,
        n_seeds: int = 3,
        refit_every: int = 21,
        epochs: int = 10,
        lr: float = 1e-2,
    ) -> None:
        self.n_seeds = int(n_seeds)
        self.refit_every = max(1, int(refit_every))
        self.epochs = int(epochs)
        self.lr = float(lr)
        self._models: list[KellySurfaceCNN] = []
        self._last_refit_t: int | None = None

    def _training_pairs(
        self, images: np.ndarray, returns_full: np.ndarray, t: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        img = np.asarray(images, dtype=np.float64)
        ret = np.asarray(returns_full, dtype=np.float64)
        k = int(ret.shape[1])
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        max_tau = min(int(t) - 1, img.shape[0] if img.ndim != 5 else img.shape[0], ret.shape[0] - 1)
        for tau in range(max(max_tau, 0)):
            batch = _select_kelly_image_batch(img, t=tau, k=k)
            if batch is None:
                continue
            row_ret = ret[tau + 1]
            mask = np.isfinite(row_ret)
            if not np.any(mask):
                continue
            xs.append(batch[mask])
            ys.append(row_ret[mask])
        if not xs:
            return None
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    def refit_if_due(self, images: np.ndarray, returns_full: np.ndarray, t: int) -> None:
        tt = int(t)
        if self._models and self._last_refit_t is not None and (tt - self._last_refit_t) < self.refit_every:
            return
        pairs = self._training_pairs(images, returns_full, tt)
        if pairs is None:
            return
        X, y = pairs
        models: list[KellySurfaceCNN] = []
        for seed in range(self.n_seeds):
            torch.manual_seed(seed)
            m = KellySurfaceCNN()
            m.fit_expanding(X, y, epochs=self.epochs, lr=self.lr)
            models.append(m)
        self._models = models
        self._last_refit_t = tt

    @property
    def is_fit(self) -> bool:
        return bool(self._models)

    def predict_ensemble(self, image_batch: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError(
                "KellyEnsemble.predict_ensemble called before any successful "
                "refit_if_due (no training pairs available yet)"
            )
        preds = [m.predict(image_batch) for m in self._models]
        return np.mean(np.stack(preds, axis=0), axis=0)


def ceiling_arm_weight_fn(name: str, **ctx: Any) -> WeightFn:
    """Build a parity-compatible weight function for a ceiling arm name."""
    key = str(name)
    if key not in CEILING_ARM_NAMES:
        raise KeyError(f"unknown ceiling arm: {key!r}; expected one of {CEILING_ARM_NAMES}")

    signals: Mapping[str, np.ndarray] = ctx.get("signals") or {}
    long_only = bool(ctx.get("long_only", True))
    l2 = float(ctx.get("l2", 1.0))
    returns_full = ctx.get("returns")
    kelly_model: KellySurfaceCNN | None = ctx.get("kelly_model")
    kelly_images = ctx.get("kelly_images")

    if key == "zscore_composite":

        def _zscore(
            returns_hist: np.ndarray,
            *,
            t: int,
            w_prev: np.ndarray | None = None,
            **_kw: Any,
        ) -> np.ndarray:
            del returns_hist, w_prev, _kw
            return zscore_composite_weights(signals, t=int(t), long_only=long_only)

        return _zscore

    if key == "ridge_composite":

        def _ridge(
            returns_hist: np.ndarray,
            *,
            t: int,
            w_prev: np.ndarray | None = None,
            **_kw: Any,
        ) -> np.ndarray:
            del w_prev, _kw
            hist = np.asarray(returns_hist, dtype=np.float64)
            if returns_full is not None:
                # Prefer full panel slice for clarity; still PIT via t.
                full = np.asarray(returns_full, dtype=np.float64)
                hist = full[: int(t)]
            return ridge_composite_weights(
                signals, hist, t=int(t), l2=l2, long_only=long_only
            )

        return _ridge

    # kelly_cnn: an explicit kelly_model override skips the expanding-window
    # refit (deterministic unit tests / a pre-fit checkpoint); the production
    # default trains a fresh KellyEnsemble with real walk-forward refits.
    ensemble = ctx.get("kelly_ensemble") if kelly_model is None else None
    if ensemble is None and kelly_model is None:
        ensemble = KellyEnsemble(
            n_seeds=int(ctx.get("kelly_n_seeds", 3)),
            refit_every=int(ctx.get("kelly_refit_every", 21)),
            epochs=int(ctx.get("kelly_epochs", 10)),
            lr=float(ctx.get("kelly_lr", 1e-2)),
        )

    def _kelly(
        returns_hist: np.ndarray,
        *,
        t: int,
        w_prev: np.ndarray | None = None,
        **_kw: Any,
    ) -> np.ndarray:
        del w_prev, _kw
        r = np.asarray(returns_hist, dtype=np.float64)
        k = int(r.shape[1]) if r.ndim == 2 else int(np.asarray(w_prev).size) if w_prev is not None else 0
        if kelly_images is None or k <= 0:
            # B3: kelly_cnn has no legitimate equal-weight fallback; a
            # missing surface-image feed is a configuration error, not a
            # degraded-but-valid ceiling arm.
            raise RuntimeError(
                "kelly_cnn requires kelly_images (per-asset IV-surface "
                "patches); refusing the prior silent equal-weight fallback"
            )
        if kelly_model is not None:
            return _kelly_weights_at_t(
                kelly_model, np.asarray(kelly_images), t=int(t), k=k, long_only=long_only
            )
        if returns_full is None:
            raise RuntimeError(
                "kelly_cnn expanding-window refit requires returns=<(T,K) panel>"
            )
        ensemble.refit_if_due(np.asarray(kelly_images), np.asarray(returns_full), int(t))
        batch = _select_kelly_image_batch(np.asarray(kelly_images), t=int(t), k=k)
        if batch is None:
            raise RuntimeError(f"kelly_images layout did not resolve to k={k} asset patches at t={t}")
        if not ensemble.is_fit:
            # Expanding-window warm-up: no training pairs exist before the
            # decision t itself, so there is no legitimate model yet (this
            # is not the "missing images" case above -- it is inherent to
            # any walk-forward estimator's first window). Equal-weight
            # until the first refit succeeds, exactly like ridge_composite's
            # own max_tau==0 fallback.
            return _l1_normalize(np.ones(k, dtype=np.float64))
        scores = ensemble.predict_ensemble(batch)
        return _scores_to_weights(scores, long_only=long_only)

    return _kelly


def list_ceiling_arms() -> tuple[str, ...]:
    return CEILING_ARM_NAMES
