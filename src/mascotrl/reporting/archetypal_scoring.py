"""Composition-based archetype scoring (Archetypal Analysis + fallbacks).

Every cell is an honest mixture. No residual catch-all bucket.
Interpretation only. Never feeds capital gates.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Import seed weights for naming; keep names aligned with policy_behavior.
from mascotrl.reporting.policy_behavior import ARCHETYPE_SCORE_WEIGHTS

DEFAULT_K = 5


def _zscore(X: np.ndarray) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    out = (x - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _row_simplex(W: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(W, dtype=np.float64), 0.0, None)
    s = w.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return w / s


def _fit_aa(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Try archetypes.AA; raise on ImportError / numerical failure."""
    from archetypes import AA  # type: ignore

    model = AA(n_archetypes=int(k), init="furthest_sum", n_init=10, random_state=0)
    model.fit(X)
    alpha = np.asarray(model.transform(X), dtype=np.float64)
    archetypes = np.asarray(model.archetypes_, dtype=np.float64)
    return _row_simplex(alpha), archetypes


def _fit_nmf(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import NMF

    # Shift to non-negative for NMF.
    shifted = X - X.min(axis=0, keepdims=True) + 1e-8
    model = NMF(
        n_components=int(k),
        init="nndsvda",
        random_state=0,
        max_iter=500,
    )
    W = model.fit_transform(shifted)
    H = model.components_
    return _row_simplex(W), np.asarray(H, dtype=np.float64)


def _fit_gmm(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=int(k),
        covariance_type="diag",
        random_state=0,
        n_init=5,
    )
    model.fit(X)
    alpha = model.predict_proba(X)
    return _row_simplex(alpha), np.asarray(model.means_, dtype=np.float64)


def fit_composition(
    X_std: np.ndarray,
    k: int = DEFAULT_K,
    method: str = "aa",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit mixture composition. Returns ``(alpha, archetypes)``.

    ``method``:
      - ``aa``: Archetypal Analysis (falls back to NMF on ImportError/failure)
      - ``nmf``: non-negative matrix factorization with row-normalized W
      - ``gmm``: Gaussian mixture soft membership (cross-check)
    """
    X = np.asarray(X_std, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"X_std must be (n>=2, p); got {X.shape}")
    k = int(min(max(int(k), 2), X.shape[0]))
    key = str(method or "aa").lower()
    if key == "gmm":
        return _fit_gmm(X, k)
    if key == "nmf":
        return _fit_nmf(X, k)
    # aa with fallback
    try:
        return _fit_aa(X, k)
    except Exception:
        return _fit_nmf(X, k)


def name_archetypes(
    Z: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    seed_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> list[str]:
    """Map fitted archetype rows to Cheetah/Fox/... labels via cosine similarity."""
    seeds = seed_weights or ARCHETYPE_SCORE_WEIGHTS
    names = list(seeds.keys())
    feats = list(feature_names) if feature_names is not None else []
    if not feats:
        # Use union of seed feature keys in stable order.
        feats = sorted({f for w in seeds.values() for f in w})
    seed_mat = np.zeros((len(names), len(feats)), dtype=np.float64)
    for i, arch in enumerate(names):
        for j, f in enumerate(feats):
            seed_mat[i, j] = float(seeds[arch].get(f, 0.0))
    Z = np.asarray(Z, dtype=np.float64)
    # Align Z columns if wider/narrower: take min width.
    p = min(Z.shape[1], seed_mat.shape[1])
    Za = Z[:, :p]
    Sa = seed_mat[:, :p]
    # Cosine similarity.
    Zn = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-12)
    Sn = Sa / (np.linalg.norm(Sa, axis=1, keepdims=True) + 1e-12)
    sim = Zn @ Sn.T  # (n_arch, n_seed)
    used: set[int] = set()
    out: list[str] = []
    for i in range(Za.shape[0]):
        order = np.argsort(-sim[i])
        chosen = None
        for j in order:
            if int(j) not in used:
                chosen = int(j)
                break
        if chosen is None:
            chosen = int(order[0])
        used.add(chosen)
        out.append(names[chosen])
    return out


def choose_k(
    X_std: np.ndarray,
    ks: range | Sequence[int] = range(3, 9),
) -> dict[int, dict[str, float]]:
    """AA RSS elbow + GMM BIC + silhouette table for appendix."""
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    X = np.asarray(X_std, dtype=np.float64)
    table: dict[int, dict[str, float]] = {}
    for k in ks:
        kk = int(k)
        if kk < 2 or kk >= X.shape[0]:
            continue
        row: dict[str, float] = {}
        try:
            alpha, arch = fit_composition(X, k=kk, method="aa")
            recon = alpha @ arch
            # When NMF fallback returns H with different geometry, clip shapes.
            if recon.shape != X.shape:
                # Reconstruct via NNLS-style projection onto archetype rows.
                recon = alpha @ arch[:, : X.shape[1]] if arch.shape[1] >= X.shape[1] else alpha @ np.pad(
                    arch, ((0, 0), (0, X.shape[1] - arch.shape[1]))
                )
            row["rss"] = float(np.sum((X - recon[:, : X.shape[1]]) ** 2))
        except Exception:
            row["rss"] = float("nan")
        try:
            gmm = GaussianMixture(
                n_components=kk, covariance_type="diag", random_state=0, n_init=3
            )
            gmm.fit(X)
            row["bic"] = float(gmm.bic(X))
            labels = gmm.predict(X)
            if len(set(labels)) > 1:
                row["silhouette"] = float(silhouette_score(X, labels))
            else:
                row["silhouette"] = float("nan")
        except Exception:
            row["bic"] = float("nan")
            row["silhouette"] = float("nan")
        table[kk] = row
    return table


def select_k_from_table(
    table: dict[int, dict[str, float]],
    *,
    locked_k: int = 5,
) -> int:
    """Pick appendix k by min GMM BIC (else min RSS); ties prefer locked_k."""

    def _finite_ks(key: str) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        for k, row in table.items():
            try:
                v = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                out.append((v, int(k)))
        return out

    bic_ks = _finite_ks("bic")
    pool = bic_ks or _finite_ks("rss")
    if not pool:
        return int(locked_k)
    best_val = min(v for v, _ in pool)
    tied = sorted(k for v, k in pool if v == best_val)
    if int(locked_k) in tied:
        return int(locked_k)
    return int(tied[0])


def bootstrap_ari(
    X_std: np.ndarray,
    *,
    k: int = 5,
    n_boot: int = 50,
    frac: float = 0.8,
    seed: int = 0,
    method: str = "aa",
) -> dict[str, Any]:
    """Bootstrap adjusted Rand on hard composition labels (appendix stability)."""
    X = np.asarray(X_std, dtype=np.float64)
    n = int(X.shape[0]) if X.ndim == 2 else 0
    if n < 8 or X.ndim != 2:
        return {"status": "skipped", "reason": "too_few_cells", "n": n}
    from sklearn.metrics import adjusted_rand_score

    kk = int(min(max(int(k), 2), n))
    alpha_full, _ = fit_composition(X, k=kk, method=method)
    labels_full = np.argmax(alpha_full, axis=1)
    n_draw = int(max(kk + 1, int(frac * n)))
    n_draw = min(n_draw, n)
    scores: list[float] = []
    for b in range(int(n_boot)):
        rng = np.random.default_rng(int(seed) + b)
        idx = rng.choice(n, size=n_draw, replace=False)
        try:
            alpha_b, _ = fit_composition(X[idx], k=kk, method=method)
            labels_b = np.argmax(alpha_b, axis=1)
            scores.append(float(adjusted_rand_score(labels_full[idx], labels_b)))
        except Exception:
            continue
    if not scores:
        return {"status": "skipped", "reason": "all_bootstraps_failed", "n": n, "k": kk}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "status": "ok",
        "ari_mean": float(np.mean(arr)),
        "ari_std": float(np.std(arr)),
        "n_boot_ok": int(arr.size),
        "k": kk,
        "frac": float(frac),
    }


def rows_to_zmatrix(
    rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Build a z-scored panel matrix from score rows (public wrapper)."""
    feats = list(feature_names)
    X = np.zeros((len(rows), len(feats)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, f in enumerate(feats):
            try:
                v = float(row.get(f, 0.0))
            except (TypeError, ValueError):
                v = 0.0
            X[i, j] = v if np.isfinite(v) else 0.0
    return _zscore(X)


def composition_for_rows(
    rows: Sequence[Mapping[str, float]],
    *,
    feature_names: Sequence[str] | None = None,
    k: int = DEFAULT_K,
    method: str = "aa",
) -> list[dict[str, Any]]:
    """Z-score a panel of score rows and stamp composition fields on each."""
    if not rows:
        return []
    if feature_names is None:
        # Prefer union of known behaviour + exposure keys present in rows.
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key.startswith("_"):
                    continue
                if key not in seen and np.isfinite(float(row.get(key, float("nan")))):
                    # Keep all finite numeric keys.
                    try:
                        float(row[key])
                    except (TypeError, ValueError):
                        continue
                    seen.add(key)
                    keys.append(key)
        feature_names = keys
    feats = list(feature_names)
    X = np.zeros((len(rows), len(feats)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, f in enumerate(feats):
            try:
                X[i, j] = float(row.get(f, 0.0))
            except (TypeError, ValueError):
                X[i, j] = 0.0
            if not np.isfinite(X[i, j]):
                X[i, j] = 0.0
    Xz = _zscore(X)
    kk = min(int(k), max(2, Xz.shape[0]))
    alpha, archetypes = fit_composition(Xz, k=kk, method=method)
    names = name_archetypes(archetypes, feature_names=feats)
    # Ensure unique names even if cosine collided.
    if len(set(names)) < len(names):
        names = [f"{n}_{i}" if names.count(n) > 1 else n for i, n in enumerate(names)]
    # Prefer canonical five when k matches.
    if kk == len(ARCHETYPE_SCORE_WEIGHTS) and len(set(names)) == kk:
        pass
    stamped: list[dict[str, Any]] = []
    for i in range(alpha.shape[0]):
        comp = {names[j]: float(alpha[i, j]) for j in range(alpha.shape[1])}
        # Renormalize tiny drift.
        s = sum(comp.values()) or 1.0
        comp = {k_: v / s for k_, v in comp.items()}
        primary = max(comp, key=comp.get)
        stamped.append(
            {
                "archetype_composition": comp,
                "archetype_primary": primary,
                "archetype_confidence": float(comp[primary]),
                "composition_feature_names": list(feats),
            }
        )
    return stamped
