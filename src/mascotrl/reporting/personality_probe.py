"""DIAYN-style discriminability probe for behaviour feature panels.

Interpretation only. Never feeds capital gates.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def discriminability_probe(
    X: np.ndarray,
    configs: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str] = ("weight_head", "objective"),
) -> dict[str, Any]:
    """Logistic regression CV accuracy vs majority baseline + coef importances."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    out: dict[str, Any] = {"n_cells": int(n), "targets": {}}
    if n < 4:
        out["reason"] = "too_few_cells"
        return out

    for target in targets:
        labels = [str(c.get(target) or "") for c in configs]
        if len(set(labels)) < 2:
            out["targets"][target] = {
                "status": "skipped",
                "reason": "single_class",
            }
            continue
        enc = LabelEncoder()
        y = enc.fit_transform(labels)
        # Majority baseline.
        counts = np.bincount(y)
        baseline = float(counts.max() / n)
        n_splits = int(min(5, n, counts.min()))
        if n_splits < 2:
            out["targets"][target] = {
                "status": "skipped",
                "reason": "too_few_per_class",
                "baseline_accuracy": baseline,
            }
            continue
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        try:
            scores = cross_val_score(clf, X, y, cv=n_splits)
            acc = float(np.mean(scores))
        except Exception as exc:  # noqa: BLE001
            out["targets"][target] = {
                "status": "error",
                "error": str(exc)[:200],
                "baseline_accuracy": baseline,
            }
            continue
        clf.fit(X, y)
        coef = np.asarray(clf.coef_, dtype=np.float64)
        importance = np.mean(np.abs(coef), axis=0)
        out["targets"][target] = {
            "status": "ok",
            "accuracy": acc,
            "baseline_accuracy": baseline,
            "lift": float(acc - baseline),
            "classes": list(enc.classes_),
            "coef_abs_mean": [float(x) for x in importance],
        }
    return out
