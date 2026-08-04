"""Holdings vs RBSA style-vector cosine. Interpretation only. Never feeds capital gates.

RBSA loadings from fit_rbsa are simplex-constrained and non-negative.
Holdings exposures are signed characteristic means. Cosine is a diagnostic,
not a proof the two models contradict.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

STYLE_DISAGREE_COSINE = 0.2
_EPS = 1e-12
_MOM_ALIASES = ("mom", "umd")


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _rbsa_style_vector(loadings: Sequence[float], names: Sequence[str]) -> np.ndarray:
    lookup = {str(n).strip().lower(): _f(v) for n, v in zip(names, loadings)}
    mom = float("nan")
    for alias in _MOM_ALIASES:
        if alias in lookup and np.isfinite(lookup[alias]):
            mom = lookup[alias]
            break
    return np.array(
        [lookup.get("smb", float("nan")), lookup.get("hml", float("nan")), mom],
        dtype=np.float64,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return float("nan")
    aa = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    bb = np.nan_to_num(np.asarray(b, dtype=np.float64), nan=0.0)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < _EPS or nb < _EPS:
        return float("nan")
    return float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))


def style_agreement(
    exposures: Mapping[str, float],
    rbsa: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rbsa = dict(rbsa or {})
    loadings = list(rbsa.get("rbsa_loadings") or [])
    names = list(rbsa.get("factor_names") or [])
    hold = np.array(
        [
            -_f(exposures.get("exposure_size")),
            _f(exposures.get("exposure_value")),
            _f(exposures.get("exposure_momentum")),
        ],
        dtype=np.float64,
    )
    if not loadings or not names:
        return {
            "style_agreement_cosine": float("nan"),
            "style_disagreement_flag": False,
            "holdings_style_vec": hold.tolist(),
            "rbsa_style_vec": [float("nan"), float("nan"), float("nan")],
            "reason": "rbsa_unavailable",
        }
    rvec = _rbsa_style_vector(loadings, names)
    cos = _cosine(hold, rvec)
    flag = bool(np.isfinite(cos) and float(cos) < STYLE_DISAGREE_COSINE)
    reason = ""
    if not np.isfinite(cos):
        reason = "style_vector_unavailable"
    return {
        "style_agreement_cosine": float(cos) if np.isfinite(cos) else float("nan"),
        "style_disagreement_flag": flag,
        "holdings_style_vec": hold.tolist(),
        "rbsa_style_vec": rvec.tolist(),
        "reason": reason,
    }
