"""Per-rebalance decision trace for bot-to-archetype reasoning (interpretation only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mascotrl.reporting.behavior_metrics import SLEEVE_IDS, sleeve_tilt_series


def build_decision_trace_rows(
    *,
    dates: Sequence,
    weights: np.ndarray,
    turnovers: Sequence[float] | None = None,
    sleeve_matrix: np.ndarray | None = None,
    regimes: Sequence[str] | np.ndarray | None = None,
    turnover_cap: float | None = None,
) -> list[dict[str, Any]]:
    """One row per rebalance with projected weights and sleeve tilts."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    n = int(w.shape[0])
    to = list(turnovers or [float("nan")] * n)
    if len(to) < n:
        to = list(to) + [float("nan")] * (n - len(to))
    cap = float(turnover_cap) if turnover_cap is not None else None
    tilts_by_sleeve: dict[str, np.ndarray] = {}
    if sleeve_matrix is not None:
        sm = np.asarray(sleeve_matrix, dtype=np.float64)
        tilt_mat = sleeve_tilt_series(w, sm)
        for j, sid in enumerate(SLEEVE_IDS):
            if j < tilt_mat.shape[1]:
                tilts_by_sleeve[sid] = tilt_mat[:, j]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        prev = w[i - 1] if i > 0 else w[i]
        delta = w[i] - prev
        tv = float(to[i]) if i < len(to) else float("nan")
        row: dict[str, Any] = {
            "date": str(dates[i]) if i < len(dates) else str(i),
            "delta_w": [float(x) for x in delta.reshape(-1)],
            "projected_w": [float(x) for x in w[i].reshape(-1)],
            "turnover": tv,
            "turnover_cap_binding": (
                bool(cap is not None and np.isfinite(tv) and tv >= cap - 1e-9)
                if cap is not None
                else False
            ),
        }
        if regimes is not None and i < len(regimes):
            row["regime_label"] = str(regimes[i])
        if tilts_by_sleeve:
            row["sleeve_tilts"] = {
                sid: float(tilts_by_sleeve[sid][i]) for sid in tilts_by_sleeve
            }
        rows.append(row)
    return rows


def write_decision_trace(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Append-friendly JSONL: one decision record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), default=str) + "\n")
    return path
