"""Kelly grid schema and surface signal name catalog."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# OptionMetrics standardized delta-tenor grid (Kelly 2026 / OM volsurfd).
KELLY_TENORS: tuple[int, ...] = (10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730)
KELLY_DELTAS_PUT: tuple[int, ...] = tuple(range(-90, -5, 5))  # 17
KELLY_DELTAS_CALL: tuple[int, ...] = tuple(range(10, 95, 5))  # 17
GRID_POINTS_PER_DAY: int = len(KELLY_TENORS) * (
    len(KELLY_DELTAS_PUT) + len(KELLY_DELTAS_CALL)
)  # 374


def validate_kelly_grid_schema(
    *,
    tenors: Sequence[int] = KELLY_TENORS,
    deltas_put: Sequence[int] = KELLY_DELTAS_PUT,
    deltas_call: Sequence[int] = KELLY_DELTAS_CALL,
    cube_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Assert Kelly axes match OM delta-tenor nodes; optional cube shape check.

    Returns a small metadata dict for logging / tests. Raises ``ValueError``
    on schema mismatch.
    """
    tenors_t = tuple(int(t) for t in tenors)
    d_put = tuple(int(d) for d in deltas_put)
    d_call = tuple(int(d) for d in deltas_call)
    if tenors_t != KELLY_TENORS:
        raise ValueError(f"Kelly tenors must equal OM grid, got {tenors_t}")
    if d_put != KELLY_DELTAS_PUT:
        raise ValueError(f"Kelly put deltas must equal OM grid, got {d_put}")
    if d_call != KELLY_DELTAS_CALL:
        raise ValueError(f"Kelly call deltas must equal OM grid, got {d_call}")
    n_del = len(d_put) + len(d_call)
    expected = (len(tenors_t), n_del)
    if cube_shape is not None:
        if len(cube_shape) != 4:
            raise ValueError(f"Kelly cube must be (T,K,11,34), got shape={cube_shape}")
        if cube_shape[2:] != expected:
            raise ValueError(
                f"Kelly cube tenor/delta axes {cube_shape[2:]} != {expected}"
            )
    return {
        "n_tenors": len(tenors_t),
        "n_deltas": n_del,
        "grid_points_per_day": len(tenors_t) * n_del,
        "ffill_axis": "date",
        "ffill_causal": True,
    }


CW_DAYS: tuple[int, ...] = (30, 60, 91)
CW_DELTAS: tuple[int, ...] = (20, 25, 30, 40, 50)

SURFACE_SIGNAL_NAMES: tuple[str, ...] = (
    "iv_skew_30d",
    "iv_term_slope",
    "iv_convexity_30d",
    "cw_vol_spread",
    "vmp",
    "mfiv_30",
    "mfis_30",
    "mfik_30",
    "mfiv_365",
    "mfis_365",
    "mfik_365",
    "rns_term_spread",
    "svix2_30",
    "mw_xs",
    "d_iv_call_1m",
    "d_iv_put_1m",
    "surface_dispersion",
    "surface_quality",
    "os_ratio",
    "borrow_rate",
    "d_iv_term_slope_5d",
    "d_iv_skew_5d",
    "vrp_30",
)


_NAN = float("nan")

