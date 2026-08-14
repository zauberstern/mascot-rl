"""Literature-backed IV surface signals for equity allocation (Phase E).

Pure constructors operate on a standardized OptionMetrics-style surface table
(columns: secid, date, days, delta, cp_flag, impl_volatility, impl_strike,
impl_premium, dispersion) without requiring the lake. Lake materialization is
a thin DuckDB pass over ``vol_surface`` partitions.
"""
from __future__ import annotations

from mascotrl.data.surface_signals_cache import (
    align_signals_to_panel,
    align_signals_to_slots,
    cache_surface_signals,
    load_surface_signals,
    surface_signals_cache_fingerprint,
)
from mascotrl.data.surface_signals_compute import (
    build_kelly_iv_images,
    compute_surface_signals_panel,
)
from mascotrl.data.surface_signals_extract import (
    bkm_moment_failure_count,
    bkm_moment_last_failure_reason,
    extract_grid_point,
    reset_bkm_moment_failure_counter,
)
from mascotrl.data.surface_signals_grid import (
    GRID_POINTS_PER_DAY,
    KELLY_DELTAS_CALL,
    KELLY_DELTAS_PUT,
    KELLY_TENORS,
    SURFACE_SIGNAL_NAMES,
    validate_kelly_grid_schema,
)
from mascotrl.data.surface_signals_lake import (
    materialize_kelly_iv_images_from_lake,
    materialize_surface_signals_from_lake,
)

# Private re-exports for tests and internal callers.
from mascotrl.data.surface_signals_cache import _canonical_secid_key  # noqa: F401
from mascotrl.data.surface_signals_extract import _mf_moments_at_days  # noqa: F401
from mascotrl.data.surface_signals_lake import _load_vol_surface_raw  # noqa: F401

