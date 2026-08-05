"""Canonical filesystem paths for Tier A (mounted) and Tier B (local) inputs."""
from __future__ import annotations

import os
from pathlib import Path

MASCOTRL_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = MASCOTRL_ROOT / "data"

# Tier A CSVs alone are ~293GB; ZSTD parquet lake can still be 80–150GB+.
# Desktop root (~100GB free) cannot hold it — lake + Arctic live on the mounted volume.
MOUNT_ROOT = Path(os.environ.get("MASCOTRL_MOUNT_ROOT", "/mnt/volsurf"))
CANONICAL_LAKE = MOUNT_ROOT / "volsurf_data_lake"
CANONICAL_RAW = MOUNT_ROOT / "volsurf_raw"
LAKE_ROOT = Path(
    os.environ.get("MASCOTRL_LAKE_DIR")
    or os.environ.get("MASCOTRL_LAKE_BASE")
    or str(CANONICAL_LAKE)
)
ARCTIC_ROOT = Path(
    os.environ.get("MASCOTRL_ARCTIC_DIR", str(MOUNT_ROOT / "volsurf_arcticdb"))
)
RAW_ROOT = Path(
    os.environ.get("MASCOTRL_RAW_ROOT", str(CANONICAL_RAW))
)

TIER_A_DIR = RAW_ROOT / "om"

TIER_A = {
    "options_panel": TIER_A_DIR / "option_prices_2003_2024_sp500_all.csv",
    "options_slim": TIER_A_DIR / "option_prices_slim_2003_2024.csv",
    "options_constituents": TIER_A_DIR / "option_prices_2003-2024_sp500_all_constituents.csv",
    "options_std": TIER_A_DIR / "std_option_prices_sp500_2003_2024.csv",
    "vol_surface": TIER_A_DIR / "vsurd_sp500_2003_2024.csv",
    "vol_surface_legacy_a": TIER_A_DIR / "constituent_volsurfd(2003-2021).csv",
    "vol_surface_legacy_b": TIER_A_DIR / "constituent_volsurfd(2022-2024).csv",
}

TIER_B = {
    "interest_rate": RAW_ROOT / "macro" / "interest_rate_2003_2024_frb.csv",
    "pastor_stambaugh": RAW_ROOT / "macro" / "pastor_stambaugh.csv",
    "cboe_vix": RAW_ROOT / "macro" / "cboe_vix_vox_2003_2024.csv",
    "sp500_prices": RAW_ROOT / "macro" / "sp500_prices_2003_2024_all_constituents.csv",
    "sp500_hv": RAW_ROOT / "macro" / "sp500_historical_vol_2003_2024_all_constituents.csv",
    "sp500_fwd": RAW_ROOT / "macro" / "sp500_fwd_price_2003_2024_all_constituents.csv",
    "sp500_sec": RAW_ROOT / "macro" / "sp500_sec_prices_optionm_2003_2024_all_constituents.csv",
    "pit_membership": RAW_ROOT / "macro" / "sp500_pit_membership_snapshot.csv",
    "spx_zerocd": RAW_ROOT / "macro" / "SPX_zerocd.csv",
    "spx_opvold": RAW_ROOT / "macro" / "SPX_opvold.csv",
    "spx_index": RAW_ROOT / "macro" / "SPX(2003-2024).csv",
}

UNIVERSE_IDENTIFIERS = RAW_ROOT / "identifiers" / "universe_identifiers.csv"
LSEG_RAW = RAW_ROOT / "lseg"


def tier_a_available() -> bool:
    return TIER_A["options_panel"].exists() and TIER_A["vol_surface"].exists()


def assert_raw_mounted(raw: Path | None = None) -> Path:
    """Fail closed when USB raw dumps are missing. Never mkdir onto root fs."""
    root = Path(raw) if raw is not None else Path(RAW_ROOT)
    try:
        is_canonical = root.resolve() == CANONICAL_RAW.resolve()
    except OSError:
        is_canonical = str(root) == str(CANONICAL_RAW)
    if is_canonical and not MOUNT_ROOT.exists():
        raise SystemExit(
            f"raw mount missing: {MOUNT_ROOT} is not mounted; "
            f"refusing to create {root} on the root filesystem"
        )
    if not root.exists():
        raise SystemExit(
            f"raw root missing: {root}; mount the USB volume or set MASCOTRL_RAW_ROOT"
        )
    return root


def assert_lake_mounted(lake: Path | None = None) -> Path:
    """Fail closed when the data lake is missing or the USB volume is unmounted.

    When the resolved path is the canonical USB lake, require ``MOUNT_ROOT``
    itself to exist. Never mkdir the lake onto the root filesystem as a
    silent fallback.
    """
    root = Path(lake) if lake is not None else Path(LAKE_ROOT)
    try:
        is_canonical = root.resolve() == CANONICAL_LAKE.resolve()
    except OSError:
        is_canonical = str(root) == str(CANONICAL_LAKE)
    if is_canonical and not MOUNT_ROOT.exists():
        raise SystemExit(
            f"lake mount missing: {MOUNT_ROOT} is not mounted; "
            f"refusing to create {root} on the root filesystem"
        )
    if not root.exists():
        raise SystemExit(
            f"lake root missing: {root}; mount the USB volume or set MASCOTRL_LAKE_DIR"
        )
    return root


def ensure_lake_dirs() -> None:
    """Create lake subdirs only after the lake root itself is present."""
    assert_lake_mounted()
    LAKE_ROOT.mkdir(parents=True, exist_ok=True)
    if ARCTIC_ROOT.parent.exists():
        ARCTIC_ROOT.mkdir(parents=True, exist_ok=True)
