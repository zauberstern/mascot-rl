"""Surface signal cache I/O and alignment to panels/slots."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.surface_signals_grid import SURFACE_SIGNAL_NAMES

_LOG = logging.getLogger(__name__)

def cache_surface_signals(panel: pd.DataFrame, path: str | Path) -> None:
    """Write signal panel to parquet."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = panel.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(p, index=False)


def load_surface_signals(path: str | Path) -> pd.DataFrame:
    """Load signal panel from parquet."""
    df = pd.read_parquet(Path(path))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def surface_signals_cache_fingerprint(
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
    signal_names: Sequence[str] | None = None,
) -> str:
    """Sha256 key for shared surface-signal parquet reuse across campaign arms."""
    import hashlib

    names = sorted(str(n) for n in (signal_names if signal_names is not None else SURFACE_SIGNAL_NAMES))
    sec_keys = sorted(_canonical_secid_key(s) for s in secids)
    payload = "\n".join(
        [
            ",".join(sec_keys),
            str(start),
            str(end),
            ",".join(names),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shared_surface_cache_paths(cache_dir: str | Path, fingerprint: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{fingerprint}.parquet", root / f"{fingerprint}.meta.json"


def _load_shared_surface_cache(
    cache_dir: str | Path,
    fingerprint: str,
) -> pd.DataFrame | None:
    """Load fingerprint-keyed panel or return None on miss.

    Fail-closed when a cache artifact exists but is corrupt / mismatched:
    never silently rebuild over a broken shared file.
    """
    import json

    parquet_path, meta_path = _shared_surface_cache_paths(cache_dir, fingerprint)
    if not parquet_path.is_file():
        return None
    if not meta_path.is_file():
        raise RuntimeError(
            f"surface cache corrupt: missing meta for {parquet_path}"
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"surface cache corrupt: unreadable meta {meta_path}: {exc}"
        ) from exc
    if str(meta.get("fingerprint") or "") != str(fingerprint):
        raise RuntimeError(
            f"surface cache corrupt: fingerprint mismatch at {meta_path}"
        )
    try:
        df = load_surface_signals(parquet_path)
    except Exception as exc:
        raise RuntimeError(
            f"surface cache corrupt: unreadable parquet {parquet_path}: {exc}"
        ) from exc
    required = {"secid", "date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"surface cache corrupt: missing columns {sorted(missing)} in {parquet_path}"
        )
    _LOG.info("surface_signals cache hit path=%s rows=%d", parquet_path, len(df))
    return df


def _write_shared_surface_cache(
    cache_dir: str | Path,
    fingerprint: str,
    panel: pd.DataFrame,
) -> None:
    import json

    parquet_path, meta_path = _shared_surface_cache_paths(cache_dir, fingerprint)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    cache_surface_signals(panel, parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_rows": int(len(panel)),
                "columns": list(panel.columns),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _LOG.info("surface_signals cache wrote path=%s rows=%d", parquet_path, len(panel))


def _canonical_secid_key(secid: Any) -> str:
    """Stable string key so ``100892`` and ``100892.0`` align to the same column.

    DuckDB / parquet often yield float secids; equity panels use ints. A naive
    ``str(secid)`` then makes ``align_signals_to_panel`` reindex to all-NaN
    columns and dyn_dii_opt eligibility collapses to zero names.
    """
    if secid is None:
        return "None"
    try:
        if isinstance(secid, (float, np.floating)) and not np.isfinite(secid):
            return str(secid)
        return str(int(secid))
    except (TypeError, ValueError):
        return str(secid)


def align_signals_to_panel(
    signals: pd.DataFrame,
    dates: Sequence[Any],
    secids: Sequence[Any],
    *,
    lag_days: int = 1,
    signal_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Forward-fill month-end signals onto a daily ``(T, K)`` panel, PIT-safe.

    ``signals`` is the long month-end frame from
    :func:`materialize_surface_signals_from_lake` (columns ``secid, date,
    <signal columns>``). Each month-end value becomes visible only
    ``lag_days`` calendar days after the month-end date it describes (a
    publication lag), and then holds (forward-fills) until the next
    release. A daily date ``d`` therefore only ever sees a signal computed
    from a month-end ``<= d - lag_days``, matching the OptionMetrics
    end-of-day availability convention.

    Returns ``{signal_name: (T, K) float64 array}`` aligned to ``dates`` x
    ``secids``, with ``NaN`` where no published value is available yet.
    """
    dates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    t = len(dates_idx)
    k = len(secids)
    names = list(signal_names) if signal_names is not None else list(SURFACE_SIGNAL_NAMES)
    if signals is None or len(signals) == 0:
        return {name: np.full((t, k), np.nan, dtype=np.float64) for name in names}

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    # Normalize secid so float/int parquet variants share one key space.
    sig["secid"] = sig["secid"].map(_canonical_secid_key)
    # Publication lag: a month-end value is not knowable until lag_days later.
    sig["_avail"] = sig["date"] + pd.Timedelta(days=int(lag_days))

    sec_str = [_canonical_secid_key(s) for s in secids]
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in sig.columns:
            out[name] = np.full((t, k), np.nan, dtype=np.float64)
            continue
        wide = sig.pivot_table(index="_avail", columns="secid", values=name, aggfunc="last")
        wide.columns = [_canonical_secid_key(c) for c in wide.columns]
        wide = wide.reindex(columns=sec_str)
        # Union index (avail dates + target dates) then ffill so a value
        # published between two requested dates is still visible on the
        # next requested date, before restricting back to `dates`.
        union_idx = wide.index.union(dates_idx).sort_values()
        wide = wide.reindex(union_idx).ffill()
        wide = wide.reindex(dates_idx)
        out[name] = wide.to_numpy(dtype=np.float64)
    return out


def align_signals_to_slots(
    signals: pd.DataFrame,
    dates: Sequence[Any],
    slots_rows: Sequence[Sequence[Any]],
    *,
    lag_days: int = 1,
    signal_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """PIT-align surface signals onto a *dynamic* slot occupancy schedule.

    Unlike :func:`align_signals_to_panel` (fixed secid identity per column),
    this gathers the published value of whichever secid occupies slot ``k``
    on date ``t``. Inactive slots (``None``) are NaN. Publication lag and
    month-end forward-fill semantics match the static panel helper.
    """
    dates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    t_n = len(dates_idx)
    if len(slots_rows) != t_n:
        raise ValueError(
            f"slots_rows length {len(slots_rows)} != len(dates)={t_n}"
        )
    k = len(slots_rows[0]) if slots_rows else 0
    for i, row in enumerate(slots_rows):
        if len(row) != k:
            raise ValueError(f"slots_rows[{i}] length {len(row)} != k={k}")

    names = list(signal_names) if signal_names is not None else list(SURFACE_SIGNAL_NAMES)
    empty = {name: np.full((t_n, k), np.nan, dtype=np.float64) for name in names}
    if signals is None or len(signals) == 0 or k == 0:
        return empty

    # Universe of every secid that ever occupies a slot.
    all_secids: list[Any] = []
    seen: set[str] = set()
    for row in slots_rows:
        for sid in row:
            if sid is None:
                continue
            key = str(sid)
            if key not in seen:
                seen.add(key)
                all_secids.append(sid)
    if not all_secids:
        return empty

    panel = align_signals_to_panel(
        signals,
        dates_idx,
        all_secids,
        lag_days=lag_days,
        signal_names=names,
    )
    col_of = {_canonical_secid_key(s): i for i, s in enumerate(all_secids)}
    out: dict[str, np.ndarray] = {}
    for name in names:
        wide = panel[name]
        slotted = np.full((t_n, k), np.nan, dtype=np.float64)
        for t, row in enumerate(slots_rows):
            for j, sid in enumerate(row):
                if sid is None:
                    continue
                col = col_of.get(_canonical_secid_key(sid))
                if col is None:
                    continue
                slotted[t, j] = wide[t, col]
        out[name] = slotted
    return out


