"""Phase C: ArcticDB LMDB feature store with bitemporal Point-in-Time reads."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
from arcticdb import Arctic

from mascotrl.data.paths import ARCTIC_ROOT


class ArcticStateStore:
    """
    Bitemporal feature store.

    - *Event time*: DatetimeIndex on each symbol (trading / observation calendar).
    - *Knowledge time*: ArcticDB library revision timestamp, queried via ``as_of``.

    Restatements (e.g. revised NFP) must be written as new versions; readers at an
    earlier knowledge time must not see later revisions. Calendar ffill reflects
    what a live desk would observe — never Gaussian imputation.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        library_name: str = "hyper_volanet_features",
    ):
        self.db_path = Path(db_path) if db_path else ARCTIC_ROOT
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.arctic = Arctic(f"lmdb://{self.db_path.as_posix()}")
        if library_name not in self.arctic.list_libraries():
            self.arctic.create_library(library_name)
        self.lib = self.arctic[library_name]

    def persist_features(
        self, symbol: str, arrow_table: pa.Table, metadata: Optional[dict] = None
    ) -> None:
        df = arrow_table.to_pandas()
        for col in ("date", "Date", "DATE"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{symbol}: expected a date column or DatetimeIndex")
        # Duplicate event-time rows poison reindex/ffill ("cannot reindex ... duplicate labels").
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")].sort_index()
        meta = dict(metadata or {})
        meta.setdefault("knowledge_written_at", pd.Timestamp.utcnow().isoformat())
        if symbol in self.lib.list_symbols():
            # ArcticDB update cannot freely expand schemas (e.g. K=10 → K=50).
            # Rewrite when column sets diverge.
            try:
                existing = self.lib.read(symbol).data
                if set(existing.columns) != set(df.columns):
                    self.lib.delete(symbol)
                    self.lib.write(symbol, df, metadata=meta)
                    return
            except Exception:
                self.lib.delete(symbol)
                self.lib.write(symbol, df, metadata=meta)
                return
            self.lib.update(symbol, df, metadata=meta)
        else:
            self.lib.write(symbol, df, metadata=meta)

    def persist_panel(
        self,
        symbol: str,
        df: pd.DataFrame,
        metadata: Optional[dict] = None,
    ) -> None:
        """Persist a long (date, secid) panel. Does not drop duplicate dates."""
        if "p3" in str(symbol).lower() or "worldscope" in str(symbol).lower():
            raise ValueError(f"P3 Arctic symbol refused: {symbol}")
        out = df.copy()
        if "date" not in out.columns or "secid" not in out.columns:
            raise KeyError("persist_panel requires date and secid columns")
        out["date"] = pd.to_datetime(out["date"])
        out["secid"] = pd.to_numeric(out["secid"], errors="coerce")
        for col in out.columns:
            if col in ("date", "secid"):
                continue
            if pd.api.types.is_extension_array_dtype(out[col].dtype):
                if pd.api.types.is_numeric_dtype(out[col].dtype):
                    out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
                else:
                    out[col] = out[col].astype("object")
        indexed = out.set_index(["date", "secid"]).sort_index()
        if not indexed.index.is_unique:
            raise ValueError(f"{symbol}: duplicate (date, secid) rows")
        meta = dict(metadata or {})
        meta.setdefault("knowledge_written_at", pd.Timestamp.utcnow().isoformat())
        meta.setdefault("not_om_sot", True)
        meta.setdefault("index", "date_secid")
        if symbol in self.lib.list_symbols():
            self.lib.delete(symbol)
        self.lib.write(symbol, indexed, metadata=meta)

    def read_pit_state(
        self,
        symbol: str,
        as_of: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """
        Knowledge-time PIT: only versions written at-or-before ``as_of``.

        Layer-5 law: ``as_of`` should be explicit for live/WFO reads. ``None``
        means latest revision (offline rebuild after a single ETL write).
        """
        if as_of is None:
            return self.lib.read(symbol).data
        return self.lib.read(symbol, as_of=pd.Timestamp(as_of)).data

    def read_ffill(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        as_of: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """
        Bitemporal read:
          1) knowledge PIT via ArcticDB ``as_of``
          2) event-time window [start, end]
          3) strict forward-fill only (no bfill, no zeros, no randn)

        Raises if the result is empty after ffill + dropna.
        """
        df = self.read_pit_state(symbol, as_of=as_of)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{symbol}: expected DatetimeIndex, got {type(df.index)}")
        df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")].sort_index()
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        hist = df.loc[:end_ts].copy()
        if hist.empty:
            raise KeyError(
                f"{symbol}: no event-time rows at-or-before {end_ts} "
                f"(as_of/knowledge_time={as_of})"
            )
        if hist.index.has_duplicates:
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        cal = pd.date_range(start_ts, end_ts, freq="B")
        aligned = hist.reindex(hist.index.union(cal)).sort_index().ffill()
        out = aligned.loc[start_ts:end_ts].ffill()
        # Rate columns (SOFR/EFFR/DTB3) may be null before series inception.
        # Do not truncate the VIX calendar via dropna(how="any") on those cols.
        _RATE_COLS = ("sofr", "effr", "dtb3")
        rate_present = [c for c in _RATE_COLS if c in out.columns]
        if rate_present:
            core = out.drop(columns=rate_present)
            core = core.dropna(how="any")
            if core.empty:
                raise KeyError(
                    f"{symbol}: empty after strict ffill in [{start_ts}, {end_ts}] "
                    f"(as_of={as_of})"
                )
            rates = out.loc[core.index, rate_present].ffill().fillna(0.0)
            out = core.join(rates)
        else:
            out = out.dropna(how="any")
        if out.empty:
            raise KeyError(
                f"{symbol}: empty after strict ffill in [{start_ts}, {end_ts}] "
                f"(as_of={as_of})"
            )
        return out

    def list_available_features(self) -> list:
        return self.lib.list_symbols()


def get_pit_macro_features(
    store: ArcticStateStore,
    target_timestamp: pd.Timestamp | str,
    *,
    symbol: str = "macro_state",
    as_of: pd.Timestamp | str | None = ...,  # type: ignore[assignment]
    lookback_days: int = 1,
) -> pd.DataFrame:
    """
    Fetch macro exactly as known to the market at ``target_timestamp``.

    Parameters
    ----------
    target_timestamp
        Event time (trading desk / observation date).
    as_of
        Knowledge time (ArcticDB revision). Default (= ``target_timestamp``)
        hides restatements published later. Pass ``None`` to force the latest
        library revision (offline single-write lakes) while still clipping
        event time at ``target_timestamp``.

    Strict ffill; raises ``RuntimeError`` on total failure (never ``torch.randn``).
    """
    event_ts = pd.Timestamp(target_timestamp)
    # Sentinel: Ellipsis → default knowledge = event clock (restatement shield).
    if as_of is ...:
        knowledge: pd.Timestamp | None = event_ts
        allow_latest_fallback = True
    elif as_of is None:
        knowledge = None
        allow_latest_fallback = False
    else:
        knowledge = pd.Timestamp(as_of)
        allow_latest_fallback = False

    start = event_ts - pd.tseries.offsets.BDay(max(int(lookback_days), 0))
    try:
        return store.read_ffill(symbol, start=start, end=event_ts, as_of=knowledge)
    except Exception as exc:
        if allow_latest_fallback:
            # Offline lakes: write clock is "now", so as_of=event_ts may miss.
            # Latest revision + event-time ffill still blocks future calendar rows.
            try:
                return store.read_ffill(
                    symbol, start=start, end=event_ts, as_of=None
                )
            except Exception as exc2:
                raise RuntimeError(
                    f"Critical Macro Feature Failure at event={event_ts}: {exc2}"
                ) from exc2
        raise RuntimeError(
            f"Critical Macro Feature Failure at event={event_ts} as_of={knowledge}: {exc}"
        ) from exc
