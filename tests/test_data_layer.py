import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from tests.conftest import FLOAT_TOL
from pathlib import Path
from mascotrl.data.arctic_store import ArcticStateStore, get_pit_macro_features
from mascotrl.data.oos_panel import SIGNALS_SYMBOL, load_oos_panel, pivot_long_marks_to_wide

def test_duckdb_arrow_roundtrip(tmp_path: Path):
    con = duckdb.connect(':memory:')
    con.execute('CREATE TABLE t AS SELECT 1 AS a, 2.0::DOUBLE AS b')
    result = con.execute('SELECT * FROM t')
    table = result.fetch_arrow_table() if hasattr(result, 'fetch_arrow_table') else result.arrow()
    if not isinstance(table, pa.Table):
        table = table.read_all()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 1

def test_macro_read_ffill_bday_union_length():
    """5561 arctic rows → 5741 after B-day∪CBOE expand; no leading invent fill."""
    from mascotrl.data.paths import ARCTIC_ROOT, LAKE_ROOT
    from mascotrl.data.arctic_store import ArcticStateStore
    if not (LAKE_ROOT / 'macro' / 'cboe_vix.parquet').is_file():
        pytest.skip('lake not mounted')
    store = ArcticStateStore(db_path=ARCTIC_ROOT, library_name='hyper_volanet_features')
    if 'macro_state' not in store.list_available_features():
        pytest.skip('arctic macro_state missing')
    raw = store.read_pit_state('macro_state', as_of=None)
    assert not raw.index.has_duplicates
    ff = store.read_ffill('macro_state', start='2003-01-01', end='2024-12-31', as_of=None)
    bdays = pd.bdate_range('2003-01-01', '2024-12-31')
    expected = len(set(raw.index.normalize()) | set(bdays))
    expected -= sum((1 for d in set(raw.index.normalize()) | set(bdays) if d < raw.index.min()))
    assert len(ff) == expected
    assert ff.index.min() >= raw.index.min()
    assert ff['vix'].notna().all() and (ff['vix'] > 0).all()
    assert float(ff['vix'].iloc[0]) > 0

def test_duckdb_macro_dedupes_vix_duplicate_dates():
    """Lake VIX has known dup calendar rows; compute_macro_state must emit unique dates."""
    from mascotrl.data.duckdb_engine import DuckDBFeatureEngine
    from mascotrl.data.paths import LAKE_ROOT
    if not (LAKE_ROOT / 'macro' / 'cboe_vix.parquet').is_file():
        pytest.skip('lake not mounted')
    eng = DuckDBFeatureEngine(lake_base_dir=LAKE_ROOT)
    table = eng.compute_macro_state('2003-01-01', '2003-12-31')
    df = table.to_pandas()
    df['date'] = pd.to_datetime(df['date'])
    assert not df['date'].duplicated().any()
    for d in ('2003-09-22', '2003-10-30'):
        assert (df['date'] == pd.Timestamp(d)).sum() <= 1

def test_arctic_persist_and_list(tmp_path: Path):
    store = ArcticStateStore(db_path=tmp_path / 'arctic', library_name='test_lib')
    table = pa.table({'date': pa.array(['2020-01-01', '2020-01-02']), 'x': [1.0, 2.0]})
    store.persist_features('toy', table)
    assert 'toy' in store.list_available_features()

def test_arctic_persist_dedupes_event_dates(tmp_path: Path):
    """Duplicate event-time rows must not poison later PIT reindex/ffill."""
    store = ArcticStateStore(db_path=tmp_path / 'arctic_dedupe', library_name='test_lib')
    table = pa.table({'date': pa.array(['2020-01-01', '2020-01-01', '2020-01-03']), 'vix': [10.0, 11.0, 12.0]})
    store.persist_features('macro_state', table)
    raw = store.read_pit_state('macro_state', as_of=None)
    assert not raw.index.has_duplicates
    assert float(raw.loc[pd.Timestamp('2020-01-01'), 'vix']) == pytest.approx(11.0, **FLOAT_TOL)
    df = get_pit_macro_features(store, '2020-01-03', as_of=None, lookback_days=2)
    assert float(df.iloc[-1]['vix']) == pytest.approx(12.0, **FLOAT_TOL)

def test_arctic_ffill_gap_no_randn(tmp_path: Path):
    """Calendar gaps must forward-fill last known value — never invent noise."""
    store = ArcticStateStore(db_path=tmp_path / 'arctic_ffill', library_name='test_lib')
    table = pa.table({'date': pa.array(['2020-01-01', '2020-01-03']), 'x': [1.0, 3.0]})
    store.persist_features('macro_state', table)
    df = store.read_ffill('macro_state', start='2020-01-01', end='2020-01-03', as_of=None)
    assert float(df.loc[pd.Timestamp('2020-01-02'), 'x']) == pytest.approx(1.0, **FLOAT_TOL)

def test_get_pit_macro_features_raises_clean(tmp_path: Path):
    store = ArcticStateStore(db_path=tmp_path / 'arctic_empty', library_name='test_lib')
    with pytest.raises(RuntimeError, match='Critical Macro Feature Failure'):
        get_pit_macro_features(store, '2020-01-02', as_of=None)

def test_get_pit_macro_features_ffill(tmp_path: Path):
    store = ArcticStateStore(db_path=tmp_path / 'arctic_pit', library_name='test_lib')
    table = pa.table({'date': pa.array(['2020-01-01', '2020-01-03']), 'vix': [12.0, 14.0]})
    store.persist_features('macro_state', table)
    df = get_pit_macro_features(store, '2020-01-03', as_of=None, lookback_days=5)
    assert float(df.loc[pd.Timestamp('2020-01-02'), 'vix']) == pytest.approx(12.0, **FLOAT_TOL)

def test_macro_loader_refuses_randn_fallback(tmp_path: Path, monkeypatch):
    """If lake + Arctic fail, raise — do not return torch.randn synthetic macro."""
    from mascotrl.data import macro_loader

    def boom(*_a, **_k):
        raise RuntimeError('duckdb down')
    monkeypatch.setattr(macro_loader, '_duckdb_macro_frame', boom)
    monkeypatch.setattr(macro_loader, 'ArcticStateStore', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('arctic down')))
    with pytest.raises(RuntimeError, match='Refusing torch.randn|Critical Macro'):
        macro_loader.load_macro_tensor(lake_base_dir=tmp_path, start_date='2015-01-01', end_date='2015-01-10', macro_dim=8, arctic_db_path=tmp_path / 'no_arctic', prefer_arctic=True, knowledge_time=None)

def test_pivot_and_persist_constituent_signals(tmp_path: Path):
    secids = [101, 202]
    long = pa.table({'secid': [101, 202, 101, 202], 'date': ['2022-01-03', '2022-01-03', '2022-01-04', '2022-01-04'], 'mid': [1.0, 2.0, 1.1, 2.2], 'delta': [0.5, 0.5, 0.5, 0.5], 'atm_iv': [0.2, 0.25, 0.21, 0.26], 'bid_ask_spread': [0.01, 0.02, 0.01, 0.02], 'skew_25d': [0.05, 0.04, 0.05, 0.03], 'spot': [100.0, 50.0, 101.0, 49.0], 'strike': [100.0, 50.0, 100.0, 50.0], 'dh_denom': [49.0, 23.0, 49.4, 22.3], 'dh_denom_lagdelta': [48.5, 22.8, 49.0, 22.0], 'dh_ret': [0.002, -0.001, 0.0, -0.003], 'dh_ret_lagdelta': [0.0015, -0.0012, 0.0, -0.0025], 'fwd_ret': [0.1, 0.2, 0.0, -0.1]})
    wide = pivot_long_marks_to_wide(long, secids)
    assert list(wide.columns)[:2] == ['atm_iv_0', 'atm_iv_1']
    assert float(wide.loc[pd.Timestamp('2022-01-03'), 'fwd_ret_0']) == pytest.approx(0.1, **FLOAT_TOL)
    assert float(wide.loc[pd.Timestamp('2022-01-03'), 'dh_ret_0']) == pytest.approx(0.002, **FLOAT_TOL)
    assert float(wide.loc[pd.Timestamp('2022-01-03'), 'dh_ret_lagdelta_0']) == pytest.approx(0.0015, **FLOAT_TOL)
    store = ArcticStateStore(db_path=tmp_path / 'arctic_oos', library_name='test_lib')
    out = wide.reset_index()
    date_col = 'date' if 'date' in out.columns else out.columns[0]
    out = out.rename(columns={date_col: 'date'})
    store.persist_features(SIGNALS_SYMBOL, pa.Table.from_pandas(out, preserve_index=False))
    loaded, _ = load_oos_panel(store, start='2022-01-03', end='2022-01-04', as_of=None)
    assert SIGNALS_SYMBOL in store.list_available_features()
    assert len(loaded) == 2
