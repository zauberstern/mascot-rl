"""Macro cube fioracle wiring (lighter integration)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from src.data.fioracle_macro import FIORACLE_FEATURE_COLUMNS
from src.data.macro_loader import attach_fioracle_macro_cube, fioracle_cfg_from_feature_extras, load_macro_tensor, load_macro_tensor_with_fioracle
ROOT = Path(__file__).resolve().parents[1]
CRUCIBLE_YAML = ROOT / 'config' / 'workflows' / 'eq_alloc_crucible_k100.yaml'
CRUCIBLE_NOFIO_YAML = ROOT / 'config' / 'workflows' / 'eq_alloc_crucible_k100_nofioracle.yaml'

def _write_series(dest: Path, series_id: str, dates, values, lag_days: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ev = pd.to_datetime(dates)
    avail = ev + pd.to_timedelta(lag_days, unit='D')
    df = pd.DataFrame({'event_date': ev.date, 'available_date': avail.date, 'value': values, 'series_id': series_id, 'source_file': f'fixture/{series_id}.csv', 'source_sha256': 'x', 'lag_days': np.int32(lag_days)})
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dest / f'{series_id}.parquet', compression='zstd')

@pytest.fixture
def lake_with_fioracle(tmp_path: Path) -> Path:
    lake = tmp_path
    fio = lake / 'macro' / 'fioracle'
    dates = pd.bdate_range('2018-01-01', periods=320)
    rng = np.random.default_rng(1)
    for sid, lag, base in [('vix', 1, 15.0), ('hy_oas', 1, 3.5), ('term_spread', 0, 0.5), ('epu', 7, 100.0), ('gpri', 5, 90.0), ('unemployment', 7, 4.0), ('inflation', 15, 2.0), ('yield_2y', 0, 2.0)]:
        vals = base + rng.normal(0, 0.05, size=len(dates))
        _write_series(fio, sid, [d.strftime('%Y-%m-%d') for d in dates], vals.tolist(), lag)
    return lake

def _base_macro_frame(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({'vix': np.linspace(12.0, 18.0, len(idx)), 'sofr': np.zeros(len(idx))}, index=idx)

def test_fioracle_enabled_gains_feature_columns(lake_with_fioracle: Path, monkeypatch):
    from src.data import macro_loader
    start, end = ('2018-06-01', '2019-03-01')
    base = _base_macro_frame(start, end)
    monkeypatch.setattr(macro_loader, 'ArcticStateStore', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no arctic')))
    monkeypatch.setattr(macro_loader, '_duckdb_macro_frame', lambda *a, **k: base.copy())
    n_base = base.select_dtypes(include=[np.number]).shape[1]
    n_fio = len(FIORACLE_FEATURE_COLUMNS)
    macro_dim = n_base + n_fio
    t_off, meta_off = load_macro_tensor(lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, macro_dim=macro_dim, prefer_arctic=False, fioracle_enabled=False, return_meta=True)
    t_on, meta_on = load_macro_tensor(lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, macro_dim=macro_dim, prefer_arctic=False, fioracle_enabled=True, fioracle_lake_subdir='macro/fioracle', return_meta=True)
    assert len(meta_on['macro_column_order']) == len(meta_off['macro_column_order']) + n_fio
    assert t_on.shape[1] == macro_dim
    assert torch.isfinite(t_on).all()
    _, meta_on2 = load_macro_tensor(lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, macro_dim=macro_dim, prefer_arctic=False, fioracle_enabled=True, fioracle_lake_subdir='macro/fioracle', return_meta=True)
    assert meta_on['macro_column_order'] == meta_on2['macro_column_order']

def test_disabled_matches_baseline(lake_with_fioracle: Path, monkeypatch):
    from src.data import macro_loader
    start, end = ('2018-06-01', '2019-03-01')
    base = _base_macro_frame(start, end)
    monkeypatch.setattr(macro_loader, 'ArcticStateStore', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no arctic')))
    monkeypatch.setattr(macro_loader, '_duckdb_macro_frame', lambda *a, **k: base.copy())
    a = load_macro_tensor(lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, macro_dim=8, prefer_arctic=False, fioracle_enabled=False)
    b = load_macro_tensor(lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, macro_dim=8, prefer_arctic=False)
    assert torch.equal(a, b)

def test_with_fioracle_helper(lake_with_fioracle: Path, monkeypatch):
    from src.data import macro_loader
    start, end = ('2018-06-01', '2019-03-01')
    base = _base_macro_frame(start, end)
    monkeypatch.setattr(macro_loader, 'ArcticStateStore', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no arctic')))
    monkeypatch.setattr(macro_loader, '_duckdb_macro_frame', lambda *a, **k: base.copy())
    tensor, meta = load_macro_tensor_with_fioracle(lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, macro_dim=2 + len(FIORACLE_FEATURE_COLUMNS), prefer_arctic=False, fioracle_lake_subdir='macro/fioracle')
    assert 'macro_column_order' in meta
    assert torch.isfinite(tensor).all()

def test_attach_fioracle_macro_cube_records_order_and_regimes(lake_with_fioracle: Path, tmp_path: Path, monkeypatch):
    from src.data import macro_loader
    from src.features.blocks.assemble import assemble_equity_feature_cube
    start, end = ('2018-06-01', '2019-03-01')
    base = _base_macro_frame(start, end)
    monkeypatch.setattr(macro_loader, 'ArcticStateStore', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no arctic')))
    monkeypatch.setattr(macro_loader, '_duckdb_macro_frame', lambda *a, **k: base.copy())
    dates = pd.bdate_range(start, end)
    cfg = {'macro_dim': 2 + len(FIORACLE_FEATURE_COLUMNS), 'feature_extras': {'fioracle_macro': {'enabled': True, 'lake_subdir': 'macro/fioracle'}}}
    meta = attach_fioracle_macro_cube(cfg, lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, dates=dates, out_dir=tmp_path)
    assert meta['fioracle_enabled'] is True
    order = meta['macro_column_order']
    assert any((c.startswith('vix_') or c == 'vix_level' for c in order))
    for col in FIORACLE_FEATURE_COLUMNS:
        assert col in order
    extras = cfg['feature_extras']
    assert extras['macro'].shape == (len(dates), len(order))
    assert extras['macro_names'] == order
    regime_path = tmp_path / 'regime_labels.parquet'
    assert regime_path.is_file()
    assert meta.get('regime_labels_path') == str(regime_path)
    rets = np.random.default_rng(0).normal(0, 0.01, size=(len(dates), 4))
    cube, names = assemble_equity_feature_cube(rets, extras=extras, normalize=False)
    assert any((n.startswith('macro_') or n in order for n in names))
    assert cube.shape[0] == len(dates)
    assert cube.shape[1] == 4

def test_attach_ablation_skips_fioracle_columns(lake_with_fioracle: Path, tmp_path: Path, monkeypatch):
    from src.data import macro_loader
    start, end = ('2018-06-01', '2019-03-01')
    base = _base_macro_frame(start, end)
    monkeypatch.setattr(macro_loader, 'ArcticStateStore', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no arctic')))
    monkeypatch.setattr(macro_loader, '_duckdb_macro_frame', lambda *a, **k: base.copy())
    cfg = {'macro_dim': 8, 'feature_extras': {'fioracle_macro': {'enabled': False, 'lake_subdir': 'macro/fioracle'}}}
    meta = attach_fioracle_macro_cube(cfg, lake_base_dir=lake_with_fioracle, start_date=start, end_date=end, dates=pd.bdate_range(start, end), out_dir=tmp_path)
    assert meta['fioracle_enabled'] is False
    assert 'macro' not in (cfg.get('feature_extras') or {})
    assert not (tmp_path / 'regime_labels.parquet').exists()
    for col in FIORACLE_FEATURE_COLUMNS:
        assert col not in (meta.get('macro_column_order') or [])
