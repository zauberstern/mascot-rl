"""Publication rigor: DSR, regimes, baselines, ablation hooks."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
from pathlib import Path
import numpy as np
import torch
from src.eval.baselines import BASELINE_NAMES, baseline_weights_day, _garch11_forecast
from src.eval.publication import plot_publication_figures, write_limitations_section
from src.eval.stats_rigor import deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio, regime_performance_table
from src.features.extractor import AlphaFeatureExtractor
from src.policy.convex_projection import ConvexProjectionLayer
from src.policy.happo import HAPPOEngine

def test_hansen_spa_and_bootstrap_and_wilcoxon():
    from src.eval.stats_rigor import block_bootstrap_metric_ci, hansen_spa_test, wilcoxon_paired_delta
    rng = np.random.default_rng(0)
    bench = rng.normal(0.001, 0.01, size=300)
    weak = bench - 0.002
    strong = bench + 0.002
    spa = hansen_spa_test(bench, {'weak': weak, 'strong': strong}, n_boot=99, seed=0)
    assert spa['ok'] is True
    assert 'pvalue_consistent' in spa
    ci = block_bootstrap_metric_ci(bench, metric='sharpe', n_boot=99, seed=0)
    assert ci['ci_low'] <= ci['point'] <= ci['ci_high']
    w = wilcoxon_paired_delta([1.0, 1.2, 0.8, 1.1, 0.9], [0.5, 0.6, 0.4, 0.55, 0.45])
    assert w['n_pairs'] == 5
    assert w['mean_delta'] > 0

def test_n_trials_breakdown_never_one_silently():
    from src.eval.publication import estimate_n_trials
    n, br = estimate_n_trials({}, cfg={'eval_seeds': '0,1,2,3,4', 'publication_ablation_variants': 5})
    assert n > 1
    assert br['n_trials'] == n
    assert br['n_seeds'] == 5

def test_dsr_increases_with_more_trials():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=500)
    dsr_few = deflated_sharpe_ratio(rets, n_trials=2)
    dsr_many = deflated_sharpe_ratio(rets, n_trials=200)
    assert dsr_few['dsr'] >= dsr_many['dsr'] - 1e-09
    assert 0.0 <= dsr_few['psr'] <= 1.0
    assert 'Bailey' in dsr_few['citation']

def test_psr_zero_benchmark_normal():
    rets = np.full(1000, 0.002)
    sr = float(rets.mean() / (rets.std(ddof=0) + 1e-12))
    psr = probabilistic_sharpe_ratio(sr, 0.0, len(rets), 0.0, 3.0)
    assert psr > 0.95

def test_expected_max_sharpe_grows_with_n():
    a = expected_max_sharpe(2, 0.01)
    b = expected_max_sharpe(50, 0.01)
    assert b > a

def test_regime_table_slices():
    dates = [f'2020-03-{d:02d}' for d in range(1, 28)] + [f'2022-06-{d:02d}' for d in range(1, 20)]
    pnls = [0.01] * len(dates)
    turns = [0.1] * len(dates)
    tab = regime_performance_table(dates, pnls, turns)
    covid = next((r for r in tab['regimes'] if r['id'] == 'covid_2020'))
    hike = next((r for r in tab['regimes'] if r['id'] == 'hike_2022'))
    gfc = next((r for r in tab['regimes'] if r['id'] == 'gfc_2008'))
    assert covid['n_days'] > 0 and covid['available'] is True
    assert hike['n_days'] > 0 and hike['available'] is True
    assert gfc['available'] is False
    assert gfc['status'] == 'unavailable'
    assert 'N/A' in (gfc.get('note') or '')
    assert any(('gfc_2008' in w for w in tab['sanity']['warnings']))

def test_regime_length_mismatch_is_explicit_na():
    tab = regime_performance_table(['2008-10-01', '2008-10-02'], [0.01], [0.1])
    assert all((r['status'] == 'unavailable' for r in tab['regimes']))
    assert any(('mismatch' in w for w in tab['sanity']['warnings']))

def test_attach_uses_historical_calendar_for_gfc():
    from src.eval.publication import attach_publication_stats
    is_dates = [f'2008-10-{d:02d}' for d in range(1, 21)]
    oos_dates = [f'2022-06-{d:02d}' for d in range(1, 21)]
    report = {'historical_is': {'dates': is_dates}, 'historical_oos': {'dates': oos_dates, 'pnls': {'happo': [0.01] * len(oos_dates)}}, 'historical_calendar': {'is_dates': is_dates, 'is_pnls': [-0.02] * len(is_dates), 'oos_dates': oos_dates, 'oos_pnls': [0.01] * len(oos_dates)}}
    out = attach_publication_stats(report, cfg={'seed': 0, 'publication_planned_folds': 1})
    gfc = next((r for r in out['regime_performance']['regimes'] if r['id'] == 'gfc_2008'))
    assert gfc['available'] is True
    assert gfc['n_days'] >= 2
    assert np.isfinite(gfc['sharpe'])

def test_attach_dual_series_dsr_and_spa_role():
    from src.eval.publication import attach_publication_stats
    rng = np.random.default_rng(0)
    is_n, oos_n = (40, 60)
    is_pnls = rng.normal(0.001, 0.01, size=is_n).tolist()
    oos_pnls = rng.normal(0.002, 0.01, size=oos_n).tolist()
    report = {'historical_calendar': {'is_dates': [f'2020-01-{d:02d}' for d in range(1, is_n + 1)], 'is_pnls': is_pnls, 'oos_dates': [f'2021-01-{d:02d}' for d in range(1, oos_n + 1)], 'oos_pnls': oos_pnls}, 'historical_oos': {'dates': [f'2021-01-{d:02d}' for d in range(1, oos_n + 1)], 'pnls': {'happo': oos_pnls}}, 'baselines': {'pnls': {'garch11': rng.normal(0.0005, 0.01, size=oos_n).tolist(), 'short_vol_carry': rng.normal(0.0003, 0.01, size=oos_n).tolist(), 'iv_rank_timing': rng.normal(0.0002, 0.01, size=oos_n).tolist(), 'heston_iv_momentum': rng.normal(0.0001, 0.01, size=oos_n).tolist()}}}
    out = attach_publication_stats(report, cfg={'seed': 0, 'publication_planned_folds': 2})
    pooled = out['deflated_sharpe_pooled']
    oos = out['deflated_sharpe_oos']
    assert pooled['series'] == 'pooled_is_oos'
    assert oos['series'] == 'oos'
    assert oos['n_obs'] == oos_n
    assert pooled['n_obs'] == is_n + oos_n
    assert out['bootstrap_cis_pooled']['series'] == 'pooled_is_oos'
    assert out['bootstrap_cis_oos']['series'] == 'oos'
    spa = out['hansen_spa']
    assert spa.get('benchmark_role') == 'happo'
    assert spa.get('ok') is True
    assert int(spa.get('n_economic_rivals') or 0) >= 3
    assert 'SPA proves' in str(spa.get('do_not_claim') or '')

def test_pbo_appendix_and_trial_ledger():
    from src.eval.pbo_appendix import build_trial_ledger, probability_of_backtest_overfitting
    sharpes = [0.5, 1.2, -0.3, 0.8, 0.1, 2.0, 0.4, -0.1]
    pbo = probability_of_backtest_overfitting(sharpes, n_partitions=32, seed=1)
    assert pbo['n_trials'] == 8
    assert pbo['nested_wfo_is_not_cpcv'] is True
    assert 0.0 <= pbo['pbo'] <= 1.0
    ledger = build_trial_ledger(ablation_rows=[{'id': 'full', 'status': 'ok', 'oos_sharpe': 1.0}], plugin_rows=[{'id': 'honesty', 'status': 'ok', 'oos_sharpe_net': 0.9}], nested_fold_sharpes=[0.5, 0.6, 0.4, 0.7], multiseed_sharpes=[1.1, 0.9])
    assert ledger['n_trials_listed'] >= 4
    assert 'pbo_appendix' in ledger
    assert ledger['pbo_appendix']['nested_wfo_is_not_cpcv'] is True

def test_garch_forecast_positive():
    r = np.random.default_rng(1).normal(0, 0.02, size=100)
    f = _garch11_forecast(r)
    assert f.shape == r.shape
    assert np.all(f > 0)

def test_baseline_weights_project():
    k = 8
    proj = ConvexProjectionLayer(k, turnover_limit=0.15)
    atm = np.linspace(0.2, 0.4, k)
    hist = np.tile(atm, (32, 1)) + 0.01 * np.random.randn(32, k)
    deltas = np.full(k, 0.5)
    w_prev = torch.zeros(1, k)
    hv = np.linspace(0.15, 0.45, k)
    skew = np.linspace(-0.05, 0.15, k)
    for name in BASELINE_NAMES:
        w = baseline_weights_day(name, atm_row=atm, atm_hist=hist, deltas=deltas, w_prev=w_prev, projector=proj, vol_scale=0.25, skew_row=skew, hv_row=hv)
        assert w.shape == (1, k)
        assert torch.isfinite(w).all()

def test_literature_baseline_sign_conventions():
    from src.eval.baselines import _baseline_raw_signal, rolling_hv_from_returns
    k = 4
    deltas = np.zeros(k)
    atm_low = np.full(k, 0.15)
    hv_high = np.full(k, 0.4)
    hist = np.tile(atm_low, (40, 1))
    w = _baseline_raw_signal('goyal_saretto_hv_iv', atm_row=atm_low, atm_hist=hist, deltas=deltas, hv_row=hv_high)
    assert float(w.sum()) > 0.0
    w2 = _baseline_raw_signal('goyal_saretto_hv_iv', atm_row=np.full(k, 0.4), atm_hist=np.tile(np.full(k, 0.4), (40, 1)), deltas=deltas, hv_row=np.full(k, 0.15))
    assert float(w2.sum()) < 0.0
    series_up = np.linspace(0.1, 0.5, 40)
    hist_ivr = np.column_stack([series_up] * k)
    w_ivr = _baseline_raw_signal('iv_rank_timing', atm_row=np.full(k, 0.5), atm_hist=hist_ivr, deltas=deltas)
    assert float(w_ivr.sum()) < 0.0
    w_flat = _baseline_raw_signal('timed_long_gamma', atm_row=np.full(k, 0.4), atm_hist=hist, deltas=deltas, hv_row=np.full(k, 0.1))
    assert float(np.abs(w_flat).sum()) < 1e-08
    skew = np.array([0.0, 0.0, 0.0, 0.5])
    w_sk = _baseline_raw_signal('skew_risk_reversal', atm_row=np.full(k, 0.2), atm_hist=hist, deltas=deltas, skew_row=skew)
    assert w_sk[3] < w_sk[0]
    hv = np.array([0.1, 0.2, 0.3, 0.5])
    w_ch = _baseline_raw_signal('cao_han_high_ivol', atm_row=np.full(k, 0.2), atm_hist=hist, deltas=deltas, hv_row=hv)
    assert w_ch[3] < w_ch[0]
    rets = np.random.default_rng(0).normal(0, 0.01, size=(100, k))
    hv_row = rolling_hv_from_returns(rets, t=80, lookback=60)
    assert hv_row.shape == (k,)
    assert np.isfinite(hv_row).all()
    assert rolling_hv_from_returns.__doc__ and 'never ΔIV' in rolling_hv_from_returns.__doc__

def test_baseline_suite_eight_modes_synthetic():
    from src.eval.baselines import run_baseline_suite_on_panel
    rng = np.random.default_rng(0)
    n, k = (80, 6)
    atm = 0.2 + 0.05 * rng.random((n, k))
    deltas = rng.normal(0, 0.2, size=(n, k))
    fwd = rng.normal(0, 0.01, size=(n, k))
    skew = rng.normal(0, 0.05, size=(n, k))
    urets = rng.normal(0, 0.015, size=(n, k))
    dates = [f'2022-01-01'] * n
    import pandas as pd
    dates = list(pd.date_range('2022-01-01', periods=n, freq='B'))
    suite = run_baseline_suite_on_panel(atm=atm, deltas_np=deltas, fwd=fwd, dates=dates, seq_len=16, skew=skew, underlier_rets=urets, underlier_meta={'ok': True, 'coverage': 1.0})
    assert len(suite['modes']) == 8
    assert set(suite['modes']) == set(BASELINE_NAMES)
    for m in BASELINE_NAMES:
        assert m in suite['summary']
        assert suite['summary'][m]['n_days'] > 0
        assert np.isfinite(suite['summary'][m]['sharpe']) or suite['summary'][m].get('unavailable')

def test_ablation_hooks_forward():
    k, d, m = (4, 8, 4)
    fe_m = AlphaFeatureExtractor(k, d, d_state=4, temporal_backend='mamba', use_dhgnn=True)
    fe_g = AlphaFeatureExtractor(k, d, d_state=4, temporal_backend='gru', use_dhgnn=False)
    raw = torch.randn(1, k, 16, d)
    iv = torch.rand(1, k)
    z1 = fe_m(raw, iv)
    z2 = fe_g(raw, iv)
    assert z1.shape == (1, k, d)
    assert z2.shape == (1, k, d)
    pol = HAPPOEngine(k, d, m, use_projection=False)
    w, _, _, _ = pol.act_stochastic(z1, torch.zeros(1, m), torch.zeros(1, k), torch.randn(1, k), vol_scale=0.2)
    assert w.shape == (1, k)

def test_limitations_and_plots(tmp_path: Path):
    report = {'historical_oos': {'pnls': {'happo': [0.01, -0.005, 0.002] * 40, 'zero': [0.0] * 120}, 'dates': [f'2022-01-{i % 28 + 1:02d}' for i in range(120)], 'turnovers': {'happo': [0.1] * 120}, 'summary': {'happo': {'mean_pnl': 0.002, 'sharpe': 1.0}}}, 'baselines': {'pnls': {'short_vol_carry': [0.0] * 120, 'garch_vol_timing': [0.001] * 120}, 'summary': {'short_vol_carry': {'sharpe': 0.1, 'mean_pnl': 0.0, 'mean_turnover': 0.1}, 'garch_vol_timing': {'sharpe': 0.5, 'mean_pnl': 0.001, 'mean_turnover': 0.1}}}, 'deflated_sharpe': {'psr': 0.9, 'dsr': 0.8, 'n_trials': 10, 'significant_05': False, 'citation': 'Bailey & López de Prado (2014), JPM — Deflated Sharpe Ratio'}, 'regime_performance': {'regimes': [{'id': 'hike_2022', 'label': '2022', 'n_days': 10, 'sharpe': 0.5, 'max_drawdown': -0.1, 'mean_turnover': 0.1}]}, 'ablations': {'rows': [{'spec': {'id': 'full'}, 'sharpe': 1.0, 'positive_fold_rate': 0.8}, {'spec': {'id': 'no_cmdp'}, 'sharpe': 0.2, 'positive_fold_rate': 0.4}]}}
    (tmp_path / 'KNOWN_LIMITATIONS.md').write_text('Arm-1 train/eval macro version caveat (fixture).\n')
    lim = write_limitations_section(tmp_path, report)
    assert lim.is_file()
    text = lim.read_text()
    assert 'Limitations' in text
    assert 'Run-specific caveats' in text
    assert 'Arm-1 train/eval macro version caveat' in text
    written = plot_publication_figures(report, tmp_path / 'plots')
    assert any(('30_baseline' in p for p in written))
    assert any(('31_ablation' in p for p in written))
    assert any(('34_psr_dsr' in p for p in written))

def test_cmdp_env_macro_is_integer_indexed():
    """Functional: ``_macro_at`` returns row-position values, not calendar/as_of lookups."""
    from src.env.cmdp_env import CMDPEnv
    from src.features.extractor import AlphaFeatureExtractor
    from src.policy.happo import HAPPOEngine
    torch.manual_seed(0)
    k, t_steps, d, mdim = (4, 16, 8, 4)
    surfaces = torch.rand(2, k, t_steps, 5, 3) * 0.2 + 0.1
    n_macro = 40
    macro = torch.zeros(n_macro, mdim)
    macro[:, 0] = torch.arange(n_macro, dtype=torch.float32)
    fe = AlphaFeatureExtractor(k, d, d_state=4)
    pol = HAPPOEngine(k, d, mdim, use_projection=False)
    env = CMDPEnv(surfaces, fe, pol, d, mdim, use_gpu=False, macro_series=macro, seq_len=4)
    env.reset(path=0, start_t=3)
    env.macro_start_idx = 10
    env.t = 3
    got = env._macro_at()
    assert float(got[0, 0]) == pytest.approx(13.0, **FLOAT_TOL)
    env.t = 7
    got = env._macro_at()
    assert float(got[0, 0]) == pytest.approx(17.0, **FLOAT_TOL)
    env.macro_start_idx = n_macro - 2
    env.t = 50
    got = env._macro_at()
    assert float(got[0, 0]) == float(n_macro - 1)

def test_macro_window_seed_sync_across_rng_divergence():
    """Same episode_seed → same macro window even after unequal global RNG draws."""
    from src.env.cmdp_env import CMDPEnv
    from src.features.extractor import AlphaFeatureExtractor
    from src.policy.happo import HAPPOEngine
    k, t_steps, d, mdim = (4, 16, 8, 4)
    surfaces = torch.rand(2, k, t_steps, 5, 3) * 0.2 + 0.1
    macro = torch.randn(80, mdim)
    fe = AlphaFeatureExtractor(k, d, d_state=4)
    pol = HAPPOEngine(k, d, mdim, use_projection=False)
    env_a = CMDPEnv(surfaces, fe, pol, d, mdim, use_gpu=False, macro_series=macro, seq_len=4)
    env_b = CMDPEnv(surfaces, fe, pol, d, mdim, use_gpu=False, macro_series=macro, seq_len=4)
    torch.manual_seed(0)
    env_a.reset(path=0, start_t=1, episode_seed=42000007)
    idx_a = env_a.macro_start_idx
    torch.manual_seed(0)
    _ = torch.randn(1000)
    env_b.reset(path=0, start_t=1, episode_seed=42000007)
    assert env_b.macro_start_idx == idx_a

def test_episode_env_noise_sync_across_rng_divergence():
    """Same episode_seed → same spot Brownian path despite global RNG divergence."""
    from src.env.cmdp_env import CMDPEnv
    from src.features.extractor import AlphaFeatureExtractor
    from src.policy.happo import HAPPOEngine
    k, t_steps, d, mdim = (4, 20, 8, 4)
    torch.manual_seed(1)
    surfaces = torch.rand(1, k, t_steps, 5, 3) * 0.2 + 0.1
    macro = torch.randn(100, mdim)
    fe = AlphaFeatureExtractor(k, d, d_state=4)
    pol = HAPPOEngine(k, d, mdim, use_projection=False)
    env_a = CMDPEnv(surfaces.clone(), fe, pol, d, mdim, use_gpu=False, macro_series=macro, seq_len=4)
    env_b = CMDPEnv(surfaces.clone(), fe, pol, d, mdim, use_gpu=False, macro_series=macro, seq_len=4)
    seed = 99000001
    torch.manual_seed(0)
    env_a.reset(path=0, start_t=2, episode_seed=seed)
    spots_a = []
    w = torch.zeros(1, k)
    for _ in range(4):
        env_a.step(w)
        spots_a.append(env_a.spot.clone())
    torch.manual_seed(0)
    _ = torch.randn(2000)
    env_b.reset(path=0, start_t=2, episode_seed=seed)
    for i in range(4):
        env_b.step(w)
        assert torch.allclose(env_b.spot, spots_a[i]), i

def test_macro_window_held_for_full_episode():
    """macro_start_idx must not be resampled mid-episode (stable regime context)."""
    from src.env.cmdp_env import CMDPEnv
    from src.features.extractor import AlphaFeatureExtractor
    from src.policy.happo import HAPPOEngine
    k, t_steps, d, mdim = (4, 20, 8, 4)
    surfaces = torch.rand(1, k, t_steps, 5, 3) * 0.2 + 0.1
    macro = torch.randn(100, mdim)
    fe = AlphaFeatureExtractor(k, d, d_state=4)
    pol = HAPPOEngine(k, d, mdim, use_projection=False)
    env = CMDPEnv(surfaces, fe, pol, d, mdim, use_gpu=False, macro_series=macro, seq_len=4)
    env.reset(path=0, start_t=2, episode_seed=123)
    fixed = env.macro_start_idx
    w = torch.zeros(1, k)
    for _ in range(5):
        env.step(w)
        assert env.macro_start_idx == fixed
