"""Causal three-state regime labeller tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.regime_labels import REGIME_IDS, label_regimes


def _macro_frame(n: int = 900, *, vix=None, oas=None, infl=None) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "vix_level": vix if vix is not None else 15.0 + rng.normal(0, 1.0, n),
            "hy_oas_level": oas if oas is not None else 4.0 + rng.normal(0, 0.2, n),
            "inflation_yoy_level": infl if infl is not None else 2.0 + rng.normal(0, 0.1, n),
        },
        index=dates,
    )
    return df


def test_labels_causal_when_future_deleted():
    df = _macro_frame(900)
    labels, meta = label_regimes(df, min_history_days=100, persistence_days=5)
    t_cut = df.index[500]
    labels2, _ = label_regimes(df.loc[:t_cut], min_history_days=100, persistence_days=5)
    assert labels.loc[:t_cut].tolist() == labels2.tolist()


def test_warmup_flagged_calm():
    df = _macro_frame(200)
    labels, meta = label_regimes(df, min_history_days=120, persistence_days=5)
    warm = meta["warmup"]
    assert warm.iloc[:120].all()
    assert (labels.iloc[:120] == "calm").all()
    assert not warm.iloc[120:].all()


def test_short_vix_spike_no_flip():
    n = 400
    rng = np.random.default_rng(0)
    # Noisy baseline so constant-series ties do not mark every day as crisis
    vix = 15.0 + rng.normal(0, 0.5, n)
    oas = 3.0 + rng.normal(0, 0.05, n)
    infl = 2.0 + rng.normal(0, 0.05, n)
    vix[200:203] = 80.0
    df = _macro_frame(n, vix=vix, oas=oas, infl=infl)
    labels, _ = label_regimes(
        df,
        min_history_days=100,
        persistence_days=10,
        crisis_vix_q=0.85,
        crisis_oas_q=0.85,
        infl_q=0.70,
    )
    pre = labels.iloc[190:200]
    assert (pre != "crisis").all(), "pre-spike window should not already be crisis"
    # 3-day spike must not flip sticky label under 10-day persistence
    assert (labels.iloc[200:210] == pre.iloc[-1]).all()


def test_sustained_spike_flips():
    n = 400
    rng = np.random.default_rng(1)
    vix = 15.0 + rng.normal(0, 0.5, n)
    oas = 3.0 + rng.normal(0, 0.05, n)
    infl = 2.0 + rng.normal(0, 0.05, n)
    vix[200:230] = 80.0
    df = _macro_frame(n, vix=vix, oas=oas, infl=infl)
    labels, meta = label_regimes(
        df,
        min_history_days=100,
        persistence_days=10,
        crisis_vix_q=0.85,
        crisis_oas_q=0.85,
        infl_q=0.70,
    )
    assert (labels.iloc[190:200] != "crisis").all()
    # After 10 consecutive crisis raw days, sticky label must switch
    assert (labels.iloc[209:220] == "crisis").all()
    assert "crisis" in set(labels.unique())
    assert set(REGIME_IDS) >= set(labels.unique())


def test_occupancy_sums_non_warmup():
    df = _macro_frame(800)
    labels, meta = label_regimes(df, min_history_days=100, persistence_days=5)
    active = labels.loc[~meta["warmup"].to_numpy()]
    counts = active.value_counts()
    assert int(counts.sum()) == len(active)
    assert set(counts.index).issubset(set(REGIME_IDS))
    assert len(REGIME_IDS) == 3


def test_regime_meta_percentiles_reproduce_raw_rule():
    df = _macro_frame(500)
    labels, meta = label_regimes(
        df,
        min_history_days=80,
        persistence_days=1,  # sticky == raw for audit of percentile rule
        crisis_vix_q=0.85,
        crisis_oas_q=0.85,
        infl_q=0.70,
    )
    for i, ts in enumerate(df.index):
        if bool(meta.loc[ts, "warmup"]):
            assert labels.loc[ts] == "calm"
            continue
        vix_pct = float(meta.loc[ts, "vix_pct"])
        oas_pct = float(meta.loc[ts, "hy_oas_pct"])
        infl_pct = float(meta.loc[ts, "inflation_pct"])
        if vix_pct >= 0.85 or oas_pct >= 0.85:
            expect = "crisis"
        elif infl_pct >= 0.70:
            expect = "inflationary"
        else:
            expect = "calm"
        assert labels.loc[ts] == expect, (ts, labels.loc[ts], expect)
