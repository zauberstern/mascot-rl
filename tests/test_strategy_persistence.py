"""D2: per-strategy weights/turnover/cost parquet + combined ledger export."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mascotrl.reporting.strategy_persistence import (
    build_accounting_ledger,
    persist_strategy_frames,
    strategy_frame,
)


def _toy_frame(n: int = 6, k: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    w = rng.dirichlet(np.ones(k), size=n)
    return strategy_frame(
        dates=dates,
        secids=[f"S{i}" for i in range(k)],
        weights=w,
        turnover=rng.uniform(0, 0.5, size=n),
        cost=rng.uniform(0, 0.01, size=n),
        gross=rng.normal(0, 0.02, size=n),
        total_net=rng.normal(0, 0.02, size=n),
        residual=rng.normal(0, 0.02, size=n),
    )


def test_strategy_frame_has_expected_columns_and_row_count():
    df = _toy_frame(n=5, k=4)
    assert len(df) == 5
    for col in ("date", "turnover", "cost", "gross", "total_net", "residual"):
        assert col in df.columns
    for i in range(4):
        assert f"w_S{i}" in df.columns
    # Dirichlet rows sum to 1.
    w_cols = [f"w_S{i}" for i in range(4)]
    np.testing.assert_allclose(df[w_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_strategy_frame_empty_weights_returns_empty_df_with_columns():
    df = strategy_frame(
        dates=[], secids=["A", "B"], weights=np.zeros((0, 2)), turnover=[], cost=[], gross=[]
    )
    assert df.empty
    assert "date" in df.columns


def test_strategy_frame_rejects_mismatched_date_length():
    with pytest.raises(ValueError, match="dates length"):
        strategy_frame(
            dates=["2021-01-04"],
            secids=["A"],
            weights=np.zeros((2, 1)),
            turnover=[0.0, 0.0],
            cost=[0.0, 0.0],
            gross=[0.0, 0.0],
        )


def test_strategy_frame_missing_total_net_is_nan_not_crash():
    df = strategy_frame(
        dates=pd.bdate_range("2021-01-04", periods=3),
        secids=["A"],
        weights=np.ones((3, 1)),
        turnover=[0.1, 0.1, 0.1],
        cost=[0.0, 0.0, 0.0],
        gross=[0.0, 0.0, 0.0],
    )
    assert df["total_net"].isna().all()
    assert df["residual"].isna().all()


def test_persist_strategy_frames_writes_one_parquet_per_nonempty_strategy(tmp_path: Path):
    frames = {
        "equal_weight": _toy_frame(n=4, k=2, seed=1),
        "empty_strategy": pd.DataFrame(columns=["date"]),
        "olps:pamr": _toy_frame(n=4, k=2, seed=2),
    }
    written = persist_strategy_frames(frames, tmp_path)
    assert set(written.keys()) == {"equal_weight", "olps:pamr"}
    # Colon in name must be sanitized for the filesystem.
    assert (tmp_path / "olps__pamr.parquet").is_file()
    assert (tmp_path / "equal_weight.parquet").is_file()
    round_trip = pd.read_parquet(tmp_path / "equal_weight.parquet")
    assert len(round_trip) == 4


def test_build_accounting_ledger_combines_multiple_strategies():
    frames = {
        "policy": _toy_frame(n=3, k=2, seed=3),
        "equal_weight": _toy_frame(n=3, k=2, seed=4),
    }
    ledger = build_accounting_ledger(secids=["S0", "S1"], frames=frames)
    df = ledger.to_dataframe()
    assert set(df["strategy"].unique()) == {"policy", "equal_weight"}
    # 3 dates x 2 assets x 2 strategies.
    assert len(df) == 3 * 2 * 2
    # Executed weight for each (date, strategy) pair sums to ~1 (dirichlet source).
    pivot = df.pivot_table(
        index=["date", "strategy"], values="weight_exec", aggfunc="sum"
    )
    np.testing.assert_allclose(pivot["weight_exec"].to_numpy(), 1.0, atol=1e-8)


def test_build_accounting_ledger_skips_strategy_with_no_weight_columns():
    frames = {
        "policy": _toy_frame(n=2, k=2, seed=5),
        "no_weights": pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=2), "cost": [0.0, 0.0]}),
    }
    ledger = build_accounting_ledger(secids=["S0", "S1"], frames=frames)
    df = ledger.to_dataframe()
    assert set(df["strategy"].unique()) == {"policy"}
