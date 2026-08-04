"""A-4: single SoT for wide-returns helpers in equity_substrate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _fixture_panel(*, n_dates: int = 12, n_names: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    rows: list[dict] = []
    for d in dates:
        for sid in range(1, n_names + 1):
            rows.append({"date": d, "secid": sid, "return": 0.001 * sid})
    df = pd.DataFrame(rows)
    # Row 5: one NaN (5% of 20 names) — at max_row_nan_frac=0.05 boundary.
    drop_idx = df.index[(df["date"] == dates[5]) & (df["secid"] == 20)][0]
    df = df.drop(drop_idx)
    # Row 7: two NaNs (10%) — must drop under puncture mode.
    for sid in (19, 20):
        drop_idx = df.index[(df["date"] == dates[7]) & (df["secid"] == sid)][0]
        df = df.drop(drop_idx)
    return df


def test_wide_returns_max_row_nan_frac_puncture_mode() -> None:
    from src.eval.equity_substrate import _wide_returns_with_availability

    df = _fixture_panel()
    dates = pd.bdate_range("2020-01-02", periods=12)
    rets, secids, idx, avail = _wide_returns_with_availability(
        df,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        min_cov=0.1,
        ffill_limit=0,
        max_row_nan_frac=0.05,
        keep_partial_rows=False,
    )
    assert dates[5] not in idx
    assert dates[7] not in idx
    assert dates[0] in idx
    assert avail.all()


def test_wide_returns_keep_partial_rows_preserves_sparse_rows() -> None:
    from src.eval.equity_substrate import _wide_returns_with_availability

    df = _fixture_panel()
    dates = pd.bdate_range("2020-01-02", periods=12)
    rets, secids, idx, avail = _wide_returns_with_availability(
        df,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        min_cov=0.1,
        ffill_limit=0,
        keep_partial_rows=True,
    )
    assert len(idx) == len(dates)
    assert dates[7] in idx
    j = list(secids).index(20)
    row7 = int(np.where(idx == dates[7])[0][0])
    assert not avail[row7, j]


@pytest.mark.parametrize("keep_partial_rows", [True, False])
def test_eq_alloc_reexports_match_equity_substrate(keep_partial_rows: bool) -> None:
    from scripts.run_eq_alloc_campaign import (
        _wide_returns_with_availability as eq_wide,
    )
    from src.eval.equity_substrate import (
        _wide_returns_with_availability as sub_wide,
    )

    assert eq_wide is sub_wide

    df = _fixture_panel()
    dates = pd.bdate_range("2020-01-02", periods=12)
    kw = dict(
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        min_cov=0.1,
        ffill_limit=0,
        max_row_nan_frac=0.05,
        keep_partial_rows=keep_partial_rows,
    )
    rets, secids, idx, avail = eq_wide(df, **kw)
    if keep_partial_rows:
        assert len(idx) == len(dates)
        assert dates[7] in idx
    else:
        assert dates[7] not in idx
    assert rets.shape == avail.shape == (len(idx), len(secids))
