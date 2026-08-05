"""Kelly IV image materialization must be resumable via on-disk cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_kelly_images_cache_hit_skips_lake(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mascotrl.data import surface_signals as ss

    dates = pd.date_range("2014-01-01", periods=5, freq="B")
    secids = [101, 102]
    cache = tmp_path / "kelly_images.npz"

    def _empty_surface(*_a, **_k):
        return pd.DataFrame(
            columns=["secid", "date", "days", "delta", "cp_flag", "impl_volatility"]
        )

    monkeypatch.setattr(ss, "_load_vol_surface_raw", _empty_surface)
    first = ss.materialize_kelly_iv_images_from_lake(
        "/unused",
        secids=secids,
        dates=dates,
        start="2014-01-01",
        end="2014-01-10",
        cache_path=cache,
    )
    assert cache.is_file()
    assert Path(str(cache) + ".meta.json").is_file()

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("lake reload on kelly cache hit")

    monkeypatch.setattr(ss, "_load_vol_surface_raw", _boom)
    second = ss.materialize_kelly_iv_images_from_lake(
        "/unused",
        secids=secids,
        dates=dates,
        start="2014-01-01",
        end="2014-01-10",
        cache_path=cache,
    )
    assert calls["n"] == 0
    np.testing.assert_array_equal(first, second)
    n_del = len(ss.KELLY_DELTAS_PUT) + len(ss.KELLY_DELTAS_CALL)
    assert first.shape == (5, 2, len(ss.KELLY_TENORS), n_del)
