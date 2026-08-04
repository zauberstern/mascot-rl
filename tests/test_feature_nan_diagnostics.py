"""RED: feature NaN diagnostics must fail closed above 20% all-NaN dates."""
from __future__ import annotations

import numpy as np
import pytest


def test_feature_nan_diagnostics_pass_at_threshold() -> None:
    from src.eval.feature_nan_diagnostics import feature_nan_diagnostics

    # 100 dates, 5 names, 1 channel; exactly 20 all-NaN dates
    cube = np.ones((100, 5, 1), dtype=np.float64)
    cube[:20, :, 0] = np.nan
    diag = feature_nan_diagnostics(
        cube, channel_names=["mfis_30"], admitted_channels=["mfis_30"], max_all_nan_frac=0.20
    )
    assert diag["pass"] is True
    assert diag["per_channel"]["mfis_30"]["all_nan_frac"] == pytest.approx(0.20)


def test_feature_nan_diagnostics_fail_above_threshold() -> None:
    from src.eval.feature_nan_diagnostics import (
        assert_feature_nan_ok,
        feature_nan_diagnostics,
    )

    cube = np.ones((100, 5, 1), dtype=np.float64)
    cube[:21, :, 0] = np.nan
    diag = feature_nan_diagnostics(
        cube, channel_names=["mfis_30"], admitted_channels=["mfis_30"], max_all_nan_frac=0.20
    )
    assert diag["pass"] is False
    with pytest.raises(SystemExit, match="mfis_30"):
        assert_feature_nan_ok(diag)
