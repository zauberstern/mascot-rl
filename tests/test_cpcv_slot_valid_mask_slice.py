"""A-5: _slice_feature_extras must slice or fail on _slot_valid_mask."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_slice_feature_extras_slices_full_panel_slot_mask() -> None:
    from src.eval.research_alpha_cpcv import _slice_feature_extras

    t, k = 40, 5
    mask = np.arange(t * k, dtype=float).reshape(t, k) > 0
    dates = list(pd.bdate_range("2020-01-01", periods=t))
    idx = np.arange(10, 25)
    out = _slice_feature_extras(
        {"_slot_valid_mask": mask, "_dates": dates},
        idx,
    )
    sliced = np.asarray(out["_slot_valid_mask"], dtype=bool)
    np.testing.assert_array_equal(sliced, mask[idx])
    assert sliced.shape == (len(idx), k)


def test_slice_feature_extras_raises_on_slot_mask_t_mismatch() -> None:
    from src.eval.research_alpha_cpcv import _slice_feature_extras

    t, k = 22, 4
    mask = np.ones((t, k), dtype=bool)
    idx = np.arange(0, 25)
    with pytest.raises(ValueError, match="_slot_valid_mask.*mismatch"):
        _slice_feature_extras({"_slot_valid_mask": mask}, idx)


def test_slice_feature_extras_leaves_none_slot_mask() -> None:
    from src.eval.research_alpha_cpcv import _slice_feature_extras

    idx = np.arange(0, 10)
    out = _slice_feature_extras({}, idx)
    assert out.get("_slot_valid_mask") is None
