"""Arm-faithful claim_label_stem resolution for spectrum panel loading."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.arms.spec import EQUITY_LABEL_STEM, OPTION_LABEL_STEM
from mascotrl.arms.training import resolve_claim_label_stem
from mascotrl.data.oos_panel import label_matrix


def test_eq_resolves_stk_ret():
    assert resolve_claim_label_stem({"portfolio_arm": "eq", "n_assets": 10}) == EQUITY_LABEL_STEM


def test_opt_resolves_dh_ret_lagdelta():
    assert resolve_claim_label_stem({"portfolio_arm": "opt", "n_assets": 10}) == OPTION_LABEL_STEM


def test_mix_has_no_single_stem():
    with pytest.raises(ValueError, match="mix"):
        resolve_claim_label_stem({"portfolio_arm": "mix", "n_assets": 10})


def test_explicit_stem_must_match_arm():
    with pytest.raises(ValueError, match="claim_label_stem"):
        resolve_claim_label_stem(
            {
                "portfolio_arm": "eq",
                "n_assets": 8,
                "claim_label_stem": OPTION_LABEL_STEM,
            }
        )


def test_explicit_matching_stem_ok():
    stem = resolve_claim_label_stem(
        {
            "portfolio_arm": "opt",
            "n_assets": 8,
            "claim_label_stem": OPTION_LABEL_STEM,
        }
    )
    assert stem == OPTION_LABEL_STEM


def test_label_matrix_raises_on_missing_equity_stem():
    """Any missing stem must raise (not silently fall back to dh_ret)."""
    T, K = 4, 3
    cols = {f"{OPTION_LABEL_STEM}_{i}": np.zeros(T) for i in range(K)}
    df = pd.DataFrame(cols)
    with pytest.raises(KeyError, match=EQUITY_LABEL_STEM):
        label_matrix(df, K, stem=EQUITY_LABEL_STEM)
