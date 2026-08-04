"""End-to-end feature wiring: groups, excludes, slice, no nan_to_num."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.eval.research_alpha_cpcv import _slice_feature_extras
from src.features.blocks.assemble import assemble_equity_feature_cube
from src.features.groups import FEATURE_GROUPS, resolve_excludes


def test_resolve_excludes_unknown_group_fails() -> None:
    with pytest.raises(ValueError, match="unknown feature_groups_exclude"):
        resolve_excludes(["not_a_group"], [], ["log_ret_1"])


def test_resolve_excludes_drops_group() -> None:
    names = list(FEATURE_GROUPS["core_volatility"]) + ["log_ret_1"]
    drop = resolve_excludes(["core_volatility"], [], names)
    assert drop == set(FEATURE_GROUPS["core_volatility"]) & set(names)


def test_assemble_exclude_group_removes_channels() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=(50, 4))
    cube_full, names_full = assemble_equity_feature_cube(r)
    cube, names = assemble_equity_feature_cube(
        r,
        {"feature_groups_exclude": ["core_volatility"]},
    )
    for n in FEATURE_GROUPS["core_volatility"]:
        if n in names_full:
            assert n not in names
    assert cube.shape[-1] == len(names)
    assert cube.shape[-1] < cube_full.shape[-1]


def test_borrow_rate_collision_renamed() -> None:
    t, k = 30, 2
    r = np.zeros((t, k))
    borrow = np.ones((t, k)) * 0.01
    iv = {"borrow_rate": np.ones((t, k)) * 0.02}
    cube, names = assemble_equity_feature_cube(
        r,
        {"iv_surface": iv, "borrow": borrow, "include_borrow": True},
        normalize=False,
    )
    assert "borrow_rate" in names
    assert "borrow_rate_fee" in names
    assert len(names) == len(set(names))


def test_slice_feature_extras_new_keys() -> None:
    t, k = 20, 3
    cfg = {
        "feature_extras": {
            "ohlc": {"open": np.ones((t, k)), "high": np.ones((t, k))},
            "sentiment": {"si_pct": np.arange(t * k, dtype=float).reshape(t, k)},
            "jkp": {"log_me": np.ones((t, k))},
            "option_flow": {"pc_vol": np.ones((t, k))},
            "microstructure": {"eff_spread": np.ones((t, k))},
            "fundamentals_pit": {"bm": np.ones((t, k))},
            "macro_names": ["a"],
        }
    }
    idx = np.arange(5, 15)
    out = _slice_feature_extras(cfg, idx)
    ex = out["feature_extras"]
    assert ex["ohlc"]["open"].shape[0] == 10
    assert ex["sentiment"]["si_pct"].shape == (10, k)
    assert ex["jkp"]["log_me"].shape[0] == 10


def test_slot_mask_nan_before_normalize_with_new_blocks() -> None:
    t, k = 40, 3
    r = np.random.default_rng(0).normal(0, 0.01, size=(t, k))
    mask = np.ones((t, k), dtype=bool)
    mask[:, -1] = False
    ohlc = {
        "open": np.abs(r) + 1,
        "high": np.abs(r) + 1.1,
        "low": np.abs(r) + 0.9,
        "close": np.abs(r) + 1,
        "adj_close": np.abs(r) + 1,
    }
    cube, names = assemble_equity_feature_cube(
        r,
        {"ohlc": ohlc, "slot_valid_mask": mask},
        normalize=True,
    )
    assert np.isnan(cube[:, -1, :]).all()


def test_no_nan_to_num_in_new_blocks() -> None:
    root = pathlib.Path("src/features/blocks")
    new_blocks = {
        "range_volatility.py",
        "microstructure.py",
        "fundamentals_pit.py",
        "sentiment.py",
        "option_flow.py",
        "jkp_lottery.py",
        "experimental.py",
    }
    offenders = []
    for name in new_blocks:
        text = (root / name).read_text()
        if "nan_to_num" in text:
            offenders.append(name)
    assert offenders == []
