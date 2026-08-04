"""Dynamic universe arms must carry the rolling slot-masked PIT stamp."""
from __future__ import annotations

import pandas as pd

from scripts.run_eq_alloc_campaign import _stamp_dynamic_arm_pit
from src.features.pit_universe import ROLLING_TRAILING_PIT


def test_dynamic_arm_stamps_slot_masked_pit_and_universe_mode() -> None:
    cfg: dict = {}
    info = {"arm": "dyn_liquidity"}
    dates = list(pd.bdate_range("2022-01-03", periods=5))

    _stamp_dynamic_arm_pit(info, cfg=cfg, dates=dates)

    assert cfg["universe_mode"] == ROLLING_TRAILING_PIT
    assert info["pit"]["pit_clean"] is True
    assert info["pit"]["universe_protocol"] == "slot_masked"
    assert info["pit"]["eval_start"] == "2022-01-03"
