"""Locking tests for literature-standard OptionMetrics chain screens (W2).

Screens follow Cao and Han (2013, JFE 108(1)) Section 2, Goyal and Saretto
(2009), and the contract-standardization filters of the Dallas Fed IPCA study
(WP 2214).
"""
from __future__ import annotations

import duckdb
import pytest

from src.data.duckdb_engine import OptionFilterConfig

EXPECTED_SCREENS = {
    "iv_present",
    "volume_positive",
    "mid_above_tick",
    "moneyness_band",
    "no_arbitrage_bounds",
    "standard_settlement",
    "common_stock",
    "not_index_option",
    "no_dividend_in_life",
}


def test_default_config_enables_literature_screens():
    names = {n for n, _ in OptionFilterConfig().screens()}
    assert names == EXPECTED_SCREENS
    assert {n for n, _ in OptionFilterConfig().selection_screens()} == {"calls_only"}


def test_disabled_config_has_no_screens():
    cfg = OptionFilterConfig.disabled()
    assert cfg.screens() == []
    assert cfg.selection_screens() == []


def test_cao_han_thresholds_are_the_published_ones():
    cfg = OptionFilterConfig()
    assert (cfg.moneyness_lo, cfg.moneyness_hi) == (0.8, 1.2)
    # Cao-Han drop quotes with mid below one eighth of a dollar.
    assert cfg.min_mid == pytest.approx(0.125)
    assert cfg.calls_only is True
    assert (cfg.dte_lo, cfg.dte_hi) == (14, 45)


def _eval_predicates(rows: list[dict], cfg: OptionFilterConfig) -> list[bool]:
    """Evaluate the real SQL predicates against in-memory rows."""
    con = duckdb.connect()
    cols = list(rows[0].keys())
    con.execute(
        "CREATE TABLE q ("
        + ", ".join(
            f"{c} " + ("VARCHAR" if isinstance(rows[0][c], str) else "DOUBLE")
            for c in cols
        )
        + ")"
    )
    con.executemany(
        f"INSERT INTO q VALUES ({', '.join('?' for _ in cols)})",
        [[r[c] for c in cols] for r in rows],
    )
    pred = " AND ".join(f"({p})" for _, p in cfg.screens() + cfg.selection_screens())
    got = con.execute(f"SELECT COALESCE(({pred}), FALSE) FROM q").fetchall()
    return [bool(g[0]) for g in got]


def _row(**kw):
    base = {
        "impl_volatility": 0.25,
        "volume": 100.0,
        "mid": 2.0,
        "spot": 100.0,
        "strike": 100.0,
        "cp_flag": "C",
        "ss_flag": 0.0,
        "issue_type": "0",
        "index_flag": 0.0,
        "no_dividend_in_life": 1.0,
    }
    base.update(kw)
    return base


def test_screens_accept_a_clean_atm_call():
    assert _eval_predicates([_row()], OptionFilterConfig()) == [True]


def test_each_screen_rejects_its_violation():
    cfg = OptionFilterConfig()
    cases = {
        "missing iv": _row(impl_volatility=None),
        "zero volume": _row(volume=0.0),
        "sub-tick mid": _row(mid=0.05),
        "deep OTM moneyness": _row(spot=70.0),
        "deep ITM moneyness": _row(spot=130.0),
        "call above spot": _row(mid=150.0),
        "call below intrinsic": _row(spot=120.0, strike=100.0, mid=5.0),
        "non-standard settlement": _row(ss_flag=1.0),
        "not common stock": _row(issue_type="7"),
        "index option": _row(index_flag=1.0),
        "dividend in life": _row(no_dividend_in_life=0.0),
        "is a put": _row(cp_flag="P"),
    }
    got = _eval_predicates(list(cases.values()), cfg)
    failures = [name for name, ok in zip(cases, got) if ok]
    assert not failures, f"screens failed to reject: {failures}"


def test_no_arb_bounds_keep_legitimate_itm_call():
    """An ITM call priced above intrinsic must survive the bounds screen."""
    row = _row(spot=110.0, strike=100.0, mid=11.5)
    assert _eval_predicates([row], OptionFilterConfig()) == [True]


def test_puts_survive_quality_screens_when_calls_only_excluded():
    """Skew needs both wings, so quality screens alone must not drop puts."""
    cfg = OptionFilterConfig()
    con = duckdb.connect()
    pred = " AND ".join(f"({p})" for _, p in cfg.screens())
    r = _row(cp_flag="P", mid=2.0, spot=100.0, strike=100.0)
    cols = list(r.keys())
    con.execute(
        "CREATE TABLE q ("
        + ", ".join(
            f"{c} " + ("VARCHAR" if isinstance(r[c], str) else "DOUBLE") for c in cols
        )
        + ")"
    )
    con.execute(
        f"INSERT INTO q VALUES ({', '.join('?' for _ in cols)})", [r[c] for c in cols]
    )
    assert bool(con.execute(f"SELECT COALESCE(({pred}), FALSE) FROM q").fetchone()[0])
