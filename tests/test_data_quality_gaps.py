"""D2: quote audit counters, fresh-quote screen, Kelly schema validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.duckdb_engine import OptionFilterConfig, quote_quality_audit_counts
from src.data.surface_signals import (
    KELLY_DELTAS_CALL,
    KELLY_DELTAS_PUT,
    KELLY_TENORS,
    validate_kelly_grid_schema,
)


def test_default_screens_unchanged_without_fresh_quotes():
    names = {n for n, _ in OptionFilterConfig().screens()}
    assert "fresh_quotes" not in names


def test_require_fresh_quotes_adds_screen():
    cfg = OptionFilterConfig(require_fresh_quotes=True)
    screens = dict(cfg.screens())
    assert "fresh_quotes" in screens
    assert "last_date" in screens["fresh_quotes"]


def test_quote_quality_audit_counts():
    df = pd.DataFrame(
        {
            "best_bid": [1.0, 2.0, 0.0],
            "best_offer": [1.5, 1.0, 1.0],  # row1 crossed, row2 bid<=0
            "volume": [10, 0, 5],
            "impl_volatility": [0.2, np.nan, 0.3],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-02"]),
            "last_date": pd.to_datetime(["2020-01-02", "2019-12-31", "2020-01-03"]),
        }
    )
    counts = quote_quality_audit_counts(df)
    assert counts["n_rows"] == 3
    assert counts["n_crossed_bid_ask"] >= 1
    assert counts["n_zero_volume"] == 1
    assert counts["n_missing_iv"] == 1
    assert counts["n_stale_last_date"] == 1


def test_quote_quality_audit_empty():
    assert quote_quality_audit_counts(pd.DataFrame())["n_rows"] == 0


def test_validate_kelly_grid_schema_ok():
    meta = validate_kelly_grid_schema(cube_shape=(10, 5, 11, 34))
    assert meta["n_tenors"] == 11
    assert meta["n_deltas"] == 34
    assert meta["ffill_causal"] is True


def test_validate_kelly_grid_schema_rejects_bad_axes():
    with pytest.raises(ValueError):
        validate_kelly_grid_schema(tenors=(10, 30))
    with pytest.raises(ValueError):
        validate_kelly_grid_schema(cube_shape=(10, 5, 11, 30))


def test_kelly_constants_stable():
    assert KELLY_TENORS[0] == 10 and KELLY_TENORS[-1] == 730
    assert KELLY_DELTAS_PUT[0] == -90
    assert KELLY_DELTAS_CALL[-1] == 90


def test_marks_base_sql_includes_last_date_for_fresh_quotes():
    """require_fresh_quotes screen needs last_date in the marks CTE SELECT."""
    from src.data.duckdb_engine import DuckDBFeatureEngine

    eng = DuckDBFeatureEngine.__new__(DuckDBFeatureEngine)
    eng.lake_base_dir = type("P", (), {"as_posix": lambda self: "/tmp"})()
    # Minimal stubs so _options_glob / helpers are not hit hard
    eng._options_glob = lambda y: f"/tmp/opprcd/year={y}/*.parquet"  # type: ignore
    eng._sec_prices_parquet = lambda: "/tmp/sec.parquet"  # type: ignore
    eng._rates_parquet = lambda: "/tmp/rates.parquet"  # type: ignore
    eng._crsp_prices_parquet = lambda: "/tmp/crsp.parquet"  # type: ignore
    eng._dividends_parquet = lambda: "/tmp/div.parquet"  # type: ignore

    cfg = OptionFilterConfig(require_fresh_quotes=True)
    sql = eng._marks_base_sql("1", "2020-01-01", "2020-01-31", cfg)
    assert "AS last_date" in sql or "last_date" in sql
    assert "fresh_quotes" in dict(cfg.screens())


def test_require_fresh_quotes_duckdb_inmemory_no_raise():
    """Enabling require_fresh_quotes must not fail when last_date is selected."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE marks AS
        SELECT
            CAST('2020-01-02' AS DATE) AS date,
            CAST('2020-01-02' AS DATE) AS last_date,
            1.0 AS mid
        UNION ALL
        SELECT
            CAST('2020-01-03' AS DATE),
            CAST('2019-12-01' AS DATE),
            1.0
        """
    )
    cfg = OptionFilterConfig(require_fresh_quotes=True)
    pred = dict(cfg.screens())["fresh_quotes"]
    # Predicate uses last_date / date; must execute without BinderException
    n = con.execute(f"SELECT COUNT(*) FROM marks WHERE {pred}").fetchone()[0]
    assert n == 1  # only the fresh row
    con.close()


def test_build_kelly_iv_images_validates_schema():
    from src.data.surface_signals import build_kelly_iv_images

    empty = pd.DataFrame(
        columns=["date", "secid", "days", "delta", "cp_flag", "impl_volatility"]
    )
    cube = build_kelly_iv_images(
        empty,
        secids=[1],
        dates=[pd.Timestamp("2020-01-02")],
    )
    assert cube.shape == (1, 1, len(KELLY_TENORS), len(KELLY_DELTAS_PUT) + len(KELLY_DELTAS_CALL))

    with pytest.raises(ValueError):
        build_kelly_iv_images(
            empty,
            secids=[1],
            dates=[pd.Timestamp("2020-01-02")],
            tenors=(10, 30),
        )
